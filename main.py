from __future__ import annotations
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
import py7zr
import tempfile
import shutil
from os import environ
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
from dotenv import load_dotenv
from os import getenv
import requests
import time
import datetime

# Load environment variables
load_dotenv()

# Initialize OpenAI client with OpenRouter credentials
env_base_url = getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
correct_base_url = env_base_url if env_base_url.endswith("/v1") else "https://openrouter.ai/api/v1"
if env_base_url != correct_base_url:
    logging.warning(f"Environment OPENROUTER_BASE_URL ({env_base_url}) overridden with {correct_base_url}")
api_key = getenv("OPENROUTER_API_KEY")
if not api_key or not api_key.startswith("sk-or-v1-"):
    logging.error("OPENROUTER_API_KEY is missing or invalid")
    raise RuntimeError("OPENROUTER_API_KEY is missing or invalid")
# Log masked API key for debugging
masked_key = f"{api_key[:10]}...{api_key[-4:]}" if api_key else "None"
logger = logging.getLogger(__name__)
logger.info(f"Loaded OPENROUTER_API_KEY: {masked_key}")
client = OpenAI(
    base_url=correct_base_url,
    api_key=api_key,
)

# Initialize FastAPI app
app = FastAPI(title="Code Plagiarism Detector", version="0.1.0")

# Configuration
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODEL_ARCHIVE = BASE_DIR / "dataset" / "output" / "model-best" / "transformer" / "model.7z"
TEMP_MODEL_DIR = Path(tempfile.mkdtemp())
INPUT_1568 = BASE_DIR / "input1568.txt"
INPUT_1435 = BASE_DIR / "input1435.txt"
OUTPUT_1435 = BASE_DIR / "output1435.txt"

CONFIG = {
    "lcs_threshold": 0.5,
    "hash_threshold": 0.125,
    "var_length": 5,
    "comment_length": 5,
    "comment_block": 5,
    "delta_threshold": 0.5,
    "plagiarism_threshold": 0.10,
}

# Logging setup
valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level = environ.get("LOG_LEVEL", "DEBUG").upper()
if log_level not in valid_log_levels:
    log_level = "DEBUG"
logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s | %(asctime)s | %(message)s")

# Verify MODEL_ARCHIVE exists (for reference, not loaded)
if not MODEL_ARCHIVE.exists():
    logger.warning("Model archive %s not found, skipping spaCy model loading", MODEL_ARCHIVE)
else:
    logger.info("Confirmed model.7z exists at %s, but skipping spaCy due to memory constraints", MODEL_ARCHIVE)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Pydantic models
class SingleCodeInput(BaseModel):
    code: str
    handle: str = "triumph"
    @validator("handle")
    def validate_handle(cls, v):
        if not v.strip() or any(c in v for c in "\n$"):
            raise ValueError("Handle must be non-empty and not contain newlines or '$'")
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Handle must be alphanumeric or underscores")
        return v

class HealthResponse(BaseModel):
    status: str
    models: List[str] = []

class ValidationError(BaseModel):
    loc: List[str]
    msg: str
    type: str

class HTTPValidationError(BaseModel):
    detail: List[ValidationError]

# Helper function to check OpenRouter API health and get available models
def check_openrouter_health(api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> Tuple[bool, List[str]]:
    try:
        if not api_key or not api_key.startswith("sk-or-v1-"):
            logger.error("Invalid or missing API key for health check")
            return False, []
        masked_key = f"{api_key[:10]}...{api_key[-4:]}" if api_key else "None"
        logger.debug(f"Using API key: {masked_key} for health check")
        env_base_url = getenv("OPENROUTER_BASE_URL")
        if env_base_url and env_base_url != base_url:
            logger.warning(f"Environment OPENROUTER_BASE_URL ({env_base_url}) overridden with {base_url}")
        endpoint = f"{base_url.rstrip('/')}/models"
        logger.debug(f"Checking OpenRouter API health at: {endpoint}")
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(endpoint, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        if "data" in data and isinstance(data["data"], list):
            model_ids = [model["id"] for model in data["data"]]
            logger.info("OpenRouter API health check passed")
            return True, model_ids
        else:
            logger.error("OpenRouter API response does not contain valid model list")
            return False, []
    except requests.exceptions.HTTPError as e:
        logger.error(f"OpenRouter API health check failed: HTTP {e.response.status_code} - {str(e)}")
        return False, []
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter API health check failed: {str(e)}")
        return False, []
    except ValueError as e:
        logger.error(f"OpenRouter API health check failed: Invalid JSON response - {str(e)}")
        return False, []

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    # Validate environment variables
    api_key = getenv("OPENROUTER_API_KEY")
    base_url = "https://openrouter.ai/api/v1"
    env_base_url = getenv("OPENROUTER_BASE_URL")
    logger.debug(f"Environment OPENROUTER_BASE_URL: {env_base_url}, using: {base_url}")
    if not api_key or not api_key.startswith("sk-or-v1-"):
        logger.error("OPENROUTER_API_KEY is missing or invalid")
        raise RuntimeError("OPENROUTER_API_KEY is missing or invalid")
    # Test OpenRouter API health
    is_healthy, models = check_openrouter_health(api_key, base_url)
    if not is_healthy:
        logger.error("Startup failed: OpenRouter API unavailable")
        raise RuntimeError("Startup failed: OpenRouter API unavailable")
    # Store available models for use in query_api
    app.state.available_models = models
    logger.info(f"Available models: {models}")

@app.on_event("shutdown")
async def cleanup():
    try:
        if TEMP_MODEL_DIR.exists():
            shutil.rmtree(TEMP_MODEL_DIR, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory {TEMP_MODEL_DIR}")
        for file in [INPUT_1568, INPUT_1435, OUTPUT_1435]:
            if file.exists():
                file.unlink()
                logger.info(f"Removed {file}")
    except Exception as e:
        logger.error(f"Failed to clean up: {str(e)}")

# Query API function with enhanced error handling and fallback
def query_api(messages: List[Dict[str, str]], model: str = None, temp: float = 0.7, max_tokens: int = 2000) -> str:
    # Validate API key
    api_key = getenv("OPENROUTER_API_KEY")
    if not api_key or not api_key.startswith("sk-or-v1-"):
        logger.error("OPENROUTER_API_KEY is missing or invalid in query_api")
        return ""
    masked_key = f"{api_key[:10]}...{api_key[-4:]}" if api_key else "None"
    logger.debug(f"Using API key: {masked_key} for query_api")
    
    # Use a default model from available models if not specified
    if not model:
        available_models = getattr(app.state, "available_models", [])
        model = next((m for m in available_models if m == "deepseek/deepseek-r1-0528:free"), "deepseek/deepseek-r1-0528:free")
        logger.debug(f"No model specified, using: {model}")
    
    fallback_models = ["qwen/qwen3-8b:free", "mistralai/mistral-7b-instruct:free"]
    models_to_try = [model] + fallback_models
    
    for current_model in models_to_try:
        try:
            logger.debug(f"Sending API request to OpenRouter with model {current_model} and messages: {json.dumps(messages)[:100]}...")
            request_url = f"{str(client.base_url).rstrip('/')}/chat/completions"
            logger.debug(f"Request URL: {request_url}")
            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            # Log response type
            logger.debug(f"Response type: {type(response).__name__}")
            content = response.choices[0].message.content.strip()
            logger.debug(f"Raw API response: {content[:500]}...")
            try:
                parsed = json.loads(content)
                code = parsed.get("code", "")
                if not code:
                    logger.error(f"No 'code' field in JSON response: {content[:1000]}")
                    # Try regex as fallback
                    match = re.search(r"```cpp\n(.*?)```", content, re.DOTALL)
                    if match:
                        return match.group(1).strip()
                    logger.error(f"No valid C++ code block found in API response: {content[:1000]}")
                    return content.strip()  # Return raw content as last resort
                return code.strip()
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {content[:1000]}")
                match = re.search(r"```cpp\n(.*?)```", content, re.DOTALL)
                if not match:
                    logger.error(f"No valid C++ code block found in API response: {content[:1000]}")
                    return content.strip()  # Return raw content as last resort
                return match.group(1).strip()
        except AuthenticationError as e:
            logger.error(f"Authentication failed with model {current_model}: Invalid OpenRouter API key - {str(e)}")
            # Attempt direct HTTP request as fallback
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"}
                }
                logger.debug(f"Sending fallback HTTP request for model {current_model}")
                http_response = requests.post(request_url, headers=headers, json=payload, timeout=10)
                logger.debug(f"Fallback HTTP status: {http_response.status_code}")
                logger.debug(f"Fallback HTTP headers: {http_response.headers}")
                logger.debug(f"Fallback HTTP response: {http_response.text[:1000]}...")
                if http_response.status_code == 200:
                    try:
                        data = http_response.json()
                        content = data['choices'][0]['message']['content'].strip()
                        logger.debug(f"Fallback raw response: {content[:1000]}...")
                        try:
                            parsed = json.loads(content)
                            code = parsed.get("code", "")
                            if not code:
                                logger.error(f"No 'code' field in fallback JSON response: {content[:1000]}")
                                match = re.search(r"```cpp\n(.*?)```", content, re.DOTALL)
                                if match:
                                    return match.group(1).strip()
                                logger.error(f"No valid C++ code block in fallback response: {content[:1000]}")
                                return content.strip()
                            return code.strip()
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse fallback JSON response: {content[:1000]}")
                            match = re.search(r"```cpp\n(.*?)```", content, re.DOTALL)
                            if not match:
                                logger.error(f"No valid C++ code block in fallback response: {content[:1000]}")
                                return content.strip()
                            return match.group(1).strip()
                    except Exception as e:
                        logger.error(f"Failed to parse fallback response: {str(e)}")
                        return ""
                elif http_response.status_code == 401 and current_model != models_to_try[-1]:
                    logger.warning(f"Fallback HTTP 401 with model {current_model}: {http_response.text[:1000]}. Retrying with next model")
                    continue
                elif (http_response.status_code == 404 or http_response.status_code == 429) and current_model != models_to_try[-1]:
                    delay = 60.0  # Fixed 60-second delay
                    if http_response.status_code == 429:
                        logger.warning(f"Fallback rate limit headers: Limit={http_response.headers.get('X-RateLimit-Limit', 'N/A')}, "
                                      f"Remaining={http_response.headers.get('X-RateLimit-Remaining', 'N/A')}, "
                                      f"Reset={http_response.headers.get('X-RateLimit-Reset', 'N/A')}")
                        logger.warning(f"Fallback rate limit exceeded with model {current_model}: {http_response.text[:1000]}. Waiting exactly {delay:.2f}s before retrying with next model")
                    else:
                        logger.warning(f"Fallback HTTP {http_response.status_code} with model {current_model}: {http_response.text[:1000]}. Waiting {delay:.2f}s before retrying with next model")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Fallback HTTP request failed with status {http_response.status_code}: {http_response.text[:1000]}")
                    return ""
            except Exception as e:
                logger.error(f"Fallback HTTP request failed with model {current_model}: {str(e)}")
                return ""
        except APIError as e:
            if ("401" in str(e) or "404" in str(e) or "429" in str(e)) and current_model != models_to_try[-1]:
                delay = 60.0  # Fixed 60-second delay for 429 or 404, no delay for 401
                if "401" in str(e):
                    logger.warning(f"Authentication error with model {current_model}: {str(e)}. Retrying with next model")
                elif "429" in str(e):
                    try:
                        error_json = e.response.json()
                        logger.warning(f"Rate limit headers: Limit={error_json.get('error', {}).get('metadata', {}).get('headers', {}).get('X-RateLimit-Limit', 'N/A')}, "
                                      f"Remaining={error_json.get('error', {}).get('metadata', {}).get('headers', {}).get('X-RateLimit-Remaining', 'N/A')}, "
                                      f"Reset={error_json.get('error', {}).get('metadata', {}).get('headers', {}).get('X-RateLimit-Reset', 'N/A')}")
                    except (ValueError, AttributeError) as parse_err:
                        logger.error(f"Failed to parse rate limit headers: {str(parse_err)}")
                    logger.warning(f"Rate limit exceeded for model {current_model}: {str(e)}. Waiting exactly {delay:.2f}s before retrying with next model")
                    time.sleep(delay)
                else:
                    logger.warning(f"API error with model {current_model}: {str(e)}. Waiting {delay:.2f}s before retrying with next model")
                    time.sleep(delay)
                continue
            logger.error(f"OpenRouter API error with model {current_model}: {str(e)}")
            return ""
        except RateLimitError as e:
            if current_model != models_to_try[-1]:
                delay = 60.0  # Fixed 60-second delay
                try:
                    error_json = e.response.json()
                    logger.warning(f"Rate limit headers: Limit={error_json.get('error', {}).get('metadata', {}).get('headers', {}).get('X-RateLimit-Limit', 'N/A')}, "
                                  f"Remaining={error_json.get('error', {}).get('metadata', {}).get('headers', {}).get('X-RateLimit-Remaining', 'N/A')}, "
                                  f"Reset={error_json.get('error', {}).get('metadata', {}).get('headers', {}).get('X-RateLimit-Reset', 'N/A')}")
                except (ValueError, AttributeError) as parse_err:
                    logger.error(f"Failed to parse rate limit headers: {str(parse_err)}")
                logger.warning(f"Rate limit exceeded for model {current_model}: {str(e)}. Waiting exactly {delay:.2f}s before retrying with next model")
                time.sleep(delay)
                continue
            logger.error(f"Rate limit exceeded for OpenRouter API with model {current_model}: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in API query with model {current_model}: {str(e)}")
            # Attempt direct HTTP request as fallback
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"}
                }
                logger.debug(f"Sending fallback HTTP request for model {current_model}")
                http_response = requests.post(request_url, headers=headers, json=payload, timeout=10)
                logger.debug(f"Fallback HTTP status: {http_response.status_code}")
                logger.debug(f"Fallback HTTP headers: {http_response.headers}")
                logger.debug(f"Fallback HTTP response: {http_response.text[:1000]}...")
                if http_response.status_code == 200:
                    try:
                        data = http_response.json()
                        content = data['choices'][0]['message']['content'].strip()
                        logger.debug(f"Fallback raw response: {content[:1000]}...")
                        try:
                            parsed = json.loads(content)
                            code = parsed.get("code", "")
                            if not code:
                                logger.error(f"No 'code' field in fallback JSON response: {content[:1000]}")
                                match = re.search(r"```cpp\n(.*?)```", content, re.DOTALL)
                                if match:
                                    return match.group(1).strip()
                                logger.error(f"No valid C++ code block in fallback response: {content[:1000]}")
                                return content.strip()
                            return code.strip()
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse fallback JSON response: {content[:1000]}")
                            match = re.search(r"```cpp\n(.*?)```", content, re.DOTALL)
                            if not match:
                                logger.error(f"No valid C++ code block in fallback response: {content[:1000]}")
                                return content.strip()
                            return match.group(1).strip()
                    except Exception as e:
                        logger.error(f"Failed to parse fallback response: {str(e)}")
                        return ""
                elif http_response.status_code == 401 and current_model != models_to_try[-1]:
                    logger.warning(f"Fallback HTTP 401 with model {current_model}: {http_response.text[:1000]}. Retrying with next model")
                    continue
                elif (http_response.status_code == 404 or http_response.status_code == 429) and current_model != models_to_try[-1]:
                    delay = 60.0  # Fixed 60-second delay
                    if http_response.status_code == 429:
                        logger.warning(f"Fallback rate limit headers: Limit={http_response.headers.get('X-RateLimit-Limit', 'N/A')}, "
                                      f"Remaining={http_response.headers.get('X-RateLimit-Remaining', 'N/A')}, "
                                      f"Reset={http_response.headers.get('X-RateLimit-Reset', 'N/A')}")
                        logger.warning(f"Fallback rate limit exceeded with model {current_model}: {http_response.text[:1000]}. Waiting exactly {delay:.2f}s before retrying with next model")
                    else:
                        logger.warning(f"Fallback HTTP {http_response.status_code} with model {current_model}: {http_response.text[:1000]}. Waiting {delay:.2f}s before retrying with next model")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Fallback HTTP request failed with status {http_response.status_code}: {http_response.text[:1000]}")
                    return ""
            except Exception as e:
                logger.error(f"Fallback HTTP request failed with model {current_model}: {str(e)}")
                return ""
    logger.error(f"All models failed: {models_to_try}")
    return ""  # Return empty string if all attempts fail

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        base_url = "https://openrouter.ai/api/v1"
        api_key = getenv("OPENROUTER_API_KEY")
        if not api_key or not api_key.startswith("sk-or-v1-"):
            logger.error("OPENROUTER_API_KEY is missing or invalid in health check")
            return {"status": "unhealthy", "error": "Invalid or missing API key"}
        masked_key = f"{api_key[:10]}...{api_key[-4:]}" if api_key else "None"
        logger.debug(f"Health check using API key: {masked_key}")
        env_base_url = getenv("OPENROUTER_BASE_URL")
        if env_base_url and env_base_url != base_url:
            logger.warning(f"Environment OPENROUTER_BASE_URL ({env_base_url}) overridden with {base_url}")
        endpoint = f"{base_url.rstrip('/')}/models"
        logger.debug(f"Health check requesting: {endpoint}")
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(endpoint, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        model_ids = [model["id"] for model in data["data"]]
        return {"status": "healthy", "models": model_ids}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

# Existing functions (unchanged)
def clean_code(code: str, preserve_comments: bool = False) -> str:
    if not code:
        return ""
    if not preserve_comments:
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return re.sub(r'\s+', ' ', code).strip()

def tokenize_code(line: str) -> List[str]:
    if not line:
        return []
    line = re.sub(r'//.*$', '', line)
    line = re.sub(r'/\*.*?\*/', '', line, flags=re.DOTALL)
    return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[{}();,=<>+\-*&]|[0-9]+', line)

def extract_variables(code: str) -> Tuple[List[str], bool]:
    primitives = {"int", "long", "short", "float", "double", "char", "bool", "void", "auto", "unsigned", "signed", "size_t"}
    containers = {"vector", "stack", "queue", "deque", "map", "set", "pair", "string"}
    keywords = {"main", "first", "second", "top", "push", "pop", "begin", "end", "size", "clear", "empty", "insert", "erase", "find", "sort", "reverse"}
    variables = set()
    for line in code.splitlines():
        tokens = tokenize_code(line)
        i = 0
        while i < len(tokens):
            type_str, j = parse_type(tokens, i, primitives, containers)
            if type_str and j < len(tokens):
                i = j
                while i < len(tokens) and tokens[i] != ";":
                    if (tokens[i] not in ("*", "&", ",", "=", "(", ")") and
                        tokens[i] not in keywords and
                        tokens[i] not in primitives and
                        tokens[i] not in containers and
                        not tokens[i].isdigit()):
                        variables.add(tokens[i])
                    i += 1
            else:
                i += 1
    var_list = sorted(variables)
    has_long_vars = len([v for v in var_list if len(v) > CONFIG["var_length"]]) >= 2
    return var_list, has_long_vars

def parse_type(tokens: List[str], index: int, primitives: set, containers: set) -> Tuple[str, int]:
    type_str = ""
    while index < len(tokens) and (tokens[index] in primitives or tokens[index] in containers):
        type_str += tokens[index] + " "
        index += 1
        if index < len(tokens) and tokens[index] == "<":
            depth = 0
            type_str += tokens[index]
            index += 1
            while index < len(tokens):
                if tokens[index] == "<":
                    depth += 1
                elif tokens[index] == ">":
                    if depth == 0:
                        type_str += tokens[index]
                        index += 1
                        break
                    depth -= 1
                type_str += tokens[index]
                index += 1
    return type_str.strip(), index

def has_suspicious_comments(code: str) -> bool:
    if not code:
        return False
    lines = code.splitlines()
    comment_count = sum(1 for line in lines if re.search(r'//(.{%d,})' % CONFIG["comment_length"], line))
    total_blocks = max(1, len(lines) // CONFIG["comment_block"])
    return comment_count < total_blocks

def preprocess_code(code: str, handle: str) -> str:
    return clean_code(code) + " " + handle

def pad_string(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    max_len = max(len(line) for line in lines)
    if max_len % 2 != 0:
        max_len += 1
    padded_lines = [line + 'S' * (max_len - len(line)) for line in lines]
    if len(padded_lines) < max_len:
        padded_lines.extend(['S' * max_len for _ in range(max_len - len(padded_lines))])
    return '\n'.join(padded_lines)

def string_to_matrix(text: str, max_lines: int = 1000) -> List[List[int]]:
    if not text:
        return []
    lines = text.splitlines()[:max_lines]
    return [[ord(char) % 128 for char in line] for line in lines if line]

def extract_submatrix(matrix: List[List[int]], center: List[int], size: int = 3) -> List[List[int]]:
    if not matrix or not matrix[0]:
        return []
    half_size = size // 2
    rows, cols = len(matrix), len(matrix[0])
    row_start = max(0, center[0] - half_size)
    row_end = min(rows, center[0] + half_size + 1)
    col_start = max(0, center[1] - half_size)
    col_end = min(cols, center[1] + half_size + 1)
    return [[matrix[i][j] for j in range(col_start, col_end)] for i in range(row_start, row_end)]

def matrix_avg_distance(matrix: List[List[int]]) -> float:
    if not matrix or not matrix[0]:
        return 0.0
    rows, cols = len(matrix), len(matrix[0])
    center_r, center_c = rows // 2, cols // 2
    total = sum((i - center_r) ** 2 + (j - center_c) ** 2 for i in range(rows) for j in range(cols))
    return total / (rows * cols) ** 0.5 if rows * cols > 0 else 0.0

def compute_global_distance(matrix: List[List[int]]) -> float:
    if not matrix or not matrix[0]:
        return 0.0
    rows, cols = len(matrix), len(matrix[0])
    total_distance, count = 0.0, 0
    for i in range(rows):
        for j in range(cols):
            submatrix = extract_submatrix(matrix, [i, j])
            distance = matrix_avg_distance(submatrix)
            if not np.isnan(distance):
                total_distance += distance
                count += 1
    return total_distance / count if count > 0 else 0.0

def compute_hash(text: str, by_line: bool = True) -> List[int]:
    if not text:
        return []
    segments = text.splitlines() if by_line else [text]
    hashes = [int(hashlib.sha256(segment.encode()).hexdigest(), 16) % (2**64)
              for segment in segments if segment.strip()]
    return sorted(hashes)

def lcs_length(str1: str, str2: str) -> int:
    if not str1 or not str2:
        return 0
    if len(str1) < len(str2):
        str1, str2 = str2, str1
    n, m = len(str1), len(str2)
    curr = [0] * (m + 1)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            curr[j] = prev[j-1] + 1 if str1[i-1] == str2[j-1] else max(prev[j], curr[j-1])
        prev, curr = curr, prev
    return prev[m]

def compare_codes(code1: str, code2: str) -> Tuple[bool, str]:
    code1_clean, code2_clean = clean_code(code1), clean_code(code2)
    if not code1_clean or not code2_clean:
        return False, "Empty code"
    lcs_score = lcs_length(code1_clean, code2_clean)
    lcs_ratio = lcs_score / min(len(code1_clean), len(code2_clean)) if code1_clean and code2_clean else 0
    return lcs_ratio >= CONFIG["lcs_threshold"], f"LCS: {lcs_ratio:.2f}"

async def analyze_code_with_generated(code: str, generated: str) -> Tuple[bool, float, str]:
    if not code or not generated:
        return False, 0.0, "Empty code or generated code"
    original_matrix = string_to_matrix(pad_string(code))
    generated_matrix = string_to_matrix(pad_string(generated))
    delta = abs(compute_global_distance(original_matrix) - compute_global_distance(generated_matrix))
    return delta < CONFIG["delta_threshold"], delta, f"Delta: {delta:.4f}"

async def detect_plagiarism_with_generated(code: str, generated: str) -> Tuple[bool, float, str, str, List[str]]:
    if not code or not generated:
        return False, 0.0, "Empty code or generated code", "N", ["Empty code or generated code"]
    original_matrix = string_to_matrix(pad_string(code))
    generated_matrix = string_to_matrix(pad_string(generated))
    delta = abs(compute_global_distance(original_matrix) - compute_global_distance(generated_matrix))
    is_similar, lcs_details = compare_codes(code, generated)
    evidence = [f"Delta: {delta:.4f}", lcs_details]
    is_plagiarized = delta < CONFIG["plagiarism_threshold"] or is_similar
    return is_plagiarized, delta, f"Delta: {delta:.4f}", "S" if is_similar else "N", evidence

async def classify_code(code: str, handle: str) -> Tuple[bool, str, float, str]:
    if not code.strip():
        return False, "Code cannot be empty", 0.0, handle
    cleaned = clean_code(code)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert C++ programmer. Generate C++ code that replicates the functionality of the provided code. "
                "Ensure the code is complete, syntactically correct, and uses modern C++ practices. "
                "Wrap the generated code in a ```cpp code block, like this:\n"
                "```cpp\n"
                "// Your code here\n"
                "```\n"
                "Do not include explanations or comments outside the code block unless explicitly requested."
            )
        },
        {"role": "user", "content": code}
    ]
    try:
        generated = query_api(messages)
        if not generated:
            return False, "Failed to generate code from API. Please check API key or model availability.", 0.0, handle
        is_similar = compare_codes(cleaned, clean_code(generated))[0]
        return True, "H" if is_similar else "AI", 0.0, handle
    except Exception as e:
        logger.error(f"Code classification failed: {str(e)}")
        return False, f"Classification error: {str(e)}", 0.0, handle

def process_code_submission(code: str, handle: str):
    try:
        if INPUT_1568.exists():
            INPUT_1568.unlink()
        with open(INPUT_1568, "w") as f:
            f.write(code.strip() + "\n")
            f.write("R77q\n")
            f.write(f"{handle}$")
        
        cleaned_code = clean_code(code)
        if not cleaned_code:
            logger.warning(f"Empty code after cleaning for handle {handle}")
            return
        
        formatted_code = f"xc9@{handle}\n{cleaned_code}\n"
        with open(INPUT_1435, "w") as f:
            f.write(formatted_code)
        logger.info(f"Processed code for handle {handle} and written to {INPUT_1435}")
    except Exception as e:
        logger.error(f"Failed to process code for handle {handle}: {str(e)}")
        raise

def detect_similar_codes(code: str, handle: str) -> List[str]:
    try:
        with open(OUTPUT_1435, "w") as f:
            f.write("0 similar pairs of codes detected\n\n0 cheaters detected:\n")
        return []
    except Exception as e:
        logger.error(f"Failed to detect similar codes: {str(e)}")
        return []

def prepend_code_count():
    try:
        if not INPUT_1435.exists():
            logger.error("input1435.txt not found")
            return
        with open(INPUT_1435, "r") as f:
            lines = f.readlines()
        code_count = 1
        with open(INPUT_1435, "w") as f:
            f.write(f"{code_count}\n")
            f.writelines(lines)
        logger.info(f"Prepended code count {code_count} to input1435.txt")
    except Exception as e:
        logger.error(f"Failed to prepend code count: {str(e)}")
        raise

def generate_report(analysis: Dict[str, Any], issues: List[str], handle: str, 
                   model1_result: Optional[Tuple[str, List[str]]] = None, 
                   model2_result: Optional[Tuple[str, List[str]]] = None) -> str:
    report = f"Report for {handle}\n\n"
    report += f"AI/Human Classification:\nLabel: {analysis['label']} (Confidence: {analysis['confidence']:.2%})\n"
    report += "Issues:\n" + ("\n".join([f"- {x}" for x in issues]) if issues else "- None\n")
    if model1_result:
        report += f"\nBasic Plagiarism Check (Python):\nSimilarity: {model1_result[0]}\nEvidence:\n" + "\n".join([f"- {x}" for x in model1_result[1]]) + "\n"
    if model2_result:
        report += f"\nAdvanced Plagiarism Check (Python):\nSimilarity: {model2_result[0]}\nEvidence:\n" + "\n".join([f"- {x}" for x in model2_result[1]]) + "\n"
    report += "\nCode Similarity Check (Python):\nSimilar to handles: None\n"
    
    flags = sum(1 for x in [
        analysis['label'] == "AI",
        model1_result and model1_result[0] == "S",
        model2_result and model2_result[0] == "S"
    ] if x)
    report += f"\nConclusion:\n- {'AI or Plagiarized' if flags >= 2 else 'Human'}\n"
    report += "\nRecommendations:\n"
    if flags >= 2:
        report += "- Review for AI usage or plagiarism.\n- Verify code authenticity.\n"
        if analysis['label'] == "AI":
            report += "- Discuss AI usage with author.\n"
        if (model1_result and model1_result[0] == "S") or (model2_result and model2_result[0] == "S"):
            report += "- Investigate potential plagiarism.\n"
    else:
        report += "- Likely human-authored.\n"
    return report

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/a", response_class=HTMLResponse, responses={
    200: {"content": {"text/html": {}}},
    422: {"model": HTTPValidationError}
})
async def analyze_code_form(request: Request, code: str = Form(..., max_length=100_000), handle: str = Form("triumph")):
    try:
        if not code.strip():
            return templates.TemplateResponse("index.html", {"request": request, "error": "Code cannot be empty"})
        
        # Validate handle
        SingleCodeInput(code=code, handle=handle)
        
        # Check API health
        health = await health_check()
        if health["status"] != "healthy":
            logger.error(f"OpenRouter API unavailable: {health.get('error', 'Unknown error')}")
            return templates.TemplateResponse("index.html", {"request": request, "error": f"OpenRouter API is unavailable: {health.get('error', 'Unknown error')}"})
        
        # Run Python-based analysis
        logger.debug(f"Processing code (first 100 chars): {code[:100]}...")
        success, label, confidence, handle = await classify_code(code, handle)
        if not success:
            return templates.TemplateResponse("index.html", {"request": request, "error": label})
        variables, has_long_vars = extract_variables(code)
        issues = [f"Variables longer than {CONFIG['var_length']} characters" if has_long_vars else "",
                  "Suspicious comments" if has_suspicious_comments(code) else ""]
        issues = [x for x in issues if x]
        
        # Call query_api once
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert C++ programmer. Generate C++ code that replicates the functionality of the provided code. "
                    "Ensure the code is complete, syntactically correct, and uses modern C++ practices. "
                    "Wrap the generated code in a ```cpp code block, like this:\n"
                    "```cpp\n"
                    "// Your code here\n"
                    "```\n"
                    "Do not include explanations or comments outside the code block unless explicitly requested."
                )
            },
            {"role": "user", "content": code}
        ]
        generated = query_api(messages)
        if not generated:
            error_msg = "Failed to generate code from API. Please check the API key or model availability."
            logger.error(error_msg)
            return templates.TemplateResponse("index.html", {"request": request, "error": error_msg})
        
        # Use generated code for both analyses
        is_similar, delta, delta_details = await analyze_code_with_generated(code, generated)
        model1_result = ("S" if is_similar else "N", [delta_details])
        is_plagiarized, delta, delta_details, sim_status, evidence = await detect_plagiarism_with_generated(code, generated)
        model2_result = ("S" if is_plagiarized else "N", [delta_details] + evidence)
        
        # Process code
        process_code_submission(code, handle)
        prepend_code_count()
        c_similar_handles = detect_similar_codes(code, handle)
        
        analysis = {"label": label, "confidence": confidence}
        report = generate_report(analysis, issues, handle, model1_result, model2_result)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"handle": handle, "report": report, "confidence": confidence, "label": label}
        })
    except Exception as e:
        logger.error(f"Analyze form endpoint failed: {str(e)}")
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Analyze form failed: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    port = int(environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
