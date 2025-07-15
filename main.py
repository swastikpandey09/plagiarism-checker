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
from os import environ, getenv
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
from dotenv import load_dotenv
import requests
import time
import datetime
from collections import Counter
import math

# Logging setup
valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level = environ.get("LOG_LEVEL", "INFO").upper()
if log_level not in valid_log_levels:
    log_level = "INFO"
logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s | %(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client with OpenRouter credentials
env_base_url = getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
correct_base_url = env_base_url if env_base_url.endswith("/v1") else "https://openrouter.ai/api/v1"
if env_base_url != correct_base_url:
    logger.warning(f"Environment OPENROUTER_BASE_URL ({env_base_url}) overridden with {correct_base_url}")
api_key = getenv("OPENROUTER_API_KEY")
if not api_key or not api_key.startswith("sk-or-v1-"):
    logger.error("OPENROUTER_API_KEY is missing or invalid")
    raise RuntimeError("OPENROUTER_API_KEY is missing or invalid")
masked_key = f"{api_key[:10]}...{api_key[-4:]}" if api_key else "None"
logger.info(f"Loaded OPENROUTER_API_KEY: {masked_key}")
client = OpenAI(base_url=correct_base_url, api_key=api_key)

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
    "delta_threshold": 0.10,
    "plagiarism_threshold": 0.10,
}

# Token frequencies from Codeforces
HUMAN_FREQUENCIES = {
    'i': 43.15, 'n': 34.04, 't': 32.95, 'e': 28.78, 's': 23.73, ';': 22.71, '(': 21.37, ')': 21.36,
    'r': 19.57, '=': 18.26, 'a': 18.10, 'o': 17.22, 'c': 14.98, 'l': 14.78, ',': 13.93, '1': 13.16,
    'p': 12.35, 'u': 12.31, 'd': 11.24, 'f': 10.57, 'int': 7.22, '0': 5.64, 'if': 3.72,
    'x': 3.07, 'for': 2.93, 'j': 2.80, 'k': 2.67, 'b': 2.12, 'v': 1.89, 'cin': 1.81,
    '2': 1.78, 'm': 1.70, 'y': 1.59, 'return': 1.35, 'using': 1.32, 'else': 1.27, 'cout': 1.22,
    'll': 1.13, '#include': 1.01, '<iostream>': 1.01, '<cmath>': 0.8, '<set>': 0.8, '<vector>': 0.8,
    '<map>': 0.8, '<string>': 0.8, '<algorithm>': 0.8, 'namespace': 1.0, 'std': 1.0, 'char': 1.5,
    'while': 1.5, '>>': 1.2, '<<': 1.2, 'endl': 1.0, 'main': 1.0, 'num': 1.0, 'let': 0.5,
    'letter': 0.5, '!=': 1.0, '++': 1.0, '--': 1.0, '+': 1.0, '<=': 1.0, "'a'": 0.5,
    ' ': 50.0, '; ': 8.80, '{ ': 3.58, '} ': 2.68, 'if ': 2.53, 'for ': 2.20, 'int ': 1.93,
    'int i': 1.54, '= 0;': 1.49, '(int': 1.40, 'cin ': 1.20, 'cout ': 1.20
}

# Verify MODEL_ARCHIVE
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

# Helper function to check OpenRouter API health
def check_openrouter_health(api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> Tuple[bool, List[str]]:
    try:
        if not api_key or not api_key.startswith("sk-or-v1-"):
            logger.error("Invalid or missing API key for health check")
            return False, []
        masked_key = f"{api_key[:10]}...{api_key[-4:]}" if api_key else "None"
        logger.debug(f"Health check using API key: {masked_key}")
        endpoint = f"{base_url.rstrip('/')}/models"
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
    api_key = getenv("OPENROUTER_API_KEY")
    base_url = "https://openrouter.ai/api/v1"
    if not api_key or not api_key.startswith("sk-or-v1-"):
        logger.error("OPENROUTER_API_KEY is missing or invalid")
        raise RuntimeError("OPENROUTER_API_KEY is missing or invalid")
    is_healthy, models = check_openrouter_health(api_key, base_url)
    if not is_healthy:
        logger.error("Startup failed: OpenRouter API unavailable")
        raise RuntimeError("Startup failed: OpenRouter API unavailable")
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

# Query API function
def query_api(messages: List[Dict[str, str]], model: str = None, temp: float = 0.7, max_tokens: int = 2000) -> str:
    api_key = getenv("OPENROUTER_API_KEY")
    if not api_key or not api_key.startswith("sk-or-v1-"):
        logger.error("OPENROUTER_API_KEY is missing or invalid in query_api")
        return ""
    masked_key = f"{api_key[:10]}...{api_key[-4:]}" if api_key else "None"
    logger.debug(f"Using API key: {masked_key} for query_api")
    
    if not model:
        available_models = getattr(app.state, "available_models", [])
        model = next((m for m in available_models if m == "moonshotai/kimi-k2:free"), "moonshotai/kimi-k2:free")
        logger.debug(f"No model specified, using: {model}")
    
    fallback_models = ["qwen/qwen3-8b:free", "mistralai/mistral-7b-instruct:free"]
    models_to_try = [model] + fallback_models
    
    for current_model in models_to_try:
        try:
            logger.debug(f"Sending API request to OpenRouter with model {current_model}")
            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            try:
                parsed = json.loads(content)
                code = parsed.get("code", "")
                if not code:
                    match = re.search(r"```cpp\n(.*?)```", content, re.DOTALL)
                    if match:
                        return match.group(1).strip()
                    logger.error(f"No valid C++ code block found in API response")
                    return content.strip()
                return code.strip()
            except json.JSONDecodeError:
                match = re.search(r"```cpp\n(.*?)```", content, re.DOTALL)
                if not match:
                    logger.error(f"No valid C++ code block found in API response")
                    return content.strip()
                return match.group(1).strip()
        except AuthenticationError as e:
            logger.error(f"Authentication failed with model {current_model}: {str(e)}")
            if current_model != models_to_try[-1]:
                logger.warning(f"Retrying with next model")
                continue
            return ""
        except RateLimitError as e:
            if current_model != models_to_try[-1]:
                delay = 60.0
                logger.warning(f"Rate limit exceeded for model {current_model}: {str(e)}. Waiting {delay:.2f}s")
                time.sleep(delay)
                continue
            logger.error(f"Rate limit exceeded for all models: {str(e)}")
            return ""
        except APIError as e:
            if ("401" in str(e) or "404" in str(e) or "429" in str(e)) and current_model != models_to_try[-1]:
                delay = 60.0 if "429" in str(e) or "404" in str(e) else 0
                logger.warning(f"API error with model {current_model}: {str(e)}. Waiting {delay:.2f}s")
                time.sleep(delay)
                continue
            logger.error(f"OpenRouter API error with model {current_model}: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in API query with model {current_model}: {str(e)}")
            if current_model != models_to_try[-1]:
                continue
            return ""
    logger.error(f"All models failed: {models_to_try}")
    return ""

# Code processing functions
def clean_code(code: str, preserve_comments: bool = False) -> str:
    if not code:
        return ""
    if not preserve_comments:
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return re.sub(r'\s+', ' ', code).strip()

def tokenize_code(code: str) -> List[str]:
    if not code:
        return []
    tokens = []
    current = ""
    i = 0
    while i < len(code):
        if code[i].isspace():
            if current:
                tokens.append(current)
                current = ""
            whitespace = ""
            while i < len(code) and code[i].isspace():
                whitespace += code[i]
                i += 1
            if whitespace:
                tokens.append(' ')
            continue
        elif code[i] in '();,=+-*/{}<>!':
            if current:
                tokens.append(current)
                current = ""
            if i + 1 < len(code) and code[i:i + 2] in ('>>', '<<', '++', '--', '!=', '<='):
                tokens.append(code[i:i + 2])
                i += 2
            else:
                tokens.append(code[i])
                i += 1
        else:
            current += code[i]
            i += 1
    if current:
        tokens.append(current)

    merged = []
    i = 0
    while i < len(tokens):
        for length in range(5, 0, -1):
            if i + length <= len(tokens):
                sequence = ''.join(tokens[i:i + length])
                if sequence in HUMAN_FREQUENCIES:
                    merged.append(sequence)
                    i += length
                    break
        else:
            merged.append(tokens[i])
            i += 1
    return merged

def compute_frequency_features(code: str) -> Dict[str, float]:
    tokens = tokenize_code(code)
    total_chars = sum(len(str(t)) for t in tokens)
    if total_chars == 0:
        return {}
    token_counts = Counter(tokens)
    return {token: (count / total_chars) * 1000 for token, count in token_counts.items()}

def classify_code(code: str, handle: str) -> Tuple[bool, str, float, str]:
    if not code.strip():
        return False, "Code cannot be empty", 0.0, handle
    code_freq = compute_frequency_features(code)
    if not code_freq:
        return False, "Empty code after tokenization", 0.0, handle
    deviation = 0.0
    matched_tokens = 0
    deviation_details = []
    for token, expected_freq in HUMAN_FREQUENCIES.items():
        actual_freq = code_freq.get(token, 0)
        diff = (actual_freq - expected_freq) ** 2
        deviation += diff
        matched_tokens += 1
        if diff > 0:
            deviation_details.append(f"Token '{token}': expected {expected_freq:.2f}, actual {actual_freq:.2f}, diff {diff:.2f}")
    for token, actual_freq in code_freq.items():
        if token not in HUMAN_FREQUENCIES:
            penalty = 1.5 if token in ('z', 'q', 'w') else 0.5
            diff = actual_freq ** 2 * penalty
            deviation += diff
            deviation_details.append(f"Unknown token '{token}': freq {actual_freq:.2f}, penalty {penalty}, diff {diff:.2f}")
    deviation = math.sqrt(deviation / max(matched_tokens, 1))
    threshold = 50.0
    confidence = min(0.99, 1.0 / (1.0 + math.exp(-0.05 * (threshold - deviation))))
    label = "Human" if deviation < threshold else "AI"
    return True, label, confidence, handle

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
                    name = tokens[i]
                    i += 1
                    while i < len(tokens) and tokens[i] in ("*", "&", ",", "="):
                        if tokens[i] == "=" and i + 1 < len(tokens):
                            i += 1
                        i += 1
                        name = tokens[i] if i < len(tokens) else ""
                        i += 1
                    if name and name != "(" and name not in keywords and name not in primitives and name not in containers and not name.isdigit():
                        variables.add(name)
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
    for i in range(0, len(lines), CONFIG["comment_block"]):
        block = lines[i:i + CONFIG["comment_block"]]
        has_long_comment = any(re.search(r'//(.{%d,})' % CONFIG["comment_length"], line) for line in block)
        if not has_long_comment:
            return True
    return False

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
    total = sum(((i - center_r) ** 2 + (j - center_c) ** 2) ** 0.5 for i in range(rows) for j in range(cols))
    return total / (rows * cols) if rows * cols > 0 else 0.0

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
    if not code1 or not code2:
        return False, "Empty code"
    cleaned1, cleaned2 = clean_code(code1), clean_code(code2)
    lcs_score = lcs_length(cleaned1, cleaned2)
    lcs_ratio = lcs_score / min(len(cleaned1), len(cleaned2)) if cleaned1 and cleaned2 else 0
    hash1, hash2 = compute_hash(code1), compute_hash(code2)
    common_hashes = len(set(hash1) & set(hash2))
    hash_ratio = common_hashes / min(len(hash1), len(hash2)) if hash1 and hash2 else 0
    is_similar = lcs_ratio >= CONFIG["lcs_threshold"] or hash_ratio >= CONFIG["hash_threshold"]
    return is_similar, f"LCS: {lcs_ratio:.2f}, Hash: {hash_ratio:.2f}"

async def detect_plagiarism(code: str, handle: str) -> Tuple[bool, float, str, str, List[str]]:
    if not code:
        return False, 0.0, "Empty code", "N", ["Empty code"]
    cleaned_code = clean_code(code)
    intent_prompt = [
        {"role": "system", "content": "Analyze the following C++ code and describe its purpose in a concise manner."},
        {"role": "user", "content": f"```cpp\n{code}\n```"}
    ]
    intent = query_api(intent_prompt)
    if not intent:
        return False, 0.0, "Failed to get intent", "N", ["Failed to get intent"]
    generate_prompt = [
        {
            "role": "system",
            "content": (
                "You are an expert C++ programmer. Generate C++ code that solves the same problem as described below. "
                "Ensure the code is complete, syntactically correct, and uses modern C++ practices. "
                "Wrap the generated code in a ```cpp code block, like this:\n"
                "```cpp\n// Your code here\n```\n"
                "Do not include explanations or comments outside the code block."
            )
        },
        {"role": "user", "content": intent}
    ]
    generated_code = query_api(generate_prompt)
    match = re.search(r"```cpp\n(.*?)```", generated_code, re.DOTALL)
    generated_code = match.group(1).strip() if match else generated_code.strip()
    if not generated_code:
        return False, 0.0, "No generated code", "N", ["No generated code"]
    matrix1 = string_to_matrix(pad_string(cleaned_code))
    matrix2 = string_to_matrix(pad_string(generated_code))
    delta = abs(compute_global_distance(matrix1) - compute_global_distance(matrix2))
    lcs_score = lcs_length(cleaned_code, clean_code(generated_code))
    lcs_ratio = lcs_score / min(len(cleaned_code), len(clean_code(generated_code))) if cleaned_code else 0
    evidence = [f"Delta: {delta:.4f}", f"LCS: {lcs_ratio:.2f}"]
    is_plagiarized = delta < CONFIG["plagiarism_threshold"] or lcs_ratio >= CONFIG["lcs_threshold"]
    return is_plagiarized, delta, f"Delta: {delta:.4f}", "S" if is_plagiarized else "N", evidence

def process_code_submission(code: str, handle: str) -> None:
    try:
        cleaned_code = clean_code(code)
        if not cleaned_code:
            raise ValueError("Empty code after cleaning")
        with INPUT_1568.open("w", encoding="utf-8") as f:
            f.write(f"{cleaned_code}\nRqq7\n{handle}\n")
        with INPUT_1435.open("a+", encoding="utf-8") as f:
            f.seek(0)
            lines = f.readlines()
            count = len([line for line in lines if line.strip() and not line.startswith("Rqq7")])
            f.write(f"xc9@{handle}\n{cleaned_code}\n")
        with INPUT_1435.open("r+", encoding="utf-8") as f:
            lines = f.readlines()
            f.seek(0)
            f.write(f"{count + 1}\n")
            f.writelines(lines)
        logger.info(f"Processed code for handle {handle}")
    except (IOError, ValueError) as e:
        logger.error(f"Failed to process submission: {e}")
        raise

def detect_similar_codes(code: str, handle: str) -> List[str]:
    try:
        similar_handles = []
        if not INPUT_1435.exists():
            logger.warning("input1435.txt not found for similarity check")
            return []
        with INPUT_1435.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        current_code = None
        current_handle = None
        codes = {}
        for line in lines:
            if line.startswith("xc9@"):
                current_handle = line[4:].strip()
            elif current_handle and line.strip() and not line.startswith("Rqq7"):
                current_code = line.strip()
                if current_handle != handle:
                    codes[current_handle] = current_code
        cleaned_code = clean_code(code)
        for other_handle, other_code in codes.items():
            is_similar, details = compare_codes(cleaned_code, other_code)
            if is_similar:
                similar_handles.append(f"{other_handle}: {details}")
        with OUTPUT_1435.open("w", encoding="utf-8") as f:
            if similar_handles:
                f.write(f"{len(similar_handles)} similar pairs of codes detected\n\n")
                f.write(f"{len(similar_handles)} cheaters detected:\n")
                f.write("\n".join(similar_handles) + "\n")
            else:
                f.write("0 similar pairs of codes detected\n\n0 cheaters detected:\n")
        return similar_handles
    except Exception as e:
        logger.error(f"Failed to detect similar codes: {e}")
        return []

def generate_report(analysis: Dict[str, Any], issues: List[str], handle: str,
                   model1_result: Optional[Tuple[str, List[str]]] = None,
                   model2_result: Optional[Tuple[str, List[str]]] = None) -> str:
    report = f"Plagiarism Report for {handle}\n\n"
    report += "AI/Human Classification:\n"
    report += f"Label: {analysis['label']} (Confidence: {analysis['confidence']:.2%})\n"
    report += "Suspicious Patterns:\n"
    report += "\n".join([f"- {x}" for x in issues]) if issues else "- None\n"
    if model1_result:
        report += "\nCode Similarity (Model 1):\n"
        report += f"Similar: {model1_result[0]}\nEvidence:\n" + "\n".join([f"- {x}" for x in model1_result[1]]) + "\n"
    if model2_result:
        report += "\nCode Similarity (Model 2):\n"
        report += f"Similar: {model2_result[0]}\nEvidence:\n" + "\n".join([f"- {x}" for x in model2_result[1]]) + "\n"
    report += f"\nCode Similarity Check:\nSimilar to handles: {', '.join(analysis.get('similar_handles', [])) or 'None'}\n"
    flags = sum(1 for x in [
        analysis['label'] == "AI",
        model1_result and model1_result[0] == "S",
        model2_result and model2_result[0] == "S",
        analysis.get('similar_handles', [])
    ] if x)
    verdict = "AI-generated or Plagiarized" if flags >= 2 else "Human-written"
    report += f"\nConclusion:\n- {verdict}\n"
    report += "\nRecommendations:\n"
    if verdict == "AI-generated or Plagiarized":
        report += "- Review for AI usage or plagiarism.\n- Verify submission authenticity.\n"
        if analysis['label'] == "AI":
            report += "- Discuss potential AI generation with submitter.\n"
        if (model1_result and model1_result[0] == "S") or (model2_result and model2_result[0] == "S") or analysis.get('similar_handles', []):
            report += "- Investigate potential code copying.\n"
    else:
        report += "- No further action required.\n"
    return report

# Endpoints
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        api_key = getenv("OPENROUTER_API_KEY")
        if not api_key or not api_key.startswith("sk-or-v1-"):
            logger.error("OPENROUTER_API_KEY is missing or invalid in health check")
            return {"status": "unhealthy", "error": "Invalid or missing API key"}
        is_healthy, models = check_openrouter_health(api_key)
        if not is_healthy:
            return {"status": "unhealthy", "error": "OpenRouter API unavailable"}
        return {"status": "healthy", "models": models}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/analyze", response_class=HTMLResponse, responses={
    200: {"content": {"text/html": {}}},
    422: {"model": HTTPValidationError}
})
async def analyze_code(request: Request, code: str = Form(..., max_length=100_000), handle: str = Form("triumph")):
    try:
        SingleCodeInput(code=code, handle=handle)
        health = await health_check()
        if health["status"] != "healthy":
            return templates.TemplateResponse("index.html", {"request": request, "error": f"OpenRouter API unavailable: {health.get('error', 'Unknown error')}"})
        
        success, label, confidence, handle = classify_code(code, handle)  # Removed await
        if not success:
            return templates.TemplateResponse("index.html", {"request": request, "error": label})
        variables, has_long_vars = extract_variables(code)
        issues = [
            f"Variables longer than {CONFIG['var_length']} characters" if has_long_vars else "",
            "Suspicious comments" if has_suspicious_comments(code) else ""
        ]
        issues = [x for x in issues if x]
        
        is_plagiarized, delta, delta_details, sim_status, evidence = await detect_plagiarism(code, handle)
        model1_result = ("S" if is_plagiarized else "N", [delta_details])
        model2_result = (sim_status, evidence)
        similar_handles = detect_similar_codes(code, handle)
        
        process_code_submission(code, handle)
        analysis = {"label": label, "confidence": confidence, "similar_handles": similar_handles}
        report = generate_report(analysis, issues, handle, model1_result, model2_result)
        
        with OUTPUT_1435.open("a+", encoding="utf-8") as f:
            f.write(f"Handle: {handle} ({label}-generated)\n")
            f.write(f"Confidence: {confidence:.2%}\n")
            f.write(f"Plagiarism: {'Detected' if is_plagiarized else 'Not detected'}\n")
            f.write("\n".join(evidence) + "\n\n")
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"handle": handle, "report": report, "confidence": confidence, "label": label}
        })
    except Exception as e:
        logger.error(f"Analyze code failed: {str(e)}")
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Analyze code failed: {str(e)}"})

@app.post("/prepare", response_class=HTMLResponse)
async def prepare_submission(request: Request, code: str = Form(...), handle: str = Form(...)):
    try:
        SingleCodeInput(code=code, handle=handle)
        process_code_submission(code, handle)
        return templates.TemplateResponse("index.html", {"request": request, "message": f"Code for {handle} prepared"})
    except Exception as e:
        logger.error(f"Prepare submission failed: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Prepare submission failed: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    port = int(environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
