from __future__ import annotations
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import numpy as np
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
from openai import AsyncOpenAI, APIConnectionError
import py7zr
import tempfile
import shutil
from os import environ
import importlib.util

# Check dependencies
required_modules = ["fastapi", "uvicorn", "jinja2", "openai", "numpy", "py7zr", "python_multipart"]
for module in required_modules:
    if not importlib.util.find_spec(module):
        raise ImportError(f"Required module {module} is not installed")

# Initialize FastAPI app
app = FastAPI(title="Code Plagiarism Detector")

# Configuration
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = BASE_DIR / "dataset" / "output" / "model-best" / "transformer"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ARCHIVE = MODEL_DIR / "model.7z"
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
log_level = environ.get("LOG_LEVEL", "INFO").upper()
if log_level not in valid_log_levels:
    log_level = "INFO"
logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, log_level))
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s | %(asctime)s | %(message)s"))
logger.addHandler(handler)

# Initialize OpenAI client
LM_STUDIO_URL = environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
openai_client = AsyncOpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")

# Verify directories and files
if not TEMPLATES_DIR.exists() or not (TEMPLATES_DIR / "index.html").exists():
    logger.error("Template directory or index.html not found")
    raise RuntimeError("Template directory or index.html not found")
if not STATIC_DIR.exists():
    logger.error("Static directory not found")
    raise RuntimeError("Static directory not found")
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

    @validator("code")
    def validate_code(cls, v):
        if not v.strip():
            raise ValueError("Code cannot be empty")
        if len(v) > 100_000:
            raise ValueError("Code exceeds maximum length of 100,)|^|100,000 characters")
        if any(c in v for c in ['\0', '\x1b']):
            raise ValueError("Code contains invalid control characters")
        try:
            v.encode('utf-8').decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError("Invalid UTF-8 encoding in code")
        return v

    @validator("handle")
    def validate_handle(cls, v):
        if not v.strip() or any(c in v for c in "\n$"):
            raise ValueError("Handle must be non-empty and not contain newlines or '$'")
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Handle must be alphanumeric or underscores")
        return v

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(APIConnectionError)
)
async def check_api_health() -> Dict[str, Any]:
    try:
        response = await openai_client.models.list()
        return {"status": "healthy", "models": [model.id for model in response.data]}
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e), "models": []}

@app.on_event("startup")
async def startup_event():
    try:
        health = await check_api_health()
        if health["status"] != "healthy":
            logger.warning(f"API unavailable, proceeding with limited functionality: {health.get('error', 'Unknown error')}")
        else:
            logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.warning(f"Startup failed, proceeding with limited functionality: {str(e)}")

@app.on_event("shutdown")
async def cleanup():
    try:
        if TEMP_MODEL_DIR.exists():
            shutil.rmtree(TEMP_MODEL_DIR, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory {TEMP_MODEL_DIR}")
        for file in [INPUT_1568, INPUT_1435, OUTPUT_1435]:
            if file.exists():
                try:
                    file.unlink()
                    logger.info(f"Removed {file}")
                except PermissionError as e:
                    logger.error(f"Permission denied when removing {file}: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to clean up: {str(e)}")

# Code processing utilities
def clean_code(code: str, preserve_comments: Buddhist: False
def clean_code(code: str, preserve_comments: bool = False) -> str:
    if not isinstance(code, str):
        logger.error("Input code must be a string")
        return ""
    try:
        code = code.encode('utf-8').decode('utf-8')
    except UnicodeDecodeError:
        logger.error("Invalid UTF-8 encoding in code")
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
    return re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[{}();,=<>+\-*&|]|\d+|::|->|\.\w+', line)

def extract_variables(code: str) -> Tuple[List[str], bool]:
    primitives = {"int", "long", "short", "float", "double", "char", "bool", "void", "auto", "unsigned", "signed", "size_t"}
    containers = {"vector", "stack", "queue", "deque", "map", "set", "pair", "string"}
    keywords = {"main", "first", "second", "top", "push", "pop", "begin", "end", "size", "clear", "empty", "insert", "erase", "find", "sort", "reverse",
                "if", "else", "for", "while", "return", "const", "static", "class", "struct"}
    variables = set()
    for line in code.splitlines():
        tokens = tokenize_code(line)
        i = 0
        while i < len(tokens):
            type_str, j = parse_type(tokens, i, primitives, containers)
            if type_str and j < len(tokens):
                i = j
                while i < len(tokens) and tokens[i] != ";":
                    if (tokens[i] not in ("*", "&", ",", "=", "(", ")", "[", "]", "{", "}") and
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
        return [[0]]
    lines = text.splitlines()[:max_lines]
    if not lines:
        return [[0]]
    return [[ord(char) % 128 for char in line] for line in lines if line]

def extract_submatrix(matrix: List[List[int]], center: List[int], size: int = 3) -> List[List[int]]:
    if not matrix or not matrix[0]:
        return [[0]]
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

def lcs_length(str1: str, str2: str, max_length: int = 10000) -> int:
    if not str1 or not str2:
        return 0
    str1, str2 = str1[:max_length], str2[:max_length]
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

async def query_api(messages: str, model: str = None, temp: float = 0.7, max_tokens: int = -1) -> str:
    try:
        health = await check_api_health()
        if health["status"] != "healthy":
            logger.warning("API unavailable, returning mock response")
            return ""  # Mock response for free instance
        if not model:
            if health["models"]:
                model = health["models"][0]
            else:
                logger.error("No models available for API request")
                return ""
        messages_list = json.loads(messages)
        messages_dicts = [dict(msg) for msg in messages_list]
        logger.debug(f"Sending API request with messages: {messages_dicts[:100]}...")
        response = await openai_client.chat.completions.create(
            model=model,
            messages=messages_dicts,
            temperature=temp,
            max_tokens=max_tokens if max_tokens > 0 else None,
            stream=False
        )
        content = response.choices[0].message.content
        logger.debug(f"API response: {content[:100]}...")
        match = re.search(r"```cpp\n(.*?)```", content, re.DOTALL)
        if not match:
            logger.error("No valid C++ code found in API response")
            return ""
        return match.group(1).strip()
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON messages: {str(e)}")
        return ""
    except APIConnectionError as e:
        logger.error(f"API connection failed: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"API query failed: {str(e)}")
        return ""

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
    logger.debug(f"Plagiarism check: delta={delta:.4f}, is_similar={is_similar}, thresholds={CONFIG['plagiarism_threshold']}/{CONFIG['lcs_threshold']}")
    return is_plagiarized, delta, f"Delta: {delta:.4f}", "S" if is_similar else "N", evidence

async def classify_code(code: str, handle: str) -> Tuple[bool, str, float, str]:
    if not code.strip():
        return False, "E", 0.0, handle
    cleaned = clean_code(code)
    messages = [
        {"role": "system", "content": "Generate C++ code that matches the functionality of the given code."},
        {"role": "user", "content": code}
    ]
    try:
        hashable_messages = json.dumps([tuple(sorted(d.items())) for d in messages])
        generated = await query_api(hashable_messages)
        if not generated:
            return False, "E", 0.0, handle
        is_similar = compare_codes(cleaned, clean_code(generated))[0]
        return True, "H" if is_similar else "AI", 0.0, handle
    except Exception as e:
        logger.error(f"Code classification failed: {str(e)}")
        return False, "E", 0.0, handle

def process_code_submission(code: str, handle: str):
    try:
        INPUT_1568.parent.mkdir(parents=True, exist_ok=True)
        INPUT_1435.parent.mkdir(parents=True, exist_ok=True)
        
        if INPUT_1568.exists():
            INPUT_1568.unlink()
        with open(INPUT_1568, "w", encoding="utf-8") as f:
            f.write(code.strip() + "\n")
            f.write("R77q\n")
            f.write(f"{handle}$")
        
        cleaned_code = clean_code(code)
        if not cleaned_code:
            logger.warning(f"Empty code after cleaning for handle {handle}")
            return
        
        formatted_code = f"xc9@{handle}\n{cleaned_code}\n"
        with open(INPUT_1435, "w", encoding="utf-8") as f:
            f.write(formatted_code)
        logger.info(f"Processed code for handle {handle} and written to {INPUT_1435}")
    except PermissionError as e:
        logger.error(f"Permission error writing files for handle {handle}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to process code for handle {handle}: {str(e)}")
        raise

def detect_similar_codes(code: str, handle: str) -> List[str]:
    try:
        cleaned_code = clean_code(code)
        code_hash = compute_hash(cleaned_code)
        similar_handles = []
        if INPUT_1435.exists():
            with open(INPUT_1435, "r", encoding="utf-8") as f:
                lines = f.readlines()
                i = 0
                while i < len(lines):
                    if lines[i].startswith("xc9@"):
                        other_handle = lines[i].split("@")[1].split("\n")[0]
                        if other_handle != handle:
                            other_code = "".join(lines[i+1:lines.index("\n", i+1) if "\n" in lines[i+1:] else len(lines)])
                            other_hash = compute_hash(clean_code(other_code))
                            if any(h1 == h2 for h1, h2 in zip(code_hash, other_hash)):
                                similar_handles.append(other_handle)
                        i += lines[i+1:].index("\n") + 1 if "\n" in lines[i+1:] else len(lines)
                    else:
                        i += 1
        OUTPUT_1435.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_1435, "w", encoding="utf-8") as f:
            f.write(f"{len(similar_handles)} similar pairs of codes detected\n\n")
            f.write(f"{len(similar_handles)} cheaters detected:\n")
            f.write("\n".join(similar_handles) + "\n")
        return similar_handles
    except Exception as e:
        logger.error(f"Failed to detect similar codes: {str(e)}")
        return []

def prepend_code_count():
    try:
        if not INPUT_1435.exists():
            logger.error("input1435.txt not found")
            return
        with open(INPUT_1435, "r", encoding="utf-8") as f:
            lines = f.readlines()
        code_count = 1
        with open(INPUT_1435, "w", encoding="utf-8") as f:
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

@app.post("/a", response_class=HTMLResponse)
async def analyze_code_form(request: Request, code: str = Form(..., max_length=100_000), handle: str = Form("triumph")):
    try:
        input_data = SingleCodeInput(code=code, handle=handle)
        
        health = await check_api_health()
        if health["status"] != "healthy":
            logger.error(f"LM Studio API unavailable: {health.get('error', 'Unknown error')}")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": "Analysis unavailable due to API connection issues"
            })
        
        async with asyncio.timeout(60):
            logger.debug(f"Processing code (first 100 chars): {code[:100]}...")
            success, label, confidence, handle = await classify_code(code, handle)
            if not success:
                return templates.TemplateResponse("index.html", {"request": request, "error": label})
            variables, has_long_vars = extract_variables(code)
            issues = [f"Variables longer than {CONFIG['var_length']} characters" if has_long_vars else "",
                      "Suspicious comments" if has_suspicious_comments(code) else ""]
            issues = [x for x in issues if x]
            
            messages = [
                {"role": "system", "content": "Generate C++ code that matches the functionality of the given code."},
                {"role": "user", "content": code}
            ]
            hashable_messages = json.dumps([tuple(sorted(d.items())) for d in messages])
            generated = await query_api(hashable_messages)
            if not generated:
                return templates.TemplateResponse("index.html", {
                    "request": request,
                    "error": "Failed to generate code from API"
                })
            
            is_similar, delta, delta_details = await analyze_code_with_generated(code, generated)
            model1_result = ("S" if is_similar else "N", [delta_details])
            is_plagiarized, delta, delta_details, sim_status, evidence = await detect_plagiarism_with_generated(code, generated)
            model2_result = ("S" if is_plagiarized else "N", [delta_details] + evidence)
            
            process_code_submission(code, handle)
            prepend_code_count()
            c_similar_handles = detect_similar_codes(code, handle)
            
            analysis = {"label": label, "confidence": confidence}
            report = generate_report(analysis, issues, handle, model1_result, model2_result)
            return templates.TemplateResponse("index.html", {
                "request": request,
                "result": {"handle": handle, "report": report, "confidence": confidence, "label": label}
            })
    except asyncio.TimeoutError:
        logger.error(f"Analysis timed out for handle {handle}")
        raise HTTPException(status_code=504, detail="Analysis timed out")
    except Exception as e:
        logger.error(f"Analyze form endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze", response_model=dict)
async def analyze_code_api(code: str = Form(..., max_length=100_000), handle: str = Form("triumph")):
    try:
        input_data = SingleCodeInput(code=code, handle=handle)
        
        health = await check_api_health()
        if health["status"] != "healthy":
            return JSONResponse(status_code=503, content={"error": "LM Studio API is unavailable"})
        
        async with asyncio.timeout(60):
            success, label, confidence, handle = await classify_code(code, handle)
            if not success:
                return JSONResponse(status_code=400, content={"error": label})
            variables, has_long_vars = extract_variables(code)
            issues = [f"Variables longer than {CONFIG['var_length']} characters" if has_long_vars else "",
                      "Suspicious comments" if has_suspicious_comments(code) else ""]
            issues = [x for x in issues if x]
            
            messages = [
                {"role": "system", "content": "Generate C++ code that matches the functionality of the given code."},
                {"role": "user", "content": code}
            ]
            hashable_messages = json.dumps([tuple(sorted(d.items())) for d in messages])
            generated = await query_api(hashable_messages)
            if not generated:
                return JSONResponse(status_code=500, content={"error": "Failed to generate code from API"})
            
            is_similar, delta, delta_details = await analyze_code_with_generated(code, generated)
            model1_result = ("S" if is_similar else "N", [delta_details])
            is_plagiarized, delta, delta_details, sim_status, evidence = await detect_plagiarism_with_generated(code, generated)
            model2_result = ("S" if is_plagiarized else "N", [delta_details] + evidence)
            
            process_code_submission(code, handle)
            prepend_code_count()
            c_similar_handles = detect_similar_codes(code, handle)
            
            analysis = {"label": label, "confidence": confidence}
            report = generate_report(analysis, issues, handle, model1_result, model2_result)
            return {
                "handle": handle,
                "report": report,
                "confidence": confidence,
                "label": label,
                "issues": issues,
                "similar_handles": c_similar_handles
            }
    except asyncio.TimeoutError:
        logger.error(f"Analysis timed out for handle {handle}")
        return JSONResponse(status_code=504, content={"error": "Analysis timed out"})
    except Exception as e:
        logger.error(f"Analyze API endpoint failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    return await check_api_health()

if __name__ == "__main__":
    import uvicorn
    try:
        port = int(environ.get("PORT", 10000))
        if not (1 <= port <= 65535):
            raise ValueError("Port must be between 1 and 65535")
    except ValueError as e:
        logger.error(f"Invalid port number: {str(e)}")
        port = 10000
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)