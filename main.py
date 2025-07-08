from __future__ import annotations
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import asyncio
from collections import defaultdict
import numpy as np
import spacy
import uvicorn
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from functools import lru_cache
from openai import AsyncOpenAI
import py7zr
import tempfile
import shutil
import os
# Configuration
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODEL_ARCHIVE = BASE_DIR / "model-best" / "transformer.7z"
TEMP_MODEL_DIR = tempfile.mkdtemp()


CONFIG = {
    "lcs_threshold": 0.5,
    "hash_threshold": 0.125,
    "var_length": 5,
    "comment_length": 5,
    "comment_block": 5,
    "delta_threshold": 0.5,
    "plagiarism_threshold": 0.10
}
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# --- Decompress the model on startup ---
try:
    with py7zr.SevenZipFile(MODEL_ARCHIVE, 'r') as archive:
        archive.extractall(path=TEMP_MODEL_DIR)
    logger.info(f"Successfully decompressed model to {TEMP_MODEL_DIR}")
except Exception as e:
    logger.error(f"Failed to decompress model: {e}")
    # Handle the error appropriately, maybe exit the application
    # if the model is critical.

# Initialize OpenAI client and spaCy model
openai_client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
try:
    # Load the model from the temporary directory
    nlp = spacy.load(os.path.join(TEMP_MODEL_DIR, "output", "model-best"))
    logger.info("Successfully loaded spaCy model.")
except Exception:
    nlp = None
    logger.warning("spaCy model not loaded; AI/Human classification disabled")


# FastAPI setup
app = FastAPI(title="Integrated Code Inspector")

@app.on_event("shutdown")
def cleanup():
    # Clean up the temporary directory on shutdown
    shutil.rmtree(TEMP_MODEL_DIR)
    logger.info(f"Cleaned up temporary model directory: {TEMP_MODEL_DIR}")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
# Pydantic models
class CodeInput(BaseModel):
    code1: str
    code2: str
    handle1: str | None = "user1"
    handle2: str | None = "user2"

class SingleCodeInput(BaseModel):
    code: str
    handle: str | None = triumph

class BatchCodeInput(BaseModel):
    codes: List[str]
    handles: List[str]

# String processing functions
def pad_string(text: str) -> str:
    lines = text.split('\n')
    if not lines:
        return ""
    max_len = max(len(line) for line in lines)
    if max_len % 2 != 0:
        for i, line in enumerate(lines):
            if len(line) == max_len:
                lines[i] = line + 'S'
                max_len += 1
                break
    padded_lines = [line + 'S' * (max_len - len(line)) for line in lines]
    if len(padded_lines) < max_len:
        padded_lines.extend(['S' * max_len for _ in range(max_len - len(padded_lines))])
    return '\n'.join(padded_lines)

def string_to_matrix(text: str) -> List[List[int]]:
    lines = text.split('\n')
    char_to_index = defaultdict(lambda: len(char_to_index))
    return [[char_to_index[char] for char in line] for line in lines if line]

def extract_submatrix(matrix: List[List[int]], center: List[int], size: int = 3) -> List[List[int]]:
    half_size = size // 2
    rows, cols = len(matrix), len(matrix[0]) if matrix else 0
    row_start, row_end = max(0, center[0] - half_size), min(rows, center[0] + half_size + 1)
    col_start, col_end = max(0, center[1] - half_size), min(cols, center[1] + half_size + 1)
    return [[matrix[i][j] for j in range(col_start, col_end)] for i in range(row_start, row_end)]

def matrix_avg_distance(matrix: List[List[int]]) -> float:
    if not matrix or not matrix[0]:
        return 0
    r, c = len(matrix), len(matrix[0])
    center_r, center_c = r // 2, c // 2
    total = sum((i - center_r) ** 2 + (j - center_c) ** 2 for i in range(r) for j in range(c))
    return total / (r * c) ** 0.5 if r * c else 0

async def compute_global_distance(matrix: List[List[int]]) -> float:
    if not matrix or not matrix[0]:
        return 0
    r, c = len(matrix), len(matrix[0])
    tasks = [asyncio.create_task(asyncio.to_thread(matrix_avg_distance, extract_submatrix(matrix, [i, j])))
             for i in range(r) for j in range(c)]
    results = await asyncio.gather(*tasks)
    return sum(results) / len(results) if results else 0

# Hashing and similarity functions
def compute_hash(text: str, by_line: bool = True) -> List[int]:
    if by_line:
        segments = text.splitlines()
    else:
        segments = segment_code(text)
    hashes = []
    for segment in segments:
        if not segment.strip():
            continue
        h = hashlib.sha256(segment.encode()).hexdigest()
        hashes.append(int(h, 16) % (2**32))
    return sorted(hashes)

def segment_code(text: str) -> List[str]:
    segments = []
    current = []
    i = 0
    while i < len(text):
        if i >= 3 and text[i-3:i+1] == 'xc9@':
            if current:
                segments.append(''.join(current))
                current = []
            i += 1
        else:
            current.append(text[i])
            i += 1
    if current:
        segments.append(''.join(current))
    return segments

def lcs_length(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    n, m = len(a), len(b)
    curr = [0] * (m + 1)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i-1] == b[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev
    return prev[m]

def compare_hashes(hash1: List[int], hash2: List[int]) -> int:
    i, j, common = 0, 0, 0
    while i < len(hash1) and j < len(hash2):
        if hash1[i] == hash2[j]:
            common += 1
            i += 1
            j += 1
        elif hash1[i] < hash2[j]:
            i += 1
        else:
            j += 1
    return common

def compare_codes(code1: str, code2: str) -> Tuple[bool, str]:
    code1_clean, code2_clean = clean_code(code1), clean_code(code2)
    lcs_score = lcs_length(code1_clean, code2_clean)
    lcs_ratio = lcs_score / min(len(code1_clean), len(code2_clean)) if code1_clean and code2_clean else 0
    return lcs_ratio < CONFIG["lcs_threshold"], f"LCS: {lcs_ratio:.2f}"

# Code preprocessing and analysis
def clean_code(code: str) -> str:
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return re.sub(r'\s+', ' ', code).strip()

def tokenize_code(line: str) -> List[str]:
    tokens = []
    current = ""
    for char in line:
        if char.isspace() or char in ",;=(){}[].<>*&":
            if current:
                tokens.append(current)
                current = ""
            if not char.isspace():
                tokens.append(char)
        else:
            current += char
    if current:
        tokens.append(current)
    return tokens

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
            if not type_str:
                i += 1
                continue
            i = j
            while i < len(tokens) and tokens[i] != ";":
                var = tokens[i]
                i += 1
                while var in ("*", "&", ",", "="):
                    if var == "=" and i < len(tokens):
                        i += 1
                    var = tokens[i] if i < len(tokens) else ""
                    i += 1
                if var and var != "(" and var not in keywords and var not in primitives and var not in containers and not var.isdigit():
                    variables.add(var)
            if i < len(tokens) and tokens[i] == ";":
                i += 1
    var_list = sorted(variables)
    has_long_vars = len([v for v in var_list if len(v) > CONFIG["var_length"]]) >= 2
    return var_list, has_long_vars

def parse_type(tokens: List[str], index: int, primitives: set, containers: set) -> Tuple[str, int]:
    type_str = ""
    while index < len(tokens) and (tokens[index] in primitives or tokens[index] in containers):
        if type_str:
            type_str += " "
        type_str += tokens[index]
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
    return type_str, index

def has_suspicious_comments(code: str) -> bool:
    lines = code.splitlines()
    for i in range(0, len(lines), CONFIG["comment_block"]):
        if not any(re.search(r'//(.{%d,})' % CONFIG["comment_length"], line) for line in lines[i:i+CONFIG["comment_block"]]):
            return True
    return False

def preprocess_code(code: str, handle: str) -> str:
    cleaned = clean_code(code)
    tokens = []
    for word in cleaned.split():
        for char in word:
            if char in "{};":
                tokens.extend(["x", "c", "9"])
            elif char not in " \t":
                tokens.append(char)
        tokens.extend(["x", "c", "9"])
    tokens.append(" ")
    return "".join(tokens) + handle

# AI analysis and plagiarism detection
@lru_cache(maxsize=1000)
async def query_api(messages: tuple, model: str = "gemma-3-12b-it", temp: float = 0.7, max_tokens: int = -1, stream: bool = False) -> str:
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temp,
            max_tokens=max_tokens if max_tokens > 0 else None,
            stream=stream
        )
        content = response.choices[0].message.content
        match = re.search(r"```cpp\n(.*?)```", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        logger.warning("No code block found in API response")
        return content
    except Exception as e:
        logger.error(f"API query failed: {str(e)}")
        return "API unavailable"

async def analyze_code(code: str, threshold: float = CONFIG["delta_threshold"]) -> Tuple[bool, float, str]:
    messages = (
        {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
        {"role": "user", "content": code}
    )
    generated = await query_api(messages)
    if not generated or "API unavailable" in generated:
        return False, 0.0, "API unavailable"
    original_matrix = string_to_matrix(pad_string(code))
    generated_matrix = string_to_matrix(pad_string(generated))
    delta = abs(await compute_global_distance(original_matrix) - await compute_global_distance(generated_matrix))
    return delta < threshold, delta, f"Delta: {delta:.4f}"

async def detect_plagiarism(code: str, threshold: float = CONFIG["plagiarism_threshold"]) -> Tuple[bool, float, str, str, List[str]]:
    messages = (
        {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
        {"role": "user", "content": code}
    )
    generated = await query_api(messages)
    if not generated or "API unavailable" in generated:
        return False, 0.0, "API unavailable", "N", ["API unavailable"]
    original_matrix = string_to_matrix(pad_string(code))
    generated_matrix = string_to_matrix(pad_string(generated))
    delta = abs(await compute_global_distance(original_matrix) - await compute_global_distance(generated_matrix))
    is_similar, metrics = compare_codes(code, generated)
    lcs_similar, lcs_details = compare_codes(code, generated)
    evidence = [f"Delta: {delta:.4f}", lcs_details] + metrics.split('\n')
    is_plagiarized = delta < threshold or is_similar or lcs_similar
    return is_plagiarized, delta, f"Delta: {delta:.4f}", "S" if is_similar else "N", evidence

async def classify_code(code: str, handle: str) -> Tuple[bool, str, float, str]:
    if not code.strip():
        return False, "E", 0.0, handle
    cleaned = clean_code(code)
    if nlp:
        doc = nlp(cleaned)
        ai_conf, human_conf = doc.cats.get("AI", 0.0), doc.cats.get("HUMAN", 0.0)
        conf = max(ai_conf, human_conf)
        messages = (
            {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
            {"role": "user", "content": code}
        )
        generated = await query_api(messages)
        if not generated or "API unavailable" in generated:
            is_similar = False
        else:
            is_similar, _ = compare_codes(cleaned, clean_code(generated))
        label = "H" if human_conf >= 0.95 or is_similar else "AI"
        return True, label, conf, handle
    else:
        messages = (
            {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
            {"role": "user", "content": code}
        )
        generated = await query_api(messages)
        if not generated or "API unavailable" in generated:
            return False, "E", 0.0, handle
        is_similar, _ = compare_codes(cleaned, clean_code(generated))
        label = "H" if is_similar else "AI"
        return True, label, 0.0, handle

async def compare_code_pair(code1: str, code2: str, handle1: str, handle2: str) -> Tuple[str, List[str]:
    code1_clean, code2_clean = clean_code(code1), clean_code(code2)
    if not code1_clean or not code2_clean:
        return "N", ["Empty"]
    lcs_score = lcs_length(code1_clean, code2_clean)
    lcs_ratio = lcs_score / min(len(code1_clean), len(code2_clean)) if code1_clean and code2_clean else 0
    hash1 = compute_hash(code1_clean, by_line=False)
    hash2 = compute_hash(code2_clean, by_line=False)
    common_hashes = compare_hashes(hash1, hash2)
    hash_ratio = common_hashes / min(len(hash1), len(hash2)) if hash1 and	hash2 else 0
    evidence = [f"LCS: {lcs_ratio:.2f}", f"Hash: {hash_ratio:.2f}"]
    return "S" if lcs_ratio >= 0.95 or hash_ratio >= 0.95 else "N", evidence

async def compare_code_pair_advanced(code1: str, code2: str, handle1: str, handle2: str) -> Tuple[str, List[str]]:
    messages1 = (
        {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
        {"role": "user", "content": code1}
    )
    messages2 = (
        {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
        {"role": "user", "content": code2}
    )
    generated1, generated2 = await asyncio.gather(query_api(messages1), query_api(messages2))
    if not generated1 or "API unavailable" in generated1:
        result1 = False, 0.0, "API unavailable"
    else:
        matrix1 = string_to_matrix(pad_string(code1))
        matrix_gen1 = string_to_matrix(pad_string(generated1))
        delta1 = abs(await compute_global_distance(matrix1) - await compute_global_distance(matrix_gen1))
        result1 = delta1 < CONFIG["delta_threshold"], delta1, f"Delta: {delta1:.4f}"
    if not generated2 or "API unavailable" in generated2:
        result2 = False, 0.0, "API unavailable"
    else:
        matrix2 = string_to_matrix(pad_string(code2))
        matrix_gen2 = string_to_matrix(pad_string(generated2))
        delta2 = abs(await compute_global_distance(matrix2) - await compute_global_distance(matrix_gen2))
        result2 = delta2 < CONFIG["delta_threshold"], delta2, f"Delta: {delta2:.4f}"
    is_similar1, delta1, evidence1 = result1
    is_similar2, delta2, evidence2 = result2
    if is_similar1 and is_similar2:
        return "S", [evidence1, evidence2, "Both AI"]
    return "N", [evidence1, evidence2]

# Reporting
def generate_report(analysis: Dict[str, Any], issues: List[str], handle: str, 
                   model1_result: Tuple[str, List[str]] = None, 
                   model2_result: Tuple[str, List[str]] = None, 
                   model3_result: Tuple[str, List[str]] = None) -> str:
    report = f"Report for {handle}\n\n"
    report += "Model 1 (AI/Human):\n"
    report += f"AI: {analysis['label']} (Conf: {analysis['confidence']:.2%})\n"
    report += "<span style='color: blue;'>Suspicious:</span>\n<span style='color: blue;'>- Variables</span>\n<span style='color: blue;'>- Comments</span>\nRules:\n"
    report += "\n".join([f"- {x}" for x in issues]) if issues else "- None\n"
    if model1_result:
        report += "\nModel 1 Plagiarism:\n" + f"Similarity: {model1_result[0]}\nEvidence:\n" + "\n".join([f"- {x}" for x in model1_result[1]]) + "\n"
    if model2_result:
        report += "\nModel 2 Plagiarism:\n" + f"Similarity: {model2_result[0]}\nEvidence:\n" + "\n".join([f"- {x}" for x in model2_result[1]]) + "\n"
    if model3_result:
        report += "\nModel 3 Plagiarism:\n" + f"Similarity: {model3_result[0]}\nEvidence:\n" + "\n".join([f"- {x}" for x in model3_result[1]]) + "\n"
    flags = sum(1 for x in [analysis['label'] == "AI", model2_result and model2_result[0] == "S", model3_result and model3_result[0] == "S"] if x)
    report += "\nConclusion:\n" + ("- AI\n" if flags >= 2 else "- Human\n")
    report += "\nRecommendations:\n"
    if analysis['label'] == "AI" or issues or (model1_result and model1_result[0] == "S") or (model2_result and model2_result[0] == "S") or (model3_result and model3_result[0] == "S"):
        report += "- Review for AI/Plagiarism.\n- Verify.\n"
        if analysis['label'] == "AI":
            report += "- Discuss AI usage.\n"
        if (model1_result and model1_result[0] == "S") or (model2_result and model2_result[0] == "S") or (model3_result and model3_result[0] == "S"):
            report += "- Investigate plagiarism.\n"
    else:
        report += "- Human-authored.\n"
    return report

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=JSONResponse)
async def analyze_single_code(input: SingleCodeInput):
    success, label, confidence, handle = await classify_code(input.code, input.handle)
    if not success:
        return {"handle": handle, "error": label}
    analysis = {"label": label, "confidence": confidence}
    report = generate_report(analysis, [], handle)
    return {
        "handle": handle,
        "ai_probability": confidence,
        "label": label,
        "report": report,
        "message": "Analysis complete"
    }

@app.post("/compare", response_class=JSONResponse)
async def compare_codes(input: CodeInput):
    is_similar, lcs_details = compare_codes(input.code1, input.code2)
    code1_proc = preprocess_code(input.code1, input.handle1)
    code2_proc = preprocess_code(input.code2, input.handle2)
    lcs_score = lcs_length(code1_proc, code2_proc)
    hash1 = compute_hash(code1_proc, by_line=False)
    hash2 = compute_hash(code2_proc, by_line=False)
    common_hashes, min_hashes = compare_hashes(hash1, hash2), min(len(hash1), len(hash2))
    compare3_score = compare_hashes(hash1, hash2)  # Simplified comparison
    is_plagiarized = (lcs_score >= 1100 or 
                      common_hashes >= min_hashes / 2 + min_hashes / 5 or 
                      common_hashes >= 100 or 
                      (common_hashes >= 70 and lcs_score >= 700) or 
                      lcs_score >= len(code1_proc) / 2 + len(code1_proc) / 3 + min_hashes * 2 or 
                      compare3_score >= 240)
    return {
        "handle1": input.handle1,
        "handle2": input.handle2,
        "plagiarised": is_plagiarized,
        "lcs": lcs_details,
        "detailed_metrics": {
            "lcs_score": lcs_score,
            "hash_common": common_hashes,
            "hash_min": min_hashes,
            "compare3_score": compare3_score
        },
        "message": f"{input.handle1} and {input.handle2} {'plagiarized' if is_plagiarized else 'not plagiarized'}"
    }

@app.post("/plagiarism_batch_compare", response_class=JSONResponse)
async def batch_compare_codes(input: BatchCodeInput):
    if len(input.codes) != len(input.handles):
        return {"error": "Number of codes and handles must match"}
    processed_codes = [preprocess_code(code, handle) for code, handle in zip(input.codes, input.handles)]
    similar_pairs = []
    cheaters = set()
    for i in range(len(processed_codes)):
        for j in range(i + 1, len(processed_codes)):
            lcs_score = lcs_length(processed_codes[i], processed_codes[j])
            hash1 = compute_hash(processed_codes[i], by_line=False)
            hash2 = compute_hash(processed_codes[j], by_line=False)
            common_hashes, min_hashes = compare_hashes(hash1, hash2), min(len(hash1), len(hash2))
            compare3_score = compare_hashes(hash1, hash2)
            if (lcs_score >= 1100 or 
                common_hashes >= min_hashes / 2 + min_hashes / 5 or 
                common_hashes >= 100 or 
                (common_hashes >= 70 and lcs_score >= 700) or 
                lcs_score >= len(processed_codes[i]) / 2 + len(processed_codes[i]) / 3 + min_hashes * 2 or 
                compare3_score >= 240):
                similar_pairs.append({
                    "handle1": input.handles[i],
                    "handle2": input.handles[j],
                    "lcs": lcs_score,
                    "hash_common": common_hashes,
                    "hash_min": min_hashes,
                    "compare3": compare3_score
                })
                cheaters.add(input.handles[i])
                cheaters.add(input.handles[j])
    return {
        "similar_pairs": similar_pairs,
        "num_similar_pairs": len(similar_pairs),
        "num_cheaters": len(cheaters),
        "cheaters": list(cheaters),
        "message": "Batch comparison complete"
    }

@app.post("/a", response_class=HTMLResponse)
async def analyze_code_form(request: Request, code: str = Form(...), handle: str = Form("user1")):
    success, label, confidence, handle = await classify_code(code, handle)
    if not success:
        return templates.TemplateResponse("index.html", {"request": request, "error": label})
    variables, has_long_vars = extract_variables(code)
    issues = [f"Var > {CONFIG['var_length']} chars" if has_long_vars else "", 
              "Susp com" if has_suspicious_comments(code) else ""]
    issues = [x for x in issues if x]
    is_similar, delta, delta_details = await analyze_code(code)
    model1_result = ("S" if is_similar else "N", [delta_details])
    is_plagiarized, delta, delta_details, sim_status, evidence = await detect_plagiarism(code)
    model2_result = ("S" if is_plagiarized else "N", [delta_details] + evidence)
    analysis = {"label": label, "confidence": confidence}
    report = generate_report(analysis, issues, handle, model1_result=model1_result, model2_result=model2_result)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": {"handle": handle, "report": report, "confidence": confidence, "label": label}
    })

@app.post("/c", response_class=HTMLResponse)
async def compare_codes_form(request: Request):
    form = await request.form()
    codes, handles = [], []
    i = 1
    while f"code{i}" in form and f"handle{i}" in form:
        codes.append(form[f"code{i}"])
        handles.append(form[f"handle{i}"])
        i += 1
    if len(codes) < 2:
        raise HTTPException(status_code=400, detail="Need 2+ codes")
    results = []
    for i in range(0, len(codes), 2):
        if i + 1 >= len(codes):
            break
        code1, handle1 = codes[i], handles[i]
        code2, handle2 = codes[i + 1], handles[i + 1]
        model1_result, model1_evidence = await compare_code_pair(code1, code2, handle1, handle2)
        model2_result, model2_evidence = await compare_code_pair_advanced(code1, code2, handle1, handle2)
        vars1, has_long_vars1 = extract_variables(code1)
        vars2, has_long_vars2 = extract_variables(code2)
        issues = []
        if has_long_vars1:
            issues.append(f"{handle1}: Var > {CONFIG['var_length']} chars")
        if has_long_vars2:
            issues.append(f"{handle2}: Var > {CONFIG['var_length']} chars")
        if has_suspicious_comments(code1):
            issues.append(f"{handle1}: Susp com")
        if has_suspicious_comments(code2):
            issues.append(f"{handle2}: Susp com")
        analysis = {"label": "N/A", "confidence": 0.0}
        report = generate_report(analysis, issues, f"{handle1} vs {handle2}", model1_result=model1_result, model2_result=model2_result)
        results.append({
            "handle1": handle1,
            "handle2": handle2,
            "similarity": model1_result[0],
            "message": f"{handle1} and {handle2} {'plagiarised' if any(x == 'S' for x in [model1_result[0], model2_result[0]]) else 'not plagiarised'}",
            "evidence": issues + model1_evidence + ["\nM2 Evidence:"] + model2_evidence,
            "report": report
        })
        await asyncio.sleep(2)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "batch_results": results,
        "num_similar_pairs": len(results),
        "num_cheaters": len(set([r["handle1"] for r in results] + [r["handle2"] for r in results])),
        "cheaters": list(set([r["handle1"] for r in results] + [r["handle2"] for r in results]))
    })

@app.post("/p", response_class=HTMLResponse)
async def prepare_code(request: Request, code: str = Form(...), handle: str = Form(...)):
    processed = preprocess_code(code, handle)
    return templates.TemplateResponse("index.html", {"request": request, "message": f"Code for {handle} prepared"})

@app.post("/compare_lines", response_class=JSONResponse)
async def compare_lines(request: Request):
    data = await request.json()
    code1 = data.get("code1", "")
    code2 = data.get("code2", "")
    if not code1 or not code2:
        return {"error": "Both codes are required"}
    is_similar, details = compare_codes(code1, code2)
    return {"similar": is_similar, "details": details}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)