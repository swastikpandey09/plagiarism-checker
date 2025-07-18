from __future__ import annotations
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator, EmailStr
from py7zr import SevenZipFile
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
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import clang.cindex as clang

# Logging setup
valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
log_level = environ.get("LOG_LEVEL", "INFO").upper()
if log_level not in valid_log_levels:
    log_level = "INFO"
logging.basicConfig(level=getattr(logging, log_level), format="%(levelname)s | %(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = getenv("OPENROUTER_API_KEY")
if not api_key or not api_key.startswith("sk-or-v1-"):
    logger.error("OPENROUTER_API_KEY is missing or invalid")
    raise RuntimeError("OPENROUTER_API_KEY is missing or invalid")
secret_key = getenv("SECRET_KEY")
if not secret_key:
    logger.error("SECRET_KEY is missing")
    raise RuntimeError("SECRET_KEY is missing")

# Initialize Redis
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}")
    raise RuntimeError("Redis server is not running or accessible")

# Initialize FastAPI app
app = FastAPI(title="Code Plagiarism Detector", version="0.1.0")

# Security setup
SECRET_KEY = secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Configuration
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODEL_ARCHIVE = BASE_DIR / "dataset" / "output" / "model-best" / "transformer" / "model.7z"
TEMP_MODEL_DIR = Path(tempfile.mkdtemp())
INPUT_1568 = BASE_DIR / "input1568.txt"
INPUT_1435 = BASE_DIR / "input1435.txt"
OUTPUT_1435 = BASE_DIR / "output1435.txt"
INTERACTION_LOG = BASE_DIR / "interaction_log.json"
BANNED_IPS = BASE_DIR / "banned_ips.txt"
LSTM_MODEL_PATH = BASE_DIR / "lstm_plagiarism_model.h5"

CONFIG = {
    "lcs_threshold": 0.5,
    "hash_threshold": 0.125,
    "var_length": 5,
    "comment_length": 5,
    "comment_block": 5,
    "delta_threshold": 0.10,
    "plagiarism_threshold": 0.10,
    "similarity_threshold": 0.9,
    "ast_similarity_threshold": 0.85,
    "max_logins_per_email": 2,
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

# Initialize OpenAI client
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Pydantic models
class UserRegister(BaseModel):
    email1: EmailStr
    email2: EmailStr
    password: str
    handle: str
    @validator("handle")
    def validate_handle(cls, v):
        if not v.strip() or any(c in v for c in "\n$"):
            raise ValueError("Handle must be non-empty and not contain newlines or '$'")
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Handle must be alphanumeric or underscores")
        return v
    @validator("email2")
    def emails_different(cls, v, values):
        if "email1" in values and v == values["email1"]:
            raise ValueError("Second email must be different from first email")
        return v

class SingleCodeInput(BaseModel):
    code: str
    handle: str
    @validator("handle")
    def validate_handle(cls, v):
        if not v.strip() or any(c in v for c in "\n$"):
            raise ValueError("Handle must be non-empty and not contain newlines or '$'")
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Handle must be alphanumeric or underscores")
        return v

class Token(BaseModel):
    access_token: str
    token_type: str

class HealthResponse(BaseModel):
    status: str
    models: List[str] = []

# User database
users_db: Dict[str, Dict[str, Any]] = {}
email_usage: Dict[str, int] = {}

# LSTM model setup
tokenizer = Tokenizer()
lstm_model = None
MAX_SEQUENCE_LENGTH = 1000

def init_lstm_model(vocab_size: int = 10000):
    global lstm_model
    lstm_model = Sequential([
        Embedding(vocab_size, 128, input_length=MAX_SEQUENCE_LENGTH),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def prepare_lstm_input(code: str) -> np.ndarray:
    tokens = tokenize_code(code)
    sequences = tokenizer.texts_to_sequences([tokens])
    return pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Log interaction
def log_interaction(request: Request, response: Dict[str, Any], input_data: Dict[str, Any]):
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "client_ip": request.client.host,
        "input": input_data,
        "response": response
    }
    try:
        INTERACTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        if INTERACTION_LOG.exists():
            with INTERACTION_LOG.open("r", encoding="utf-8") as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(interaction)
        with INTERACTION_LOG.open("w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")

# Train LSTM model
async def train_lstm_model():
    try:
        if not INTERACTION_LOG.exists():
            logger.info("No interaction log found for LSTM training")
            return
        with INTERACTION_LOG.open("r", encoding="utf-8") as f:
            logs = json.load(f)
        
        codes = []
        labels = []
        for log in logs:
            code = log["input"].get("code", "")
            is_plagiarized = log["response"].get("result", {}).get("report", "").lower().find("ai-generated or plagiarized") != -1
            codes.append(code)
            labels.append(1 if is_plagiarized else 0)
        
        if len(codes) < 2:
            logger.info("Insufficient data for LSTM training")
            return
        
        tokenizer.fit_on_texts(codes)
        X = pad_sequences(tokenizer.texts_to_sequences(codes), maxlen=MAX_SEQUENCE_LENGTH)
        y = np.array(labels)
        
        init_lstm_model(len(tokenizer.word_index) + 1)
        lstm_model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
        LSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        lstm_model.save(LSTM_MODEL_PATH)
        logger.info("LSTM model trained and saved")
    except Exception as e:
        logger.error(f"Failed to train LSTM model: {e}")

# IP ban management
def is_ip_banned(ip: str) -> bool:
    try:
        BANNED_IPS.parent.mkdir(parents=True, exist_ok=True)
        if BANNED_IPS.exists():
            with BANNED_IPS.open("r", encoding="utf-8") as f:
                return ip in {line.strip() for line in f}
        return False
    except Exception as e:
        logger.error(f"Error checking banned IPs: {e}")
        return False

def ban_ip(ip: str):
    try:
        BANNED_IPS.parent.mkdir(parents=True, exist_ok=True)
        with BANNED_IPS.open("a", encoding="utf-8") as f:
            f.write(f"{ip}\n")
        logger.info(f"Banned IP: {ip}")
    except Exception as e:
        logger.error(f"Failed to ban IP {ip}: {e}")

# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None or email not in users_db:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return users_db[email]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# AST-based comparison
def parse_code_to_ast(code: str) -> Optional[clang.cindex.Cursor]:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        index = clang.Index.create()
        translation_unit = index.parse(temp_file_path, args=['-std=c++17'])
        Path(temp_file_path).unlink()
        if translation_unit.diagnostics:
            for diag in translation_unit.diagnostics:
                logger.warning(f"AST parsing diagnostic: {diag}")
            return None
        return translation_unit.cursor
    except Exception as e:
        logger.error(f"Failed to parse code to AST: {e}")
        return None

def serialize_ast(cursor: clang.cindex.Cursor, depth: int = 0) -> List[Dict[str, Any]]:
    if not cursor:
        return []
    nodes = []
    node = {
        "kind": cursor.kind.name,
        "children": []
    }
    for child in cursor.get_children():
        node["children"].extend(serialize_ast(child, depth + 1))
    nodes.append(node)
    return nodes

def compare_ast(ast1: List[Dict[str, Any]], ast2: List[Dict[str, Any]]) -> float:
    if not ast1 or not ast2:
        return 0.0
    def ast_to_string(ast: List[Dict[str, Any]]) -> str:
        result = []
        for node in ast:
            result.append(node["kind"])
            result.extend(ast_to_string(node["children"]))
        return "|".join(result)
    
    str1, str2 = ast_to_string(ast1), ast_to_string(ast2)
    lcs_len = lcs_length(str1, str2)
    return lcs_len / max(len(str1), len(str2)) if str1 and str2 else 0.0

# Helper functions
def check_openrouter_health(api_key: str, base_url: str = "https://openrouter.ai/api/v1") -> Tuple[bool, List[str]]:
    try:
        endpoint = f"{base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(endpoint, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        if "data" in data and isinstance(data["data"], list):
            model_ids = [model["id"] for model in data["data"]]
            logger.info("OpenRouter API health check passed")
            return True, model_ids
        return False, []
    except Exception as e:
        logger.error(f"OpenRouter API health check failed: {str(e)}")
        return False, []

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
    lcs_ratio = lcs_score / max(len(cleaned1), len(cleaned2)) if cleaned1 and cleaned2 else 0
    hash1, hash2 = compute_hash(code1), compute_hash(code2)
    common_hashes = len(set(hash1) & set(hash2))
    hash_ratio = common_hashes / max(len(hash1), len(hash2)) if hash1 and hash2 else 0
    ast1 = parse_code_to_ast(code1)
    ast2 = parse_code_to_ast(code2)
    ast_similarity = compare_ast(serialize_ast(ast1), serialize_ast(ast2)) if ast1 and ast2 else 0.0
    is_similar = lcs_ratio >= CONFIG["lcs_threshold"] or hash_ratio >= CONFIG["hash_threshold"] or ast_similarity >= CONFIG["ast_similarity_threshold"]
    return is_similar, f"LCS: {lcs_ratio:.2f}, Hash: {hash_ratio:.2f}, AST: {ast_similarity:.2f}"

def compare_with_previous_submission(current_code: str, handle: str) -> Tuple[bool, str]:
    try:
        INPUT_1435.parent.mkdir(parents=True, exist_ok=True)
        if not INPUT_1435.exists():
            return False, "No previous submissions"
        
        with INPUT_1435.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        
        previous_code = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith(f"xc9@{handle}"):
                if i + 1 < len(lines) and not lines[i + 1].startswith("xc9@"):
                    previous_code = lines[i + 1].strip()
                break
        
        if not previous_code:
            return False, "No previous submission found"
        
        cleaned_current = clean_code(current_code)
        cleaned_previous = clean_code(previous_code)
        lcs_score = lcs_length(cleaned_current, cleaned_previous)
        lcs_ratio = lcs_score / max(len(cleaned_current), len(cleaned_previous)) if cleaned_current and cleaned_previous else 0
        tokens1 = tokenize_code(cleaned_current)
        tokens2 = tokenize_code(cleaned_previous)
        structural_similarity = len(set(tokens1) & set(tokens2)) / max(len(tokens1), len(tokens2)) if tokens1 and tokens2 else 0
        ast1 = parse_code_to_ast(current_code)
        ast2 = parse_code_to_ast(previous_code)
        ast_similarity = compare_ast(serialize_ast(ast1), serialize_ast(ast2)) if ast1 and ast2 else 0.0
        is_suspicious = lcs_ratio >= CONFIG["similarity_threshold"] or structural_similarity >= CONFIG["similarity_threshold"] or ast_similarity >= CONFIG["ast_similarity_threshold"]
        return is_suspicious, f"LCS: {lcs_ratio:.2f}, Structural: {structural_similarity:.2f}, AST: {ast_similarity:.2f}"
    except Exception as e:
        logger.error(f"Error comparing with previous submission: {e}")
        return False, str(e)

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
                "Wrap the generated code in a ```cpp code block."
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
    lcs_ratio = lcs_score / max(len(cleaned_code), len(clean_code(generated_code))) if cleaned_code else 0
    ast1 = parse_code_to_ast(code)
    ast2 = parse_code_to_ast(generated_code)
    ast_similarity = compare_ast(serialize_ast(ast1), serialize_ast(ast2)) if ast1 and ast2 else 0.0
    evidence = [f"Delta: {delta:.4f}", f"LCS: {lcs_ratio:.2f}", f"AST: {ast_similarity:.2f}"]
    is_plagiarized = delta < CONFIG["plagiarism_threshold"] or lcs_ratio >= CONFIG["lcs_threshold"] or ast_similarity >= CONFIG["ast_similarity_threshold"]
    return is_plagiarized, delta, f"Delta: {delta:.4f}", "S" if is_plagiarized else "N", evidence

def query_api(messages: List[Dict[str, str]], model: str = "moonshotai/kimi-k2:free", temp: float = 0.7, max_tokens: int = 2000) -> str:
    fallback_models = ["qwen/qwen3-8b:free", "mistralai/mistral-7b-instruct:free"]
    models_to_try = [model] + fallback_models
    for current_model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens
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
                if match:
                    return match.group(1).strip()
                logger.error(f"No valid C++ code block found in API response")
                return content.strip()
        except AuthenticationError as e:
            logger.error(f"Authentication failed with model {current_model}: {str(e)}")
            if current_model != models_to_try[-1]:
                continue
            return ""
        except RateLimitError as e:
            if current_model != models_to_try[-1]:
                delay = 60.0
                logger.warning(f"Rate limit exceeded for model {current_model}. Waiting {delay:.2f}s")
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

def process_code_submission(code: str, handle: str) -> None:
    try:
        cleaned_code = clean_code(code)
        if not cleaned_code:
            raise ValueError("Empty code after cleaning")
        INPUT_1568.parent.mkdir(parents=True, exist_ok=True)
        with INPUT_1568.open("w", encoding="utf-8") as f:
            f.write(f"{cleaned_code}\nRqq7\n{handle}\n")
        INPUT_1435.parent.mkdir(parents=True, exist_ok=True)
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
        OUTPUT_1435.parent.mkdir(parents=True, exist_ok=True)
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

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    await FastAPILimiter.init(redis_client)
    is_healthy, models = check_openrouter_health(api_key)
    if not is_healthy:
        logger.error("Startup failed: OpenRouter API unavailable")
        raise RuntimeError("Startup failed: OpenRouter API unavailable")
    app.state.available_models = models
    if LSTM_MODEL_PATH.exists():
        global lstm_model
        lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
        logger.info("Loaded existing LSTM model")
    logger.info(f"Available models: {models}")

@app.on_event("shutdown")
async def cleanup():
    try:
        if TEMP_MODEL_DIR.exists():
            shutil.rmtree(TEMP_MODEL_DIR, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory {TEMP_MODEL_DIR}")
    except Exception as e:
        logger.error(f"Failed to clean up: {str(e)}")

# Endpoints
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if is_ip_banned(request.client.host):
        raise HTTPException(status_code=403, detail="IP address is banned")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        is_healthy, models = check_openrouter_health(api_key)
        if not is_healthy:
            return {"status": "unhealthy", "error": "OpenRouter API unavailable"}
        return {"status": "healthy", "models": models}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/register", response_class=HTMLResponse)
async def register(request: Request, email1: str = Form(...), email2: str = Form(...), password: str = Form(...), handle: str = Form(...)):
    try:
        user_data = UserRegister(email1=email1, email2=email2, password=password, handle=handle)
        if email_usage.get(email1, 0) >= CONFIG["max_logins_per_email"] or email_usage.get(email2, 0) >= CONFIG["max_logins_per_email"]:
            raise HTTPException(status_code=400, detail="Email usage limit exceeded")
        
        if email1 in users_db:
            raise HTTPException(status_code=400, detail="Primary email already registered")
        
        hashed_password = get_password_hash(password)
        users_db[email1] = {
            "email1": email1,
            "email2": email2,
            "password": hashed_password,
            "handle": handle
        }
        email_usage[email1] = email_usage.get(email1, 0) + 1
        email_usage[email2] = email_usage.get(email2, 0) + 1
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": f"User {handle} registered successfully"
        })
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Registration failed: {str(e)}"
        })

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/analyze", response_class=HTMLResponse, dependencies=[Depends(RateLimiter(times=10, seconds=3600))])
async def analyze_code(request: Request, code: str = Form(..., max_length=100_000), handle: str = Form(...), current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        if is_ip_banned(request.client.host):
            raise HTTPException(status_code=403, detail="IP address is banned")
        
        SingleCodeInput(code=code, handle=handle)
        health = await health_check()
        if health["status"] != "healthy":
            raise HTTPException(status_code=503, detail=f"OpenRouter API unavailable: {health.get('error', 'Unknown error')}")
        
        is_suspicious, similarity_details = compare_with_previous_submission(code, handle)
        if is_suspicious:
            ban_ip(request.client.host)
            OUTPUT_1435.parent.mkdir(parents=True, exist_ok=True)
            with OUTPUT_1435.open("a", encoding="utf-8") as f:
                f.write(f"Cheater detected: {handle} - Suspicious similarity with previous submission: {similarity_details}\n")
            raise HTTPException(status_code=403, detail="Suspicious code similarity detected. IP banned.")
        
        success, label, confidence, handle = classify_code(code, handle)
        if not success:
            raise HTTPException(status_code=400, detail=label)
        
        variables, has_long_vars = extract_variables(code)
        issues = [
            f"Variables longer than {CONFIG['var_length']} characters" if has_long_vars else "",
            "Suspicious comments" if has_suspicious_comments(code) else ""
        ]
        ast = parse_code_to_ast(code)
        if not ast:
            issues.append("Failed to parse code to AST")
        issues = [x for x in issues if x]
        
        lstm_confidence = 0.0
        if lstm_model:
            lstm_input = prepare_lstm_input(code)
            lstm_confidence = float(lstm_model.predict(lstm_input)[0][0])
            if lstm_confidence > 0.8:
                issues.append(f"LSTM model indicates AI-generated code (confidence: {lstm_confidence:.2%})")
        
        is_plagiarized, delta, delta_details, sim_status, evidence = await detect_plagiarism(code, handle)
        model1_result = ("S" if is_plagiarized else "N", [delta_details])
        model2_result = (sim_status, evidence)
        similar_handles = detect_similar_codes(code, handle)
        
        process_code_submission(code, handle)
        analysis = {"label": label, "confidence": confidence, "similar_handles": similar_handles}
        report = generate_report(analysis, issues, handle, model1_result, model2_result)
        
        OUTPUT_1435.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_1435.open("a+", encoding="utf-8") as f:
            f.write(f"Handle: {handle} ({label}-generated)\n")
            f.write(f"Confidence: {confidence:.2%}\n")
            f.write(f"LSTM Confidence: {lstm_confidence:.2%}\n")
            f.write(f"Plagiarism: {'Detected' if is_plagiarized else 'Not detected'}\n")
            f.write("\n".join(evidence) + "\n\n")
        
        response_data = {
            "result": {"handle": handle, "report": report, "confidence": confidence, "label": label, "lstm_confidence": lstm_confidence}
        }
        log_interaction(request, response_data, {"code": code, "handle": handle})
        
        await train_lstm_model()
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": response_data["result"]
        })
    except Exception as e:
        logger.error(f"Analyze code failed: {str(e)}")
        log_interaction(request, {"error": str(e)}, {"code": code, "handle": handle})
        raise HTTPException(status_code=500, detail=f"Analyze code failed: {str(e)}")

@app.post("/prepare", response_class=HTMLResponse, dependencies=[Depends(RateLimiter(times=10, seconds=3600))])
async def prepare_submission(request: Request, code: str = Form(...), handle: str = Form(...), current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        if is_ip_banned(request.client.host):
            raise HTTPException(status_code=403, detail="IP address is banned")
        SingleCodeInput(code=code, handle=handle)
        process_code_submission(code, handle)
        response_data = {"message": f"Code for {handle} prepared"}
        log_interaction(request, response_data, {"code": code, "handle": handle})
        return templates.TemplateResponse("index.html", {"request": request, "message": f"Code for {handle} prepared"})
    except Exception as e:
        logger.error(f"Prepare submission failed: {e}")
        log_interaction(request, {"error": str(e)}, {"code": code, "handle": handle})
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Prepare submission failed: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    port = int(environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
