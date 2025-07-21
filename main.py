from __future__ import annotations
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, field_validator
import py7zr
import tempfile
import shutil
from os import environ, getenv
from dotenv import load_dotenv
from openai import OpenAI, APIError, AuthenticationError
import requests
from datetime import datetime, timedelta
from collections import Counter
import math
from fastapi.security import OAuth2PasswordBearer
from jwt import PyJWTError, decode, encode
from passlib.context import CryptContext
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import clang.cindex as clang
import ast

# Logging setup
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "level": record.levelname,
            "timestamp": datetime.now().isoformat(),
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno
        }
        return json.dumps(log_data)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.handlers = [handler]
log_level = environ.get("LOG_LEVEL", "DEBUG").upper()
valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
logging.basicConfig(level=getattr(logging, log_level if log_level in valid_log_levels else "DEBUG"))

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY")
SECRET_KEY = getenv("SECRET_KEY")
if not OPENROUTER_API_KEY or not OPENROUTER_API_KEY.startswith("sk-or-v1-"):
    logger.error("OPENROUTER_API_KEY is missing or invalid")
    raise RuntimeError("OPENROUTER_API_KEY is missing or invalid")
if not SECRET_KEY:
    logger.error("SECRET_KEY is missing")
    raise RuntimeError("SECRET_KEY is missing")
masked_key = f"{OPENROUTER_API_KEY[:10]}...{OPENROUTER_API_KEY[-4:]}" if OPENROUTER_API_KEY else "None"
logger.info(f"Loaded OPENROUTER_API_KEY: {masked_key}")

# Initialize OpenAI client
base_url = getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
correct_base_url = base_url if base_url.endswith("/v1") else "https://openrouter.ai/api/v1"
if base_url != correct_base_url:
    logger.warning(f"Environment OPENROUTER_BASE_URL ({base_url}) overridden with {correct_base_url}")
client = OpenAI(base_url=correct_base_url, api_key=OPENROUTER_API_KEY)

# FastAPI app setup
app = FastAPI(title="Code Plagiarism Detector", version="0.2.5")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Configuration
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
    "delta_threshold": 0.5,
    "plagiarism_threshold": 0.10,
    "ast_similarity_threshold": 0.85,
    "similarity_threshold": 0.9,
    "max_logins_per_email": 2,
}

# Token frequencies for C++
HUMAN_FREQUENCIES = {
    'int': 7.22, 'if': 3.72, 'for': 2.93, 'cin': 1.81, 'cout': 1.22, '#include': 1.01,
    '<iostream>': 1.01, 'namespace': 1.0, 'std': 1.0, 'return': 1.35, 'while': 1.5,
    '>>': 1.2, '<<': 1.2, 'endl': 1.0, 'main': 1.0, '+': 1.0, ' ': 50.0,
    '}': 1.2, '{': 1.2, '<': 1.0, '>': 1.0, '-': 1.0, '*': 1.0, '/': 1.0,
    '=': 1.0, '==': 1.0, '!=': 1.0, '<=': 1.0, '>=': 1.0, '&&': 1.0, '||': 1.0,
    '!': 1.0, '&': 1.0, '|': 1.0, '^': 1.0, '~': 1.0, '(': 50.0, ')': 50.0,
    '[': 1.0, ']': 1.0, ';': 50.0, ',': 1.0, '.': 1.0, '"': 1.0, '\n': 50.0, '\t': 50.0
}

# Security setup
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Global state
users_db: Dict[str, Dict[str, Any]] = {}
email_usage: Dict[str, Dict[str, Any]] = {}
tokenizer = Tokenizer()
lstm_model = None
MAX_SEQUENCE_LENGTH = 1000
app.state.available_models = []

# Initialize tokenizer
if not tokenizer.word_index:
    tokenizer.fit_on_texts(["def main():\n    print(\"Hello World\")", "import os\nimport sys"])

# Pydantic models
class UserRegister(BaseModel):
    email1: EmailStr
    email2: EmailStr
    password: str
    handle: str

    @field_validator("handle")
    def validate_handle(cls, v):
        if not v.strip() or any(c in v for c in "\n$"):
            raise ValueError("Handle must be non-empty and not contain newlines or '$'")
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Handle must be alphanumeric or underscores")
        return v

    @field_validator("email2")
    def emails_different(cls, v, values):
        if "email1" in values and v == values["email1"]:
            raise ValueError("Emails must be different")
        return v

class CodeInput(BaseModel):
    code: str
    handle: str
    language: str

    @field_validator("handle")
    def validate_handle(cls, v):
        if not v.strip() or any(c in v for c in "\n$"):
            raise ValueError("Handle must be non-empty and not contain newlines or '$'")
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Handle must be alphanumeric or underscores")
        return v

    @field_validator("language")
    def validate_language(cls, v):
        if v not in ["cpp", "python"]:
            raise ValueError("Language must be 'cpp' or 'python'")
        return v

class SingleCodeInput(BaseModel):
    code: str
    handle: str = "triumph"

    @field_validator("handle")
    def validate_handle(cls, v):
        if not v.strip() or any(c in v for c in "\n$"):
            raise ValueError("Handle must be non-empty and not contain newlines or '$'")
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Handle must be alphanumeric or underscores")
        return v

class HealthResponse(BaseModel):
    status: str
    models: List[str] = []

# Helper functions
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

def load_lstm_model():
    global lstm_model
    if lstm_model is None:
        try:
            lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
            logger.info("LSTM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading LSTM model: {e}")
            init_lstm_model()
            logger.warning("Initialized new LSTM model")

def preprocess_text_for_lstm(text: str) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequences

def predict_plagiarism_lstm(text: str) -> float:
    if lstm_model is None:
        load_lstm_model()
    processed_text = preprocess_text_for_lstm(text)
    prediction = lstm_model.predict(processed_text, verbose=0)[0][0]
    return float(prediction)

async def train_lstm_model():
    try:
        if not INTERACTION_LOG.exists():
            return
        with INTERACTION_LOG.open("r", encoding="utf-8") as f:
            logs = json.load(f)
        codes, labels = [], []
        for log in logs:
            code = log["input"].get("code", "")
            is_plagiarized = "plagiarized" in log["response"].get("result", {}).get("report", "").lower()
            codes.append(code)
            labels.append(1 if is_plagiarized else 0)
        if len(codes) < 2:
            return
        tokenizer.fit_on_texts(codes)
        X = pad_sequences(tokenizer.texts_to_sequences(codes), maxlen=MAX_SEQUENCE_LENGTH)
        y = np.array(labels)
        init_lstm_model(len(tokenizer.word_index) + 1)
        lstm_model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2, verbose=0)
        LSTM_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        lstm_model.save(LSTM_MODEL_PATH)
        logger.info("LSTM model trained")
    except Exception as e:
        logger.error(f"Failed to train LSTM: {e}")

def is_ip_banned(ip: str) -> bool:
    try:
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
        logger.error(f"Failed to ban IP: {e}")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user_from_cookie(request: Request):
    access_token = request.cookies.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("sub")
        if email is None or email not in users_db:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return users_db[email]
    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

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
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[{}();,=<>+\-*&]|[0-9]+|[^\w\s]', line)
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

def compute_frequency_features(code: str) -> Dict[str, float]:
    tokens = tokenize_code(code)
    total_chars = sum(len(str(t)) for t in tokens)
    if total_chars == 0:
        return {}
    token_counts = Counter(tokens)
    return {token: (count / total_chars) * 1000 for token, count in token_counts.items()}

def classify_code(code: str, handle: str) -> Dict[str, Any]:
    if not code.strip():
        return {"success": False, "label": "Empty code", "confidence": 0.0, "handle": handle}
    code_freq = compute_frequency_features(code)
    if not code_freq:
        return {"success": False, "label": "Invalid code", "confidence": 0.0, "handle": handle}
    deviation = sum((code_freq.get(token, 0) - expected) ** 2 for token, expected in HUMAN_FREQUENCIES.items())
    deviation = math.sqrt(deviation / len(HUMAN_FREQUENCIES))
    threshold = 50.0
    confidence = min(0.99, 1.0 / (1.0 + math.exp(-0.05 * (threshold - deviation))))
    label = "Human" if deviation < threshold else "AI"
    return {"success": True, "label": label, "confidence": confidence, "handle": handle}

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

def jaccard_similarity(s1: str, s2: str) -> float:
    set1 = set(s1.split())
    set2 = set(s2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def cosine_similarity(text1: str, text2: str) -> float:
    words1 = Counter(text1.lower().split())
    words2 = Counter(text2.lower().split())
    all_words = list(set(words1.keys()) | set(words2.keys()))
    vec1 = [words1[word] for word in all_words]
    vec2 = [words2[word] for word in all_words]
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v1**2 for v1 in vec1))
    magnitude2 = math.sqrt(sum(v2**2 for v2 in vec2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

def winnowing_hash_similarity(text1: str, text2: str, k: int = 5, w: int = 10) -> float:
    def get_ngrams(text, n):
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    def get_hashes(ngrams):
        return [hash(ngram) for ngram in ngrams]
    def get_fingerprints(hashes, window_size):
        fingerprints = set()
        if len(hashes) < window_size:
            return set(hashes)
        for i in range(len(hashes) - window_size + 1):
            window = hashes[i : i + window_size]
            min_hash = min(window)
            fingerprints.add(min_hash)
        return fingerprints
    ngrams1 = get_ngrams(text1, k)
    ngrams2 = get_ngrams(text2, k)
    hashes1 = get_hashes(ngrams1)
    hashes2 = get_hashes(ngrams2)
    fingerprints1 = get_fingerprints(hashes1, w)
    fingerprints2 = get_fingerprints(hashes2, w)
    intersection = len(fingerprints1.intersection(fingerprints2))
    union = len(fingerprints1.union(fingerprints2))
    return intersection / union if union > 0 else 0.0

def cpp_ast_similarity(code1: str, code2: str) -> float:
    def parse_code_to_ast(code: str) -> Optional[clang.cindex.Cursor]:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            index = clang.Index.create()
            tu = index.parse(temp_file_path, args=['-std=c++17'])
            Path(temp_file_path).unlink()
            if tu.diagnostics:
                logger.warning(f"AST parsing issues: {[diag for diag in tu.diagnostics]}")
                return None
            return tu.cursor
        except Exception as e:
            logger.error(f"AST parsing failed: {e}")
            return None
    def serialize_ast(cursor: clang.cindex.Cursor) -> List[Dict[str, Any]]:
        if not cursor:
            return []
        nodes = [{"kind": cursor.kind.name, "children": []}]
        for child in cursor.get_children():
            nodes[0]["children"].extend(serialize_ast(child))
        return nodes
    ast1 = parse_code_to_ast(code1)
    ast2 = parse_code_to_ast(code2)
    if not ast1 or not ast2:
        return 0.0
    def ast_to_string(ast: List[Dict[str, Any]]) -> str:
        result = []
        for node in ast:
            result.append(node["kind"])
            result.extend(ast_to_string(node["children"]))
        return "|".join(result)
    str1, str2 = ast_to_string(serialize_ast(ast1)), ast_to_string(serialize_ast(ast2))
    lcs_len = lcs_length(str1, str2)
    return lcs_len / max(len(str1), len(str2)) if str1 and str2 else 0.0

def python_ast_similarity(code1: str, code2: str) -> float:
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
    except SyntaxError:
        return 0.0
    def get_nodes(tree):
        nodes = []
        for node in ast.walk(tree):
            nodes.append(type(node))
        return nodes
    nodes1 = get_nodes(tree1)
    nodes2 = get_nodes(tree2)
    common_nodes = len(set(nodes1).intersection(set(nodes2)))
    total_nodes = len(set(nodes1).union(set(nodes2)))
    return common_nodes / total_nodes if total_nodes > 0 else 0.0

def calculate_plagiarism_score(text1: str, text2: str) -> Dict[str, float]:
    lcs = lcs_length(text1, text2) / max(len(text1), len(text2)) if text1 and text2 else 0.0
    jaccard = jaccard_similarity(text1, text2)
    cosine = cosine_similarity(text1, text2)
    winnowing = winnowing_hash_similarity(text1, text2)
    ast_sim = python_ast_similarity(text1, text2)
    lstm_pred = predict_plagiarism_lstm(text1)
    combined_score = (
        lcs * 0.2 +
        jaccard * 0.15 +
        cosine * 0.15 +
        winnowing * 0.2 +
        ast_sim * 0.15 +
        lstm_pred * 0.15
    )
    return {
        "lcs_similarity": lcs,
        "jaccard_similarity": jaccard,
        "cosine_similarity": cosine,
        "winnowing_hash_similarity": winnowing,
        "ast_similarity": ast_sim,
        "lstm_prediction": lstm_pred,
        "combined_score": combined_score
    }

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
    return is_plagiarized, delta, f"Delta: {delta:.4f}", "S" if is_plagiarized else "N", evidence

def compare_with_previous_submission(code: str, handle: str, language: str) -> Dict[str, Any]:
    try:
        if not INPUT_1435.exists():
            return {"is_suspicious": False, "details": "No previous submissions"}
        with INPUT_1435.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        previous_code = None
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith(f"xc9@{handle}"):
                if i + 1 < len(lines) and not lines[i + 1].startswith("xc9@"):
                    previous_code = lines[i + 1].strip()
                break
        if not previous_code:
            return {"is_suspicious": False, "details": "No previous submission found"}
        cleaned_current = clean_code(code)
        cleaned_previous = clean_code(previous_code)
        lcs_score = lcs_length(cleaned_current, cleaned_previous)
        lcs_ratio = lcs_score / max(len(cleaned_current), len(cleaned_previous)) if cleaned_current and cleaned_previous else 0
        ast_sim = cpp_ast_similarity(code, previous_code) if language == "cpp" else python_ast_similarity(code, previous_code)
        is_suspicious = lcs_ratio >= CONFIG["similarity_threshold"] or ast_sim >= CONFIG["ast_similarity_threshold"]
        return {"is_suspicious": is_suspicious, "details": f"LCS: {lcs_ratio:.2f}, AST: {ast_sim:.2f}"}
    except Exception as e:
        logger.error(f"Error comparing with previous submission: {e}")
        return {"is_suspicious": False, "details": str(e)}

async def detect_plagiarism(code: str, handle: str, language: str) -> Dict[str, Any]:
    if not code:
        return {"is_plagiarized": False, "delta": 0.0, "details": "Empty code", "status": "N", "evidence": ["Empty code"]}
    cleaned_code = clean_code(code)
    intent_prompt = [
        {"role": "system", "content": f"Describe the purpose of the following {language.upper()} code concisely."},
        {"role": "user", "content": f"```{language}\n{code}\n```"}
    ]
    intent = await query_api(intent_prompt)
    if not intent:
        return {"is_plagiarized": False, "delta": 0.0, "details": "Failed to get intent", "status": "N", "evidence": ["Failed to get intent"]}
    generate_prompt = [
        {"role": "system", "content": f"Generate {language.upper()} code solving the described problem using modern {language.upper()} practices."},
        {"role": "user", "content": intent}
    ]
    generated = await query_api(generate_prompt)
    match = re.search(rf"```{language}\n(.*?)```", generated, re.DOTALL)
    generated_code = match.group(1).strip() if match else generated.strip()
    if not generated_code:
        return {"is_plagiarized": False, "delta": 0.0, "details": "No generated code", "status": "N", "evidence": ["No generated code"]}
    is_plagiarized, delta, delta_details, sim_status, evidence = await detect_plagiarism_with_generated(code, generated_code)
    ast_sim = cpp_ast_similarity(code, generated_code) if language == "cpp" else python_ast_similarity(code, generated_code)
    evidence.append(f"AST: {ast_sim:.2f}")
    is_plagiarized = is_plagiarized or ast_sim >= CONFIG["ast_similarity_threshold"]
    if language == "python" and INPUT_1568.exists():
        reference_score = calculate_plagiarism_score(code, INPUT_1568.read_text())
        evidence.append(f"Reference Similarity: {reference_score['combined_score']:.2f}")
        is_plagiarized = is_plagiarized or reference_score["combined_score"] > CONFIG["plagiarism_threshold"]
    return {
        "is_plagiarized": is_plagiarized,
        "delta": delta,
        "details": f"Delta: {delta:.4f}, AST: {ast_sim:.2f}",
        "status": sim_status,
        "evidence": evidence
    }

async def query_api(messages: List[Dict[str, str]], model: str = None, temp: float = 0.7, max_tokens: int = 2000) -> str:
    api_key = getenv("OPENROUTER_API_KEY")
    if not api_key or not api_key.startswith("sk-or-v1-"):
        logger.error("OPENROUTER_API_KEY is missing or invalid in query_api")
        return ""
    masked_key = f"{api_key[:10]}...{api_key[-4:]}" if api_key else "None"
    logger.debug(f"Using API key: {masked_key} for query_api")
    if not model:
        available_models = getattr(app.state, "available_models", [])
        model = next((m for m in available_models if m == "deepseek/deepseek-r1-0528:free"), "deepseek/deepseek-r1-0528:free")
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
                    match = re.search(r"```(cpp|python)\n(.*?)```", content, re.DOTALL)
                    if match:
                        return match.group(2).strip()
                    logger.error(f"No valid code block in response: {content[:1000]}")
                    return content.strip()
                return code.strip()
            except json.JSONDecodeError:
                match = re.search(r"```(cpp|python)\n(.*?)```", content, re.DOTALL)
                if match:
                    return match.group(2).strip()
                logger.error(f"No valid code block in response: {content[:1000]}")
                return content.strip()
        except AuthenticationError as e:
            logger.error(f"Authentication failed with model {current_model}: {str(e)}")
            try:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"}
                }
                http_response = requests.post(f"{client.base_url.rstrip('/')}/chat/completions", headers=headers, json=payload, timeout=10)
                if http_response.status_code == 200:
                    data = http_response.json()
                    content = data['choices'][0]['message']['content'].strip()
                    try:
                        parsed = json.loads(content)
                        code = parsed.get("code", "")
                        if not code:
                            match = re.search(r"```(cpp|python)\n(.*?)```", content, re.DOTALL)
                            if match:
                                return match.group(2).strip()
                            logger.error(f"No valid code block in fallback response: {content[:1000]}")
                            return content.strip()
                        return code.strip()
                    except json.JSONDecodeError:
                        match = re.search(r"```(cpp|python)\n(.*?)```", content, re.DOTALL)
                        if match:
                            return match.group(2).strip()
                        logger.error(f"No valid code block in fallback response: {content[:1000]}")
                        return content.strip()
                elif http_response.status_code in (401, 404, 429) and current_model != models_to_try[-1]:
                    delay = 60.0
                    logger.warning(f"Fallback HTTP {http_response.status_code} with model {current_model}. Waiting {delay:.2f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Fallback HTTP failed: {http_response.status_code}")
                    return ""
            except Exception as e:
                logger.error(f"Fallback HTTP failed: {str(e)}")
                return ""
        except APIError as e:
            if ("401" in str(e) or "404" in str(e) or "429" in str(e)) and current_model != models_to_try[-1]:
                delay = 60.0 if "429" in str(e) or "404" in str(e) else 0
                logger.warning(f"API error with model {current_model}: {str(e)}. Waiting {delay:.2f}s")
                time.sleep(delay)
                continue
            logger.error(f"OpenRouter API error with model {current_model}: {str(e)}")
            return ""
    logger.error(f"All models failed: {models_to_try}")
    return ""

def process_code_submission(code: str, handle: str):
    try:
        cleaned_code = clean_code(code)
        if not cleaned_code:
            raise ValueError("Empty code")
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
    except Exception as e:
        logger.error(f"Failed to process submission: {e}")
        raise

def detect_similar_codes(code: str, handle: str, language: str) -> List[str]:
    try:
        if not INPUT_1435.exists():
            return []
        with INPUT_1435.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        codes = {}
        current_handle = None
        for line in lines:
            if line.startswith("xc9@"):
                current_handle = line[4:].strip()
            elif current_handle and line.strip() and not line.startswith("Rqq7"):
                codes[current_handle] = line.strip()
        cleaned_code = clean_code(code)
        similar_handles = []
        for other_handle, other_code in codes.items():
            if other_handle != handle:
                lcs_score = lcs_length(cleaned_code, clean_code(other_code))
                lcs_ratio = lcs_score / max(len(cleaned_code), len(clean_code(other_code))) if cleaned_code else 0
                ast_sim = cpp_ast_similarity(cleaned_code, other_code) if language == "cpp" else python_ast_similarity(cleaned_code, other_code)
                if lcs_ratio >= CONFIG["lcs_threshold"] or ast_sim >= CONFIG["ast_similarity_threshold"]:
                    similar_handles.append(f"{other_handle}: LCS: {lcs_ratio:.2f}, AST: {ast_sim:.2f}")
        OUTPUT_1435.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_1435.open("w", encoding="utf-8") as f:
            f.write(f"{len(similar_handles)} similar pairs detected\n\n")
            f.write(f"{len(similar_handles)} cheaters detected:\n")
            f.write("\n".join(similar_handles) + "\n" if similar_handles else "None\n")
        return similar_handles
    except Exception as e:
        logger.error(f"Failed to detect similar codes: {e}")
        return []

def generate_report(analysis: Dict[str, Any], plagiarism: Dict[str, Any], handle: str, 
                   model1_result: Optional[Tuple[str, List[str]]] = None, 
                   model2_result: Optional[Tuple[str, List[str]]] = None) -> str:
    report = f"Plagiarism Report for {handle}\n\n"
    report += f"AI/Human Classification: {analysis['label']} (Confidence: {analysis['confidence']:.2%})\n"
    report += f"Plagiarism Check: {'Detected' if plagiarism['is_plagiarized'] else 'Not detected'}\n"
    report += "Evidence:\n" + ("\n".join([f"- {x}" for x in plagiarism["evidence"]]) if plagiarism["evidence"] else "- None\n")
    if model1_result:
        report += f"\nBasic Plagiarism Check: {model1_result[0]}\nEvidence:\n" + "\n".join([f"- {x}" for x in model1_result[1]]) + "\n"
    if model2_result:
        report += f"\nAdvanced Plagiarism Check: {model2_result[0]}\nEvidence:\n" + "\n".join([f"- {x}" for x in model2_result[1]]) + "\n"
    report += f"Similar Handles: {', '.join(analysis.get('similar_handles', [])) or 'None'}\n"
    variables, has_long_vars = extract_variables(analysis.get("code", ""))
    issues = [f"Variables longer than {CONFIG['var_length']} characters" if has_long_vars else "",
              "Suspicious comments" if has_suspicious_comments(analysis.get("code", "")) else ""]
    issues = [x for x in issues if x]
    if issues:
        report += "Issues:\n" + "\n".join([f"- {x}" for x in issues]) + "\n"
    flags = sum(1 for x in [
        analysis['label'] == "AI",
        plagiarism['is_plagiarized'],
        analysis.get('similar_handles', []),
        model1_result and model1_result[0] == "S",
        model2_result and model2_result[0] == "S"
    ] if x)
    verdict = "Plagiarized" if flags >= 2 else "Human-written"
    report += f"\nVerdict: {verdict}\n"
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

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")
    try:
        api_key = getenv("OPENROUTER_API_KEY")
        is_healthy, models = check_openrouter_health(api_key)
        if not is_healthy:
            logger.error("Startup failed: OpenRouter API unavailable")
            raise RuntimeError("OpenRouter API unavailable")
        app.state.available_models = models
        logger.info(f"Available models: {models}")
        load_lstm_model()
        if MODEL_ARCHIVE.exists():
            logger.info(f"Model archive exists at {MODEL_ARCHIVE}, skipping spaCy load")
        else:
            logger.warning(f"Model archive {MODEL_ARCHIVE} not found")
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

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
        logger.error(f"Failed to clean up: {e}")

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
        logger.error("OpenRouter API response does not contain valid model list")
        return False, []
    except Exception as e:
        logger.error(f"OpenRouter API health check failed: {str(e)}")
        return False, []

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        api_key = getenv("OPENROUTER_API_KEY")
        is_healthy, models = check_openrouter_health(api_key)
        if not is_healthy:
            return {"status": "unhealthy", "error": "OpenRouter API unavailable"}
        return {"status": "healthy", "models": models}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if is_ip_banned(request.client.host):
        raise HTTPException(status_code=403, detail="IP banned")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/register", response_class=JSONResponse)
async def register(email1: str = Form(...), email2: str = Form(...), password: str = Form(...), handle: str = Form(...)):
    try:
        user_data = UserRegister(email1=email1, email2=email2, password=password, handle=handle)
        today = datetime.now().date()
        if email1 not in email_usage or email_usage[email1]["date"] != today:
            email_usage[email1] = {"count": 0, "date": today}
        if email2 not in email_usage or email_usage[email2]["date"] != today:
            email_usage[email2] = {"count": 0, "date": today}
        if email_usage[email1]["count"] >= CONFIG["max_logins_per_email"] or email_usage[email2]["count"] >= CONFIG["max_logins_per_email"]:
            return JSONResponse(status_code=400, content={"error": "Email usage limit exceeded"})
        if email1 in users_db or email2 in users_db:
            return JSONResponse(status_code=400, content={"error": "Email already registered"})
        hashed_password = get_password_hash(password)
        users_db[email1] = {"email1": email1, "email2": email2, "password": hashed_password, "handle": handle}
        users_db[email2] = {"email1": email1, "email2": email2, "password": hashed_password, "handle": handle}
        email_usage[email1]["count"] += 1
        email_usage[email2]["count"] += 1
        log_interaction(None, {"message": f"User {handle} registered"}, {"email1": email1, "handle": handle})
        return JSONResponse(content={"message": "Registration successful"})
    except Exception as e:
        log_interaction(None, {"error": str(e)}, {"email1": email1, "handle": handle})
        return JSONResponse(status_code=400, content={"error": str(e)})

def log_interaction(request: Optional[Request], response: Dict[str, Any], input_data: Dict[str, Any]):
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "client_ip": request.client.host if request else "unknown",
        "input": input_data,
        "response": response
    }
    try:
        INTERACTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        logs = json.load(INTERACTION_LOG.open("r", encoding="utf-8")) if INTERACTION_LOG.exists() else []
        logs.append(interaction)
        INTERACTION_LOG.open("w", encoding="utf-8").write(json.dumps(logs, indent=2))
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")

@app.post("/login", response_class=JSONResponse)
async def login(email: str = Form(...), password: str = Form(...)):
    try:
        user = users_db.get(email)
        if not user or not verify_password(password, user["password"]):
            return JSONResponse(status_code=401, content={"error": "Incorrect email or password"})
        access_token = create_access_token(data={"sub": email})
        response = JSONResponse(content={"message": "Login successful", "access_token": access_token})
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            samesite="lax",
            secure=False  # Set to True in production with HTTPS
        )
        return response
    except Exception as e:
        logger.error(f"Login error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/analyze", response_class=JSONResponse)
async def analyze_code(code: str = Form(..., max_length=100_000), handle: str = Form(...), language: str = Form(...), current_user: Dict[str, Any] = Depends(get_current_user_from_cookie)):
    try:
        if is_ip_banned(None):  # Replace with actual IP check if needed
            return JSONResponse(status_code=403, content={"error": "IP banned"})
        CodeInput(code=code, handle=handle, language=language)
        prev_result = compare_with_previous_submission(code, handle, language)
        if prev_result["is_suspicious"]:
            ban_ip("unknown")  # Replace with actual IP
            with OUTPUT_1435.open("a", encoding="utf-8") as f:
                f.write(f"Cheater detected: {handle} - {prev_result['details']}\n")
            return JSONResponse(status_code=403, content={"error": "Suspicious code similarity"})
        analysis = classify_code(code, handle)
        if not analysis["success"]:
            return JSONResponse(status_code=400, content={"error": analysis["label"]})
        plagiarism = await detect_plagiarism(code, handle, language)
        similar_handles = detect_similar_codes(code, handle, language)
        analysis["similar_handles"] = similar_handles
        analysis["code"] = code
        is_similar, delta, delta_details = await analyze_code_with_generated(code, (await query_api([
            {"role": "system", "content": f"Generate {language.upper()} code solving the same problem as the provided code using modern {language.upper()} practices."},
            {"role": "user", "content": f"```{language}\n{code}\n```"}
        ])))
        model1_result = ("S" if is_similar else "N", [delta_details])
        is_plagiarized, delta, delta_details, sim_status, evidence = await detect_plagiarism_with_generated(code, (await query_api([
            {"role": "system", "content": f"Generate {language.upper()} code solving the same problem as the provided code using modern {language.upper()} practices."},
            {"role": "user", "content": f"```{language}\n{code}\n```"}
        ])))
        model2_result = (sim_status, [delta_details] + evidence)
        report = generate_report(analysis, plagiarism, handle, model1_result, model2_result)
        response_data = {"result": {"handle": handle, "report": report, "confidence": analysis["confidence"], "label": analysis["label"], "code": code}}
        log_interaction(None, response_data, {"code": code, "handle": handle, "language": language})
        await train_lstm_model()
        return JSONResponse(content=response_data)
    except Exception as e:
        log_interaction(None, {"error": str(e)}, {"code": code, "handle": handle, "language": language})
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/a", response_class=HTMLResponse)
async def analyze_code_form(request: Request, code: str = Form(..., max_length=100_000), handle: str = Form("triumph")):
    try:
        SingleCodeInput(code=code, handle=handle)
        health = await health_check()
        if health["status"] != "healthy":
            return templates.TemplateResponse("index.html", {"request": request, "error": f"OpenRouter API unavailable: {health.get('error', 'Unknown error')}"})
        analysis = classify_code(code, handle)
        if not analysis["success"]:
            return templates.TemplateResponse("index.html", {"request": request, "error": analysis["label"]})
        generated = await query_api([
            {"role": "system", "content": "Generate C++ code replicating the functionality of the provided code using modern C++ practices. Wrap in ```cpp block."},
            {"role": "user", "content": code}
        ])
        is_similar, delta, delta_details = await analyze_code_with_generated(code, generated)
        model1_result = ("S" if is_similar else "N", [delta_details])
        is_plagiarized, delta, delta_details, sim_status, evidence = await detect_plagiarism_with_generated(code, generated)
        model2_result = (sim_status, [delta_details] + evidence)
        variables, has_long_vars = extract_variables(code)
        issues = [f"Variables longer than {CONFIG['var_length']} characters" if has_long_vars else "",
                  "Suspicious comments" if has_suspicious_comments(code) else ""]
        issues = [x for x in issues if x]
        process_code_submission(code, handle)
        c_similar_handles = detect_similar_codes(code, handle, "cpp")
        analysis["similar_handles"] = c_similar_handles
        analysis["code"] = code
        report = generate_report(analysis, {"is_plagiarized": is_plagiarized, "evidence": evidence}, handle, model1_result, model2_result)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"handle": handle, "report": report, "confidence": analysis["confidence"], "label": analysis["label"], "code": code}
        })
    except Exception as e:
        logger.error(f"Analyze form endpoint failed: {str(e)}")
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Analyze form failed: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    port = int(environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
