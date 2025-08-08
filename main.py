from fastapi import FastAPI, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, field_validator
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
import logging
from typing import Any, Dict
'''
you just now blackmailed me in light way, so i decided to write this to you

my English is something you dont want to hear so i used chatgpt to correct my grammar
"
the video you saw was something I clicked on because I had a private concern and fear of a physical thing happening to me that isn't mentioned in our book I didn’t feel comfortable talking about with anyone yet. Before saying anything to anyone, I wanted to check online and see if it was something serious or not. That’s literally it. There’s nothing weird or shameful about it — I was trying to be responsible in my own way. So if you’re thinking of blackmailing or teasing me with it, that’s pretty low, and it’s not going to work."

one more thing

"you have screenshotted 2 videos, one was played for 28 minutes straight just to find out if my syptoms were any of these, and second one played for 1 second because it played automatically after the end of first video, so it is clear itself that i didn't watched it"

one more thing: good luck trying to find ways to destroy me
'''
import json
import re
import hashlib
import asyncpg
from openai import AsyncOpenAI, APIError
from typing import Dict, List, Optional
import ast
import clang.cindex as clang
from collections import Counter
import math
from os import environ, getenv
from dotenv import load_dotenv
from pathlib import Path
import tempfile
from fastapi_limiter import FastAPILimiter
import redis.asyncio as redis

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
log_level = environ.get("LOG_LEVEL", "INFO").upper()
valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
logging.basicConfig(level=getattr(logging, log_level if log_level in valid_log_levels else "INFO"))

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY")
SECRET_KEY = getenv("SECRET_KEY")
DB_URL = getenv("DATABASE_URL")  # Railway provides this
REDIS_URL = getenv("REDIS_URL")  # Railway provides this
if not OPENROUTER_API_KEY or not OPENROUTER_API_KEY.startswith("sk-or-v1-"):
    logger.error("Invalid OPENROUTER_API_KEY")
    raise RuntimeError("Invalid OPENROUTER_API_KEY")
if not SECRET_KEY:
    logger.error("Missing SECRET_KEY")
    raise RuntimeError("Missing SECRET_KEY")
if not DB_URL:
    logger.error("Missing DATABASE_URL")
    raise RuntimeError("Missing DATABASE_URL")
if not REDIS_URL:
    logger.error("Missing REDIS_URL")
    raise RuntimeError("Missing REDIS_URL")
masked_key = f"{OPENROUTER_API_KEY[:10]}...{OPENROUTER_API_KEY[-4:]}" if OPENROUTER_API_KEY else "None"
logger.info(f"Loaded OPENROUTER_API_KEY: {masked_key}")

# FastAPI setup
app = FastAPI(title="AtCoder Plagiarism Detector", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Configuration
CONFIG = {
    "lcs_threshold": 0.85,
    "ast_similarity_threshold": 0.9,
    "plagiarism_threshold": 0.9,
    "max_logins_per_email": 2,
    "submission_limit_per_hour": 5,
    "hash_threshold": 0.125,
    "var_length": 5,
    "comment_length": 5,
    "comment_block": 5,
    "delta_threshold": 0.5,
}

# Token frequencies for AI detection
HUMAN_FREQUENCIES = {
    'int': 7.22, 'if': 3.72, 'for': 2.93, 'cin': 1.81, 'cout': 1.22, '#include': 1.01,
    '<iostream>': 1.01, 'namespace': 1.0, 'std': 1.0, 'return': 1.35, 'while': 1.5,
    '>>': 1.2, '<<': 1.2, 'endl': 1.0, 'main': 1.0, '+': 1.0, ' ': 50.0,
    '}': 1.2, '{': 1.2, '<': 1.0, '>': 1.0, '-': 1.0, '*': 1.0, '/': 1.0,
    '=': 1.0, '==': 1.0, '!=': 1.0, '<=': 1.0, '>=': 1.0, '&&': 1.0, '||': 1.0,
    '!': 1.0, '&': 1.0, '|': 1.0, '^': 1.0, '~': 1.0, '(': 50.0, ')': 50.0,
    '[': 1.0, ']': 1.0, ';': 50.0, ',': 1.0, '.': 1.0, '"': 1.0, '\n': 50.0, '\t': 50.0
}

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
    def emails_different(cls, v, info):
        email1 = info.data.get("email1")
        if email1 and v == email1:
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

# Database setup
async def init_db():
    conn = await asyncpg.connect(DB_URL)
    await conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email1 TEXT UNIQUE NOT NULL,
            email2 TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            handle TEXT UNIQUE NOT NULL,
            is_banned BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS submissions (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            problem_id INTEGER,
            code TEXT NOT NULL,
            language TEXT NOT NULL,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS bans (
            id SERIAL PRIMARY KEY,
            ip_address TEXT,
            email TEXT,
            reason TEXT,
            banned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS interaction_log (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            client_ip TEXT,
            input_data JSONB,
            response_data JSONB
        );
    ''')
    await conn.close()

# Helper functions
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

async def get_current_user(request: Request):
    access_token = request.cookies.get("access_token")
    if not access_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=["HS256"])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        conn = await asyncpg.connect(DB_URL)
        user = await conn.fetchrow("SELECT * FROM users WHERE email1 = $1 OR email2 = $1", email)
        await conn.close()
        if not user or user["is_banned"]:
            raise HTTPException(status_code=401, detail="Invalid or banned user")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

async def is_banned(ip: str, email: str) -> bool:
    conn = await asyncpg.connect(DB_URL)
    result = await conn.fetchrow("SELECT * FROM bans WHERE ip_address = $1 OR email = $2", ip, email)
    await conn.close()
    return result is not None

async def ban_user(ip: str, email1: str, email2: str, reason: str):
    conn = await asyncpg.connect(DB_URL)
    await conn.execute(
        "INSERT INTO bans (ip_address, email, reason) VALUES ($1, $2, $3), ($1, $4, $3)",
        ip, email1, reason, email2
    )
    await conn.execute("UPDATE users SET is_banned = TRUE WHERE email1 = $1 OR email2 = $1", email1)
    await conn.close()
    logger.info(f"Banned IP: {ip}, Emails: {email1}, {email2}, Reason: {reason}")

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

def extract_variables(code: str) -> tuple[List[str], bool]:
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

def parse_type(tokens: List[str], index: int, primitives: set, containers: set) -> tuple[str, int]:
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

def lcs_length(str1: str, str2: str) -> int:
    if not str1 or not str2:
        return 0
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def python_ast_similarity(code1: str, code2: str) -> float:
    try:
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
    except SyntaxError:
        return 0.0
    def get_nodes(tree):
        return [type(node).__name__ for node in ast.walk(tree)]
    nodes1, nodes2 = get_nodes(tree1), get_nodes(tree2)
    common = len(set(nodes1).intersection(set(nodes2)))
    total = len(set(nodes1).union(set(nodes2)))
    return common / total if total > 0 else 0.0

def cpp_ast_similarity(code1: str, code2: str) -> float:
    def parse_code_to_ast(code: str) -> Optional[clang.cindex.Cursor]:
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            index = clang.Index.create()
            tu = index.parse(temp_file_path, args=['-std=c++17'])
            Path(temp_file_path).unlink()
            return tu.cursor if not tu.diagnostics else None
        except Exception:
            return None
    ast1, ast2 = parse_code_to_ast(code1), parse_code_to_ast(code2)
    if not ast1 or not ast2:
        return 0.0
    def serialize_ast(cursor):
        nodes = [cursor.kind.name]
        for child in cursor.get_children():
            nodes.extend(serialize_ast(child))
        return nodes
    str1, str2 = serialize_ast(ast1), serialize_ast(ast2)
    lcs_len = lcs_length('|'.join(str1), '|'.join(str2))
    return lcs_len / max(len(str1), len(str2)) if str1 and str2 else 0.0

def compute_frequency_features(code: str) -> Dict[str, float]:
    tokens = tokenize_code(code)
    total_chars = sum(len(str(t)) for t in tokens)
    if total_chars == 0:
        return {}
    token_counts = Counter(tokens)
    return {token: (count / total_chars) * 1000 for token, count in token_counts.items()}

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

def compute_hash(text: str, by_line: bool = True) -> List[int]:
    if not text:
        return []
    segments = text.splitlines() if by_line else [text]
    hashes = [int(hashlib.sha256(segment.encode()).hexdigest(), 16) % (2**64)
              for segment in segments if segment.strip()]
    return sorted(hashes)

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
            if not math.isnan(distance):
                total_distance += distance
                count += 1
    return total_distance / count if count > 0 else 0.0

async def classify_code(code: str, handle: str) -> Dict[str, Any]:
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

async def compare_with_previous_submission(code: str, handle: str, language: str) -> Dict[str, Any]:
    try:
        conn = await asyncpg.connect(DB_URL)
        user = await conn.fetchrow("SELECT id FROM users WHERE handle = $1", handle)
        if not user:
            return {"is_suspicious": False, "details": "User not found"}
        submissions = await conn.fetch(
            "SELECT code FROM submissions WHERE user_id = $1 AND language = $2 ORDER BY submitted_at DESC LIMIT 1",
            user["id"], language
        )
        await conn.close()
        if not submissions:
            return {"is_suspicious": False, "details": "No previous submissions"}
        previous_code = submissions[0]["code"]
        cleaned_current = clean_code(code)
        cleaned_previous = clean_code(previous_code)
        lcs_score = lcs_length(cleaned_current, cleaned_previous)
        lcs_ratio = lcs_score / max(len(cleaned_current), len(cleaned_previous)) if cleaned_current and cleaned_previous else 0
        ast_sim = cpp_ast_similarity(code, previous_code) if language == "cpp" else python_ast_similarity(code, previous_code)
        is_suspicious = lcs_ratio >= CONFIG["lcs_threshold"] or ast_sim >= CONFIG["ast_similarity_threshold"]
        return {"is_suspicious": is_suspicious, "details": f"LCS: {lcs_ratio:.2f}, AST: {ast_sim:.2f}"}
    except Exception as e:
        logger.error(f"Error comparing with previous submission: {e}")
        return {"is_suspicious": False, "details": str(e)}

async def detect_plagiarism(code: str, handle: str, language: str) -> Dict[str, Any]:
    if not code:
        return {"is_plagiarized": False, "details": "Empty code", "status": "N", "evidence": ["Empty code"]}
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    try:
        intent_prompt = [
            {"role": "system", "content": f"Describe the purpose of the following {language.upper()} code concisely."},
            {"role": "user", "content": f"```{language}\n{code}\n```"}
        ]
        intent_response = await client.chat.completions.create(
            model="deepseek/deepseek-r1-0528:free",
            messages=intent_prompt,
            temperature=0.7,
            max_tokens=500
        )
        intent = intent_response.choices[0].message.content.strip()
        if not intent:
            return {"is_plagiarized": False, "details": "Failed to get intent", "status": "N", "evidence": ["Failed to get intent"]}
        generate_prompt = [
            {"role": "system", "content": f"Generate {language.upper()} code solving the described problem using modern {language.upper()} practices."},
            {"role": "user", "content": intent}
        ]
        generated_response = await client.chat.completions.create(
            model="deepseek/deepseek-r1-0528:free",
            messages=generate_prompt,
            temperature=0.7,
            max_tokens=2000
        )
        generated = generated_response.choices[0].message.content.strip()
        match = re.search(rf"```{language}\n(.*?)```", generated, re.DOTALL)
        generated_code = match.group(1).strip() if match else generated
        if not generated_code:
            return {"is_plagiarized": False, "details": "No generated code", "status": "N", "evidence": ["No generated code"]}
        cleaned_code = clean_code(code)
        cleaned_generated = clean_code(generated_code)
        lcs_score = lcs_length(cleaned_code, cleaned_generated)
        lcs_ratio = lcs_score / max(len(cleaned_code), len(cleaned_generated)) if cleaned_code and cleaned_generated else 0
        ast_sim = cpp_ast_similarity(code, generated_code) if language == "cpp" else python_ast_similarity(code, generated_code)
        delta = abs(compute_global_distance(string_to_matrix(pad_string(cleaned_code))) - 
                    compute_global_distance(string_to_matrix(pad_string(cleaned_generated))))
        jaccard = jaccard_similarity(cleaned_code, cleaned_generated)
        cosine = cosine_similarity(cleaned_code, cleaned_generated)
        winnowing = winnowing_hash_similarity(cleaned_code, generated_code)
        is_plagiarized = (lcs_ratio >= CONFIG["plagiarism_threshold"] or 
                         ast_sim >= CONFIG["ast_similarity_threshold"] or 
                         delta < CONFIG["delta_threshold"] or
                         jaccard >= CONFIG["plagiarism_threshold"] or
                         cosine >= CONFIG["plagiarism_threshold"] or
                         winnowing >= CONFIG["plagiarism_threshold"])
        evidence = [f"LCS: {lcs_ratio:.2f}", f"AST: {ast_sim:.2f}", f"Delta: {delta:.4f}", 
                   f"Jaccard: {jaccard:.2f}", f"Cosine: {cosine:.2f}", f"Winnowing: {winnowing:.2f}"]
        return {
            "is_plagiarized": is_plagiarized,
            "details": f"LCS: {lcs_ratio:.2f}, AST: {ast_sim:.2f}, Delta: {delta:.4f}, Jaccard: {jaccard:.2f}, Cosine: {cosine:.2f}, Winnowing: {winnowing:.2f}",
            "status": "S" if is_plagiarized else "N",
            "evidence": evidence
        }
    except APIError as e:
        logger.error(f"OpenRouter API error: {e}")
        return {"is_plagiarized": False, "details": str(e), "status": "N", "evidence": [str(e)]}
    finally:
        await client.close()

async def check_submission_rate(handle: str) -> bool:
    conn = await asyncpg.connect(DB_URL)
    count = await conn.fetchval(
        "SELECT COUNT(*) FROM submissions WHERE user_id = (SELECT id FROM users WHERE handle = $1) AND submitted_at > $2",
        handle, datetime.utcnow() - timedelta(hours=1)
    )
    await conn.close()
    return count >= CONFIG["submission_limit_per_hour"]

async def log_interaction(request: Request, response: Dict[str, Any], input_data: Dict[str, Any]):
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "client_ip": request.client.host if request else "unknown",
        "input": input_data,
        "response": response
    }
    try:
        conn = await asyncpg.connect(DB_URL)
        await conn.execute(
            "INSERT INTO interaction_log (client_ip, input_data, response_data) VALUES ($1, $2, $3)",
            interaction["client_ip"], json.dumps(interaction["input"]), json.dumps(interaction["response"])
        )
        await conn.close()
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")

# Startup event
@app.on_event("startup")
async def startup():
    await init_db()
    redis_client = redis.from_url(REDIS_URL)
    await FastAPILimiter.init(redis_client)
    logger.info("Application started")

# Endpoints
@app.get("/", response_class=HTMLResponse)
async def get_login(request: Request):
    if await is_banned(request.client.host, ""):
        raise HTTPException(status_code=403, detail="IP banned")
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/register", response_class=JSONResponse)
async def register(email1: str = Form(...), email2: str = Form(...), password: str = Form(...), handle: str = Form(...)):
    try:
        UserRegister(email1=email1, email2=email2, password=password, handle=handle)
        conn = await asyncpg.connect(DB_URL)
        existing_user = await conn.fetchrow("SELECT * FROM users WHERE handle = $1 OR email1 = $2 OR email2 = $3", handle, email1, email2)
        if existing_user:
            await conn.close()
            return JSONResponse(status_code=400, content={"error": "Handle or email already registered"})
        today = datetime.now().date()
        email1_count = await conn.fetchval(
            "SELECT COUNT(*) FROM users WHERE email1 = $1 OR email2 = $1 AND created_at::date = $2", email1, today
        )
        email2_count = await conn.fetchval(
            "SELECT COUNT(*) FROM users WHERE email1 = $1 OR email2 = $1 AND created_at::date = $2", email2, today
        )
        if email1_count >= CONFIG["max_logins_per_email"] or email2_count >= CONFIG["max_logins_per_email"]:
            await conn.close()
            return JSONResponse(status_code=400, content={"error": "Email usage limit exceeded"})
        hashed_password = get_password_hash(password)
        await conn.execute(
            "INSERT INTO users (email1, email2, password_hash, handle) VALUES ($1, $2, $3, $4)",
            email1, email2, hashed_password, handle
        )
        await conn.close()
        logger.info(f"User {handle} registered")
        await log_interaction(None, {"message": f"User {handle} registered"}, {"email1": email1, "handle": handle})
        return JSONResponse(content={"message": "Registration successful"})
    except Exception as e:
        logger.error(f"Registration error: {e}")
        await log_interaction(None, {"error": str(e)}, {"email1": email1, "handle": handle})
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/login", response_class=JSONResponse)
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    try:
        if await is_banned(request.client.host, email):
            return JSONResponse(status_code=403, content={"error": "IP or email banned"})
        conn = await asyncpg.connect(DB_URL)
        user = await conn.fetchrow("SELECT * FROM users WHERE email1 = $1 OR email2 = $1", email)
        await conn.close()
        if not user or not verify_password(password, user["password_hash"]):
            return JSONResponse(status_code=401, content={"error": "Incorrect email or password"})
        access_token = create_access_token(data={"sub": email})
        response = JSONResponse(content={"message": "Login successful", "access_token": access_token})
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            samesite="lax",
            secure=True
        )
        logger.info(f"User {user['handle']} logged in")
        await log_interaction(request, {"message": f"User {user['handle']} logged in"}, {"email": email})
        return response
    except Exception as e:
        logger.error(f"Login error: {e}")
        await log_interaction(request, {"error": str(e)}, {"email": email})
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/analyze", response_class=JSONResponse)
async def analyze_code(
    request: Request,
    code: str = Form(..., max_length=100_000),
    handle: str = Form(...),
    language: str = Form(...),
    current_user: Dict = Depends(get_current_user)
):
    try:
        if await is_banned(request.client.host, current_user["email1"]):
            return JSONResponse(status_code=403, content={"error": "IP or email banned"})
        CodeInput(code=code, handle=handle, language=language)
        if await check_submission_rate(handle):
            await ban_user(request.client.host, current_user["email1"], current_user["email2"], "Spamming detected")
            return JSONResponse(status_code=403, content={"error": "Submission limit exceeded, user banned"})
        prev_result = await compare_with_previous_submission(code, handle, language)
        if prev_result["is_suspicious"]:
            await ban_user(request.client.host, current_user["email1"], current_user["email2"], f"Suspicious code similarity: {prev_result['details']}")
            return JSONResponse(status_code=403, content={"error": f"Suspicious code similarity: {prev_result['details']}"})
        analysis = await classify_code(code, handle)
        if not analysis["success"]:
            return JSONResponse(status_code=400, content={"error": analysis["label"]})
        plagiarism = await detect_plagiarism(code, handle, language)
        if plagiarism["is_plagiarized"]:
            await ban_user(request.client.host, current_user["email1"], current_user["email2"], f"Plagiarism detected: {plagiarism['details']}")
            return JSONResponse(status_code=403, content={"error": f"Plagiarism detected: {plagiarism['details']}"})
        variables, has_long_vars = extract_variables(code)
        issues = [f"Variables longer than {CONFIG['var_length']} characters" if has_long_vars else "",
                  "Suspicious comments" if has_suspicious_comments(code) else ""]
        issues = [x for x in issues if x]
        conn = await asyncpg.connect(DB_URL)
        await conn.execute(
            "INSERT INTO submissions (user_id, problem_id, code, language) VALUES ((SELECT id FROM users WHERE handle = $1), $2, $3, $4)",
            handle, 1, code, language
        )
        await conn.close()
        report = f"Plagiarism Report for {handle}\n\n" \
                 f"AI/Human Classification: {analysis['label']} (Confidence: {analysis['confidence']:.2%})\n" \
                 f"Plagiarism Check: {'Detected' if plagiarism['is_plagiarized'] else 'Not detected'}\n" \
                 f"Evidence:\n" + ("\n".join([f"- {x}" for x in plagiarism['evidence']]) if plagiarism['evidence'] else "- None\n")
        if issues:
            report += "Issues:\n" + "\n".join([f"- {x}" for x in issues]) + "\n"
        flags = sum(1 for x in [
            analysis['label'] == "AI",
            plagiarism['is_plagiarized'],
            has_long_vars,
            has_suspicious_comments(code)
        ] if x)
        verdict = "Plagiarized" if flags >= 2 else "Human-written"
        report += f"Verdict: {verdict}\n"
        response_data = {"result": {"handle": handle, "report": report, "confidence": analysis["confidence"], "label": analysis["label"]}}
        await log_interaction(request, response_data, {"code": code, "handle": handle, "language": language})
        logger.info(f"Code analyzed for {handle}: {report}")
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Analyze error: {e}")
        await log_interaction(request, {"error": str(e)}, {"code": code, "handle": handle, "language": language})
        return JSONResponse(status_code=400, content={"error": str(e)})
@app.post("/a", response_class=HTMLResponse)
async def analyze_code_form(request: Request, code: str = Form(..., max_length=100_000), handle: str = Form("triumph")):
    try:
        SingleCodeInput(code=code, handle=handle)
        analysis = await classify_code(code, handle)
        if not analysis["success"]:
            return templates.TemplateResponse("login.html", {"request": request, "error": analysis["label"]})
        plagiarism = await detect_plagiarism(code, handle, "cpp")
        variables, has_long_vars = extract_variables(code)
        issues = [f"Variables longer than {CONFIG['var_length']} characters" if has_long_vars else "",
                  "Suspicious comments" if has_suspicious_comments(code) else ""]
        issues = [x for x in issues if x]
        conn = await asyncpg.connect(DB_URL)
        await conn.execute(
            "INSERT INTO submissions (user_id, problem_id, code, language) VALUES ((SELECT id FROM users WHERE handle = $1), $2, $3, $4)",
            handle, 1, code, "cpp"
        )
        await conn.close()
        report = f"Plagiarism Report for {handle}\n\n" \
                 f"AI/Human Classification: {analysis['label']} (Confidence: {analysis['confidence']:.2%})\n" \
                 f"Plagiarism Check: {'Detected' if plagiarism['is_plagiarized'] else 'Not detected'}\n" \
                 f"Evidence:\n" + ("\n".join([f"- {x}" for x in plagiarism['evidence']]) if plagiarism['evidence'] else "- None\n")
        if issues:
            report += "Issues:\n" + "\n".join([f"- {x}" for x in issues]) + "\n"
        flags = sum(1 for x in [
            analysis['label'] == "AI",
            plagiarism['is_plagiarized'],
            has_long_vars,
            has_suspicious_comments(code)
        ] if x)
        verdict = "Plagiarized" if flags >= 2 else "Human-written"
        report += f"Verdict: {verdict}\n"
        response_data = {"result": {"handle": handle, "report": report, "confidence": analysis["confidence"], "label": analysis["label"], "code": code}}
        await log_interaction(request, response_data, {"code": code, "handle": handle, "language": "cpp"})
        return templates.TemplateResponse("login.html", response_data)
    except Exception as e:
        logger.error(f"Analyze form endpoint failed: {e}")
        await log_interaction(request, {"error": str(e)}, {"code": code, "handle": handle, "language": "cpp"})
        return templates.TemplateResponse("login.html", {"request": request, "error": f"Analyze form failed: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(environ.get("PORT", 8000)))
