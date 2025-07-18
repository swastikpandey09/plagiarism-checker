from __future__ import annotations
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from fastapi import FastAPI, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, validator
from py7zr import SevenZipFile
import tempfile
import shutil
from os import environ, getenv
from openai import OpenAI, APIError
import requests
from datetime import datetime, timedelta
from collections import Counter
import math
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import clang.cindex as clang

# Logging setup with JSON formatter for better Railway debugging
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
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = getenv("OPENROUTER_API_KEY")
SECRET_KEY = getenv("SECRET_KEY")
if not OPENROUTER_API_KEY or not SECRET_KEY:
    logger.error("Missing required environment variables: OPENROUTER_API_KEY or SECRET_KEY")
    raise RuntimeError("OPENROUTER_API_KEY or SECRET_KEY is missing")

# FastAPI app setup
app = FastAPI(title="Code Plagiarism Detector", version="0.2.2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Security setup
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Configuration
BASE_DIR = Path(__file__).resolve().parent
INPUT_1568 = BASE_DIR / "input1568.txt"
INPUT_1435 = BASE_DIR / "input1435.txt"
OUTPUT_1435 = BASE_DIR / "output1435.txt"
INTERACTION_LOG = BASE_DIR / "interaction_log.json"
BANNED_IPS = BASE_DIR / "banned_ips.txt"
LSTM_MODEL_PATH = BASE_DIR / "lstm_plagiarism_model.h5"
CONFIG = {
    "lcs_threshold": 0.5,
    "hash_threshold": 0.125,
    "ast_similarity_threshold": 0.85,
    "plagiarism_threshold": 0.10,
    "similarity_threshold": 0.9,
    "max_logins_per_email": 2,
}

# Token frequencies
HUMAN_FREQUENCIES = {
    'int': 7.22, 'if': 3.72, 'for': 2.93, 'cin': 1.81, 'cout': 1.22, '#include': 1.01,
    '<iostream>': 1.01, 'namespace': 1.0, 'std': 1.0, 'return': 1.35, 'while': 1.5,
    '>>': 1.2, '<<': 1.2, 'endl': 1.0, 'main': 1.0, '+': 1.0, ' ': 50.0,
}

# OpenAI client
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# Pydantic models
class UserRegister(BaseModel):
    email1: EmailStr
    email2: EmailStr
    password: str
    handle: str
    @validator("handle")
    def validate_handle(cls, v):
        if not v.strip() or any(c in v for c in "\n$"):
            raise ValueError("Invalid handle")
        return v
    @validator("email2")
    def emails_different(cls, v, values):
        if "email1" in values and v == values["email1"]:
            raise ValueError("Emails must be different")
        return v

class CodeInput(BaseModel):
    code: str
    handle: str
    @validator("handle")
    def validate_handle(cls, v):
        if not v.strip() or any(c in v for c in "\n$"):
            raise ValueError("Invalid handle")
        return v

class Token(BaseModel):
    access_token: str
    token_type: str

# Global state
users_db: Dict[str, Dict[str, Any]] = {}
email_usage: Dict[str, int] = {}
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

def log_interaction(request: Request, response: Dict[str, Any], input_data: Dict[str, Any]):
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "client_ip": request.client.host,
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
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return users_db[email]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")

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

def clean_code(code: str) -> str:
    if not code:
        return ""
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return re.sub(r'\s+', ' ', code).strip()

def tokenize_code(code: str) -> List[str]:
    if not code:
        return []
    tokens = re.findall(r'\w+|[^\w\s]', code, re.UNICODE)
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

def lcs_length(str1: str, str2: str) -> int:
    if not str1 or not str2:
        return 0
    n, m = len(str1), len(str2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if str1[i-1] == str2[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[n][m]

def compare_codes(code1: str, code2: str) -> Dict[str, Any]:
    if not code1 or not code2:
        return {"is_similar": False, "details": "Empty code"}
    cleaned1, cleaned2 = clean_code(code1), clean_code(code2)
    lcs_score = lcs_length(cleaned1, cleaned2)
    lcs_ratio = lcs_score / max(len(cleaned1), cleaned2) if cleaned1 and cleaned2 else 0
    ast1 = parse_code_to_ast(code1)
    ast2 = parse_code_to_ast(code2)
    ast_similarity = compare_ast(serialize_ast(ast1), serialize_ast(ast2)) if ast1 and ast2 else 0.0
    is_similar = lcs_ratio >= CONFIG["lcs_threshold"] or ast_similarity >= CONFIG["ast_similarity_threshold"]
    return {"is_similar": is_similar, "details": f"LCS: {lcs_ratio:.2f}, AST: {ast_similarity:.2f}"}

def compare_with_previous_submission(code: str, handle: str) -> Dict[str, Any]:
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
        ast1 = parse_code_to_ast(code)
        ast2 = parse_code_to_ast(previous_code)
        ast_similarity = compare_ast(serialize_ast(ast1), serialize_ast(ast2)) if ast1 and ast2 else 0.0
        is_suspicious = lcs_ratio >= CONFIG["similarity_threshold"] or ast_similarity >= CONFIG["ast_similarity_threshold"]
        return {"is_suspicious": is_suspicious, "details": f"LCS: {lcs_ratio:.2f}, AST: {ast_similarity:.2f}"}
    except Exception as e:
        logger.error(f"Error comparing with previous submission: {e}")
        return {"is_suspicious": False, "details": str(e)}

async def detect_plagiarism(code: str, handle: str) -> Dict[str, Any]:
    if not code:
        return {"is_plagiarized": False, "delta": 0.0, "details": "Empty code", "status": "N", "evidence": ["Empty code"]}
    cleaned_code = clean_code(code)
    intent_prompt = [
        {"role": "system", "content": "Describe the purpose of the following C++ code concisely."},
        {"role": "user", "content": f"```cpp\n{code}\n```"}
    ]
    intent = await query_api(intent_prompt)
    if not intent:
        return {"is_plagiarized": False, "delta": 0.0, "details": "Failed to get intent", "status": "N", "evidence": ["Failed to get intent"]}
    generate_prompt = [
        {"role": "system", "content": "Generate C++ code solving the described problem using modern C++ practices."},
        {"role": "user", "content": intent}
    ]
    generated_code = await query_api(generate_prompt)
    match = re.search(r"```cpp\n(.*?)```", generated_code, re.DOTALL)
    generated_code = match.group(1).strip() if match else generated_code.strip()
    if not generated_code:
        return {"is_plagiarized": False, "delta": 0.0, "details": "No generated code", "status": "N", "evidence": ["No generated code"]}
    lcs_score = lcs_length(cleaned_code, clean_code(generated_code))
    lcs_ratio = lcs_score / max(len(cleaned_code), len(clean_code(generated_code))) if cleaned_code else 0
    ast1 = parse_code_to_ast(code)
    ast2 = parse_code_to_ast(generated_code)
    ast_similarity = compare_ast(serialize_ast(ast1), serialize_ast(ast2)) if ast1 and ast2 else 0.0
    is_plagiarized = lcs_ratio >= CONFIG["lcs_threshold"] or ast_similarity >= CONFIG["ast_similarity_threshold"]
    evidence = [f"LCS: {lcs_ratio:.2f}", f"AST: {ast_similarity:.2f}"]
    return {
        "is_plagiarized": is_plagiarized,
        "delta": lcs_ratio,
        "details": f"LCS: {lcs_ratio:.2f}, AST: {ast_similarity:.2f}",
        "status": "S" if is_plagiarized else "N",
        "evidence": evidence
    }

async def query_api(messages: List[Dict[str, str]], model: str = "moonshotai/kimi-k2:free") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"```cpp\n(.*?)```", content, re.DOTALL)
        return match.group(1).strip() if match else content.strip()
    except APIError as e:
        logger.error(f"OpenRouter API error: {e}")
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
    except Exception as e:
        logger.error(f"Failed to process submission: {e}")
        raise

def detect_similar_codes(code: str, handle: str) -> List[str]:
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
                result = compare_codes(cleaned_code, other_code)
                if result["is_similar"]:
                    similar_handles.append(f"{other_handle}: {result['details']}")
        OUTPUT_1435.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_1435.open("w", encoding="utf-8") as f:
            f.write(f"{len(similar_handles)} similar pairs detected\n\n")
            f.write(f"{len(similar_handles)} cheaters detected:\n")
            f.write("\n".join(similar_handles) + "\n" if similar_handles else "None\n")
        return similar_handles
    except Exception as e:
        logger.error(f"Failed to detect similar codes: {e}")
        return []

def generate_report(analysis: Dict[str, Any], plagiarism: Dict[str, Any], handle: str) -> str:
    report = f"Plagiarism Report for {handle}\n\n"
    report += f"AI/Human Classification: {analysis['label']} (Confidence: {analysis['confidence']:.2%})\n"
    report += f"Plagiarism Check: {'Detected' if plagiarism['is_plagiarized'] else 'Not detected'}\n"
    report += "Evidence:\n" + "\n".join([f"- {x}" for x in plagiarism["evidence"]]) + "\n"
    report += f"Similar Handles: {', '.join(analysis.get('similar_handles', [])) or 'None'}\n"
    flags = sum(1 for x in [
        analysis['label'] == "AI",
        plagiarism['is_plagiarized'],
        analysis.get('similar_handles', [])
    ] if x)
    verdict = "Plagiarized" if flags >= 2 else "Human-written"
    report += f"\nVerdict: {verdict}\n"
    return report

@app.on_event("startup")
async def startup_event():
    logger.info("Starting application...")
    try:
        # Verify pyjwt import
        from jose import jwt
        logger.info("pyjwt module loaded successfully")
        if LSTM_MODEL_PATH.exists():
            global lstm_model
            lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
            logger.info("LSTM model loaded")
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    try:
        shutil.rmtree(tempfile.gettempdir(), ignore_errors=True)
        logger.info("Cleaned up temporary files")
    except Exception as e:
        logger.error(f"Shutdown cleanup failed: {e}")

@app.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy"}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if is_ip_banned(request.client.host):
        raise HTTPException(status_code=403, detail="IP banned")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/register", response_class=HTMLResponse)
async def register(request: Request, email1: str = Form(...), email2: str = Form(...), password: str = Form(...), handle: str = Form(...)):
    try:
        user_data = UserRegister(email1=email1, email2=email2, password=password, handle=handle)
        if email_usage.get(email1, 0) >= CONFIG["max_logins_per_email"] or email_usage.get(email2, 0) >= CONFIG["max_logins_per_email"]:
            raise HTTPException(status_code=400, detail="Email usage limit exceeded")
        if email1 in users_db:
            raise HTTPException(status_code=400, detail="Email already registered")
        users_db[email1] = {
            "email1": email1,
            "email2": email2,
            "password": get_password_hash(password),
            "handle": handle
        }
        email_usage[email1] = email_usage.get(email1, 0) + 1
        email_usage[email2] = email_usage.get(email2, 0) + 1
        log_interaction(request, {"message": f"User {handle} registered"}, {"email1": email1, "handle": handle})
        return templates.TemplateResponse("index.html", {"request": request, "message": f"User {handle} registered successfully"})
    except Exception as e:
        log_interaction(request, {"error": str(e)}, {"email1": email1, "handle": handle})
        return templates.TemplateResponse("index.html", {"request": request, "error": str(e)})

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_code(request: Request, code: str = Form(..., max_length=100_000), handle: str = Form(...), current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        if is_ip_banned(request.client.host):
            raise HTTPException(status_code=403, detail="IP banned")
        CodeInput(code=code, handle=handle)
        prev_result = compare_with_previous_submission(code, handle)
        if prev_result["is_suspicious"]:
            ban_ip(request.client.host)
            with OUTPUT_1435.open("a", encoding="utf-8") as f:
                f.write(f"Cheater detected: {handle} - {prev_result['details']}\n")
            raise HTTPException(status_code=403, detail="Suspicious code similarity")
        analysis = classify_code(code, handle)
        if not analysis["success"]:
            raise HTTPException(status_code=400, detail=analysis["label"])
        plagiarism = await detect_plagiarism(code, handle)
        similar_handles = detect_similar_codes(code, handle)
        analysis["similar_handles"] = similar_handles
        report = generate_report(analysis, plagiarism, handle)
        process_code_submission(code, handle)
        response_data = {"result": {"handle": handle, "report": report, "confidence": analysis["confidence"], "label": analysis["label"]}}
        log_interaction(request, response_data, {"code": code, "handle": handle})
        await train_lstm_model()
        return templates.TemplateResponse("index.html", {"request": request, "result": response_data["result"]})
    except Exception as e:
        log_interaction(request, {"error": str(e)}, {"code": code, "handle": handle})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prepare", response_class=HTMLResponse)
async def prepare_submission(request: Request, code: str = Form(...), handle: str = Form(...), current_user: Dict[str, Any] = Depends(get_current_user)):
    try:
        if is_ip_banned(request.client.host):
            raise HTTPException(status_code=403, detail="IP banned")
        CodeInput(code=code, handle=handle)
        process_code_submission(code, handle)
        response_data = {"message": f"Code for {handle} prepared"}
        log_interaction(request, response_data, {"code": code, "handle": handle})
        return templates.TemplateResponse("index.html", {"request": request, "message": response_data["message"]})
    except Exception as e:
        log_interaction(request, {"error": str(e)}, {"code": code, "handle": handle})
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(environ.get("PORT", 10000)))
