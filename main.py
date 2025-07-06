from __future__ import annotations
import re, json, hashlib, logging, os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import spacy
import uvicorn
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio
from openai import AsyncOpenAI

# ──────────────── CONFIG ────────────────
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(asctime)s | %(message)s")
log = logging.getLogger(__name__)

CODE_THRESHOLDS = {
    "p": 0.5,  # Plagiarism LCS threshold
    "h": 0.125,  # Hash similarity threshold
    "v": 5,  # Variable name length threshold
    "m": 5,  # Comment length threshold
    "b": 5,  # Block size for comment checking
    "t": 0.5,  # Structural similarity threshold (model 2)
    "u": 0.10  # Structural similarity threshold (model 3)
}

INPUT_FILE = Path("input1435.txt")
INPUT_FILE2 = Path("input1568.txt")
OUTPUT_FILE = Path("output1435.txt")

# Local client for Gemma-3-12b-it
local_client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

# ──────────────── ML MODELS ────────────────
try:
    spacy_model = spacy.load("dataset/output/model-best")
except Exception:
    spacy_model = None
    log.warning("spaCy model not loaded; AI/Human classification disabled")

# ──────────────── FASTAPI SETUP ────────────────
app = FastAPI(title="Integrated Code Inspector")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ──────────────── DATA SCHEMAS ────────────────
class CodePair(BaseModel):
    code1: str
    code2: str
    handle1: str | None = "user1"
    handle2: str | None = "user2"

class SingleCode(BaseModel):
    code: str
    handle: str | None = "user1"

class BatchCompareRequest(BaseModel):
    codes: List[str]
    handles: List[str]

# ──────────────── HEURISTICS AND UTILITIES ────────────────
def pad_code(s: str) -> str:
    """Pad code lines to form a square matrix."""
    l = s.split('\n')
    m = max([len(x) for x in l]) if l else 0
    if m % 2 != 0:
        for j in range(len(l)):
            if len(l[j]) == m:
                l[j] += 'S'
                break
    m = max([len(x) for x in l]) if l else 0
    for j in range(len(l)):
        if len(l[j]) != m:
            l[j] += 'S' * (m - len(l[j]))
    if len(l) != m:
        for _ in range(m - len(l)):
            l.append('S' * m)
    return '\n'.join(l)

def map_chars_to_grid(s: str) -> List[List[int]]:
    """Map characters in code to numerical indices."""
    l = s.split('\n')
    d = {}
    n = 0
    for r in l:
        for c in r:
            if c not in d:
                d[c] = n
                n += 1
    g = [[d[c] for c in r] for r in l]
    return g

def get_grid_center(g: List[List[int]]) -> List[int]:
    """Return the center coordinates of a grid."""
    return [len(g) // 2, len(g[0]) // 2]

def move_up(p: List[int]) -> List[int]:
    return [p[0] - 1, p[1]]

def move_down(p: List[int]) -> List[int]:
    return [p[0] + 1, p[1]]

def move_left(p: List[int]) -> List[int]:
    return [p[0], p[1] - 1]

def move_right(p: List[int]) -> List[int]:
    return [p[0], p[1] + 1]

def is_valid_position(p: List[int], w: int, z: int) -> bool:
    """Check if a position is within grid bounds."""
    return 0 <= p[0] < w and 0 <= p[1] < z

def extract_subgrid(g: List[List[int]], q: List[int], z: int = 3) -> List[List[int]]:
    """Extract a subgrid around a given position."""
    h = z // 2
    b = []
    for i in range(q[0] - h, q[0] + h + 1):
        w = []
        for j in range(q[1] - h, q[1] + h + 1):
            if 0 <= i < len(g) and 0 <= j < len(g[0]):
                w.append(g[i][j])
        if w:
            b.append(w)
    return b

def compute_subgrid_distance(b: List[List[int]]) -> float:
    """Compute average Euclidean distance in a subgrid."""
    s = 0
    n = 0
    h = len(b) // 2
    w = len(b[0]) // 2 if b else 0
    for i in range(len(b)):
        for j in range(len(b[i])):
            s += ((i - h) ** 2 + (j - w) ** 2) ** 0.5
            n += 1
    return s / n if n else 0

def compute_structural_complexity(g: List[List[int]]) -> float:
    """Compute average structural complexity of a code grid."""
    if not g or not g[0]:
        return 0
    r, c = len(g), len(g[0])
    s = 0
    n = 0
    for i in range(r):
        for j in range(c):
            b = extract_subgrid(g, [i, j])
            s += compute_subgrid_distance(b)
            n += 1
    return s / n if n else 0

async def query_external_api(
    messages: List[Dict[str, str]],
    model: str = "gemma-3-12b-it",
    temperature: float = 0.7,
    max_tokens: int = -1,
    stream: bool = False
) -> str:
    """Query the local API and extract only the code block from the response."""
    try:
        client_to_use = local_client
        response = await client_to_use.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens if max_tokens > 0 else None,
            stream=stream
        )
        full_response = response.choices[0].message.content
        # Extract code block using regex
        code_match = re.search(r"```cpp\n(.*?)```", full_response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()  # Return only the code
        log.warning("No code block found in API response")
        return full_response  # Fallback to full response if no code block
    except Exception as e:
        log.error(f"API query failed: {str(e)}")
        return "API unavailable"

async def check_structural_similarity(s: str, t: float = CODE_THRESHOLDS["t"]) -> Tuple[bool, float, str]:
    """Compare structural complexity with AI-generated code."""
    messages = [
        {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
        {"role": "user", "content": s}
    ]
    generated_code = await query_external_api(messages)
    if not generated_code or "API unavailable" in generated_code:
        return False, 0.0, "API unavailable"
    orig_score = compute_structural_complexity(map_chars_to_grid(pad_code(s)))
    gen_score = compute_structural_complexity(map_chars_to_grid(pad_code(generated_code)))
    delta = abs(orig_score - gen_score)
    return delta < t, delta, f"Delta: {delta:.4f}"

async def check_combined_similarity(s: str, t: float = CODE_THRESHOLDS["u"]) -> Tuple[bool, float, str, str, List[str]]:
    """Check combined similarity including structural, plagiarism, and LCS checks."""
    messages = [
        {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
        {"role": "user", "content": s}
    ]
    generated_code = await query_external_api(messages)
    if not generated_code or "API unavailable" in generated_code:
        return False, 0.0, "API unavailable", "N", ["API unavailable"]
    orig_score = compute_structural_complexity(map_chars_to_grid(pad_code(s)))
    gen_score = compute_structural_complexity(map_chars_to_grid(pad_code(generated_code)))
    delta = abs(orig_score - gen_score)
    plag_label, plag_evidence = model1_plagiarism(s, generated_code, "U", "O")
    lcs_similar, lcs_message = lcs_judge(s, generated_code)
    evidence = [f"Delta: {delta:.4f}", lcs_message] + plag_evidence
    is_similar = delta < t or plag_label == "S" or lcs_similar
    return is_similar, delta, f"Delta: {delta:.4f}", plag_label, evidence

def hash_lines(s: str) -> List[int]:
    """Generate sorted hash values for code lines."""
    l = s.splitlines()
    r = []
    for i, n in enumerate(l):
        if not n.strip():
            continue
        a = sum(ord(c) for c in n)
        b = sum(ord(c) for c in n if c.isalpha())
        d = sum(ord(c) for c in n if not c.isalpha())
        e = sum(ord(c) for c in n if ord(c) % 7 == 0)
        f = max([ord(c) for c in n], default=0)
        g = min([ord(c) for c in n], default=1000)
        v = (a * 132 + b * 232 + d * 67 + e * 135 + (len(l) - i) * 559 + f * 7 + g * 11 +
             a // 23 + b // 45 + d // 10 + e // 2 + f // 7 + g // 7 + a % 232)
        r.append(v)
    return sorted(r)

def lcs_count(a: List[int], b: List[int]) -> int:
    """Compute LCS count between two sorted lists."""
    i = j = s = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            s += 1
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return s

def lcs_judge(s: str, o: str) -> Tuple[bool, str]:
    """Judge similarity based on LCS of hashed lines."""
    a = hash_lines(s)
    b = hash_lines(o)
    v = lcs_count(a, b)
    m = min(len(a), len(b))
    sim = v / m if m else 0
    return sim < CODE_THRESHOLDS["p"], f"LCS: {sim:.2f}"

def tokenize(l: str) -> List[str]:
    """Tokenize a line of code."""
    t = []
    s = ""
    for c in l:
        if c.isspace() or c in ",;=(){}[].<>*&":
            if s:
                t.append(s)
                s = ""
            if not c.isspace():
                t.append(c)
        else:
            s += c
    if s:
        t.append(s)
    return t

def parse_type(t: List[str], i: int) -> Tuple[str, int]:
    """Parse C++ type declarations."""
    b = {"int", "long", "short", "float", "double", "char", "bool", "void", "auto", "unsigned", "signed", "size_t"}
    m = {"vector", "stack", "queue", "deque", "map", "set", "pair", "string"}
    s = ""
    while i < len(t) and (t[i] in b or t[i] in m):
        if s:
            s += " "
        s += t[i]
        i += 1
        if i < len(t) and t[i] == "<":
            d = 0
            s += t[i]
            i += 1
            while i < len(t):
                if t[i] == "<":
                    d += 1
                elif t[i] == ">":
                    if d == 0:
                        s += t[i]
                        i += 1
                        break
                    d -= 1
                s += t[i]
                i += 1
    return s, i

def variable_check(c: str) -> Tuple[List[str], bool]:
    """Check for suspicious variable names."""
    b = {"int", "long", "short", "float", "double", "char", "bool", "void", "auto", "unsigned", "signed", "size_t"}
    m = {"vector", "stack", "queue", "deque", "map", "set", "pair", "string"}
    k = {"main", "first", "second", "top", "push", "pop", "begin", "end", "size", "clear", "empty", "insert", "erase", "find", "sort", "reverse"}
    r = set()
    for l in c.splitlines():
        t = tokenize(l)
        i = 0
        while i < len(t):
            s = i
            u, i = parse_type(t, i)
            if not u:
                i = s + 1
                continue
            while i < len(t) and t[i] != ";":
                v = t[i]
                i += 1
                while v in ("*", "&", ",", "="):
                    if v == "=" and i < len(t):
                        i += 1
                    v = t[i] if i < len(t) else ""
                    i += 1
                if v and v != "(" and v not in k and v not in b and v not in m and not v.isdigit():
                    r.add(v)
            if i < len(t) and t[i] == ";":
                i += 1
    v = sorted(r)
    w = len([x for x in v if len(x) > CODE_THRESHOLDS["v"]]) >= 2
    return v, w

def suspicious_comment_block(c: str) -> bool:
    """Check for missing comments in code blocks."""
    l = c.splitlines()
    for i in range(0, len(l), CODE_THRESHOLDS["b"]):
        if not any(re.search(r'//(.{%d,})' % CODE_THRESHOLDS["m"], x) for x in l[i:i+CODE_THRESHOLDS["b"]]):
            return True
    return False

def strip_comments(c: str) -> str:
    """Strip comments and normalize whitespace."""
    c = re.sub(r'//.*', '', c)
    c = re.sub(r'/\*.*?\*/', '', c, flags=re.DOTALL)
    c = re.sub(r'\s+', ' ', c).strip()
    return c

def hash_code_segments(c: str) -> List[int]:
    """Generate hash values for code segments."""
    s = ""
    r = []
    for i, k in enumerate(c):
        if i >= 2 and c[i-2:i+1] == "xc9":
            if s:
                r.append(s)
                s = ""
        else:
            s += k
    if s:
        r.append(s)
    h = []
    for x in r:
        a = sum(ord(k) for k in x)
        b = sum(ord(k) for k in x if k.isalpha() and k.islower())
        d = sum(ord(k) for k in x if not (k.isalpha() and k.islower()))
        e = sum(ord(k) for k in x if ord(k) % 7 == 0)
        f = max([ord(k) for k in x], default=0)
        g = min([ord(k) for k in x], default=1000)
        v = (a * 132 + b * 232 + d * 67 + e * 135 + len(x) * 559 + f * 7 + g * 11 +
             a // 23 + b // 45 + d // 10 + e // 2 + f // 7 + g // 7 + a % 232)
        h.append(v)
    return sorted(h)

def lcs_similarity(s: str, o: str) -> int:
    """Compute LCS length between two strings."""
    d = [[0] * (len(o) + 1) for _ in range(len(s) + 1)]
    for i in range(1, len(s) + 1):
        for j in range(1, len(o) + 1):
            if s[i-1] == o[j-1]:
                d[i][j] = d[i-1][j-1] + 1
            else:
                d[i][j] = max(d[i-1][j], d[i][j-1])
    return d[-1][-1]

async def ai_human_classify(c: str, h: str) -> Tuple[bool, str, float, str]:
    """Classify code as AI or human using spaCy and LCS."""
    if not c.strip():
        return False, "E", 0.0, h
    code_clean = strip_comments(c)
    if spacy_model:
        d = spacy_model(code_clean)
        a, u = d.cats.get("AI", 0.0), d.cats.get("HUMAN", 0.0)
        confidence = max(a, u)
        messages = [
            {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
            {"role": "user", "content": c}
        ]
        generated_code = await query_external_api(messages)
        if not generated_code or "API unavailable" in generated_code:
            lcs_similar = False
        else:
            lcs_similar, _ = lcs_judge(code_clean, strip_comments(generated_code))
        label = "H" if u >= 0.95 or lcs_similar else "AI"
        return True, label, confidence, h
    else:
        messages = [
            {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
            {"role": "user", "content": c}
        ]
        generated_code = await query_external_api(messages)
        if not generated_code or "API unavailable" in generated_code:
            return False, "E", 0.0, h
        lcs_similar, _ = lcs_judge(code_clean, strip_comments(generated_code))
        label = "H" if lcs_similar else "AI"
        return True, label, 0.0, h

def model1_plagiarism(s: str, o: str, h: str, u: str) -> Tuple[str, List[str]]:
    """Plagiarism check using LCS and hash similarity."""
    s, o = strip_comments(s), strip_comments(o)
    if not s or not o:
        return "N", ["Empty"]
    l = lcs_similarity(s, o)
    m = min(len(s), len(o))
    r = l / m if m else 0
    a = hash_code_segments(s)
    b = hash_code_segments(o)
    n = len(set(a) & set(b))
    h_val = n / min(len(a), len(b)) if a and b else 0
    e = [f"LCS: {r:.2f}", f"Hash: {h_val:.2f}"]
    return "N" if r < 0.95 and h_val < 0.95 else "S", e

async def model2_plagiarism(s: str, o: str, h: str, u: str) -> Tuple[str, List[str]]:
    """Plagiarism check using structural similarity of AI-generated code."""
    messages_s = [
        {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
        {"role": "user", "content": s}
    ]
    s_code = await query_external_api(messages_s)
    messages_o = [
        {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
        {"role": "user", "content": o}
    ]
    o_code = await query_external_api(messages_o)
    if not s_code or "API unavailable" in s_code:
        s_result = False, 0.0, "API unavailable"
    else:
        orig_score_s = compute_structural_complexity(map_chars_to_grid(pad_code(s)))
        gen_score_s = compute_structural_complexity(map_chars_to_grid(pad_code(s_code)))
        delta_s = abs(orig_score_s - gen_score_s)
        s_result = delta_s < CODE_THRESHOLDS["t"], delta_s, f"Delta: {delta_s:.4f}"

    if not o_code or "API unavailable" in o_code:
        o_result = False, 0.0, "API unavailable"
    else:
        orig_score_o = compute_structural_complexity(map_chars_to_grid(pad_code(o)))
        gen_score_o = compute_structural_complexity(map_chars_to_grid(pad_code(o_code)))
        delta_o = abs(orig_score_o - gen_score_o)
        o_result = delta_o < CODE_THRESHOLDS["t"], delta_o, f"Delta: {delta_o:.4f}"

    p, d, e = s_result
    q, f, g = o_result
    if p and q:
        return "S", [e, g, "Both AI"]
    return "N", [e, g]

async def model3_plagiarism(s: str, o: str, h: str, u: str) -> Tuple[str, List[str]]:
    """Combined plagiarism check."""
    messages_s = [
        {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
        {"role": "user", "content": s}
    ]
    s_code = await query_external_api(messages_s)
    messages_o = [
        {"role": "system", "content": "You are a code analysis assistant. Generate C++ code that matches the functionality of the given code."},
        {"role": "user", "content": o}
    ]
    o_code = await query_external_api(messages_o)
    if not s_code or "API unavailable" in s_code:
        s_result = False, 0.0, "API unavailable", "N", ["API unavailable"]
    else:
        orig_score_s = compute_structural_complexity(map_chars_to_grid(pad_code(s)))
        gen_score_s = compute_structural_complexity(map_chars_to_grid(pad_code(s_code)))
        delta_s = abs(orig_score_s - gen_score_s)
        plag_label_s, plag_evidence_s = model1_plagiarism(s, s_code, "U", "O")
        lcs_similar_s, lcs_message_s = lcs_judge(s, s_code)
        evidence_s = [f"Delta: {delta_s:.4f}", lcs_message_s] + plag_evidence_s
        is_similar_s = delta_s < CODE_THRESHOLDS["u"] or plag_label_s == "S" or lcs_similar_s
        s_result = is_similar_s, delta_s, f"Delta: {delta_s:.4f}", plag_label_s, evidence_s

    if not o_code or "API unavailable" in o_code:
        o_result = False, 0.0, "API unavailable", "N", ["API unavailable"]
    else:
        orig_score_o = compute_structural_complexity(map_chars_to_grid(pad_code(o)))
        gen_score_o = compute_structural_complexity(map_chars_to_grid(pad_code(o_code)))
        delta_o = abs(orig_score_o - gen_score_o)
        plag_label_o, plag_evidence_o = model1_plagiarism(o, o_code, "U", "O")
        lcs_similar_o, lcs_message_o = lcs_judge(o, o_code)
        evidence_o = [f"Delta: {delta_o:.4f}", lcs_message_o] + plag_evidence_o
        is_similar_o = delta_o < CODE_THRESHOLDS["u"] or plag_label_o == "S" or lcs_similar_o
        o_result = is_similar_o, delta_o, f"Delta: {delta_o:.4f}", plag_label_o, evidence_o

    p, d, e, l, v = s_result
    q, f, g, m, w = o_result
    x, y = lcs_judge(s, o)
    z = [e, g, f"LCS1: {l}", f"LCS2: {m}", y] + v + w
    return "S" if p and q or l == "S" or m == "S" or x else "N", z

def process_code(code: str) -> str:
    """Process code by removing comments and normalizing it."""
    processed = []
    i = 0
    while i < len(code):
        if code[i:i+2] == '//':
            while i < len(code) and code[i] != '\n':
                i += 1
        elif code[i:i+2] == '/*':
            i += 2
            while i < len(code) and code[i:i+2] != '*/':
                i += 1
            if i < len(code):
                i += 2
        else:
            if code[i] not in ' \t\n':
                processed.append(code[i])
            if code[i] in '{};' or (i > 0 and i % 500 == 0):
                processed.extend(['x', 'c', '9', '@'])
        i += 1
    return ''.join(processed)

def lcs(va: str, vb: str) -> int:
    """Compute Longest Common Subsequence."""
    m = len(va)
    n = len(vb)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if va[i - 1] == vb[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def hash_segment(segment: str) -> int:
    """Generate hash for a code segment."""
    sum_all = sum(ord(c) for c in segment)
    sum_alpha = sum(ord(c) for c in segment if c.islower() and c.isalpha())
    sum_non_alpha = sum(ord(c) for c in segment if not (c.islower() and c.isalpha()))
    sum_mod7 = sum(ord(c) for c in segment if ord(c) % 7 == 0)
    max_char = max(ord(c) for c in segment) if segment else 0
    min_char = min(ord(c) for c in segment) if segment else 1000
    length = len(segment)
    h = (sum_all * 132 + sum_alpha * 232 + sum_non_alpha * 67 + sum_mod7 * 135 +
         length * 559 + max_char * 7 + min_char * 11 +
         sum_all // 23 + sum_alpha // 45 + sum_non_alpha // 10 + sum_mod7 // 2 +
         max_char // 7 + min_char // 7 + sum_all % 232)
    return h

def extract_segments(code: str) -> List[str]:
    """Extract segments from processed code."""
    segments = []
    current_segment = []
    i = 0
    while i < len(code):
        if i >= 3 and code[i-3:i+1] == 'xc9@':
            if current_segment:
                segments.append(''.join(current_segment))
                current_segment = []
            i += 1
        else:
            current_segment.append(code[i])
            i += 1
    if current_segment:
        segments.append(''.join(current_segment))
    return segments

def compare2(va: str, vb: str) -> Tuple[int, int]:
    """Hash-based similarity comparison."""
    segments_a = extract_segments(va)
    segments_b = extract_segments(vb)
    hashes_a = sorted([hash_segment(seg) for seg in segments_a])
    hashes_b = sorted([hash_segment(seg) for seg in segments_b])
    common = 0
    i, j = 0, 0
    while i < len(hashes_a) and j < len(hashes_b):
        if hashes_a[i] == hashes_b[j]:
            common += 1
            i += 1
            j += 1
        elif hashes_a[i] < hashes_b[j]:
            i += 1
        else:
            j += 1
    min_len = min(len(hashes_a), len(hashes_b))
    return common, min_len

def compare3(va: str, na: int, vb: str, nb: int) -> int:
    """Third comparison method."""
    vd = va.replace('xc9@', '$$$$')
    ve = vb.replace('xc9@', '$$$$')
    nx = [ord(vd[i]) * 1500000 + ord(vd[i]) * 1500 + ord(vd[i])
          for i in range(len(vd) - 2) if vd[i:i+3] != '$$$']
    nc = [ord(ve[i]) * 1500000 + ord(ve[i]) * 1500 + ord(ve[i])
          for i in range(len(ve) - 2) if ve[i:i+3] != '$$$']
    nx.sort()
    nc.sort()
    num = 0
    i1, i2 = 0, 0
    while i1 < len(nx) and i2 < len(nc):
        if nx[i1] == nc[i2]:
            num += 1
            i1 += 1
            i2 += 1
        elif nx[i1] < nc[i2]:
            i1 += 1
        else:
            i2 += 1
    return num

def prepare_code(c: str, h: str) -> None:
    """Prepare and store code submission."""
    c = strip_comments(c)
    o = []
    for l in c.split():
        for k in l:
            if k in "{};":
                o.extend(["x", "c", "9"])
            elif k not in " \t":
                o.append(k)
        o.extend(["x", "c", "9"])
    o.append(" ")
    with INPUT_FILE2.open("w") as f:
        f.write(c + "\nRqq7\n" + h + "\n ")
    with INPUT_FILE.open("a+") as f:
        f.write("".join(o) + h + "\n")

def generate_report(a: Dict[str, Any], r: List[str], h: str, model1_result: Tuple[str, List[str]] = None,
                   model2_result: Tuple[str, List[str]] = None, model3_result: Tuple[str, List[str]] = None) -> str:
    """Generate analysis report."""
    s = f"Report for {h}\n\n"
    s += "Model 1 (AI/Human):\n"
    s += f"AI: {a['l']} (Conf: {a['c']:.2%})\n"
    s += "<span style='color: blue;'>Suspicious:</span>\n<span style='color: blue;'>- Variables</span>\n<span style='color: blue;'>- Comments</span>\nRules:\n"
    s += "\n".join([f"- {x}" for x in r]) if r else "- None\n"
    if model1_result:
        s += "\nModel 1 Plagiarism:\n" + f"Similarity: {model1_result[0]}\nEvidence:\n" + "\n".join([f"- {x}" for x in model1_result[1]]) + "\n"
    if model2_result:
        s += "\nModel 2 Plagiarism:\n" + f"Similarity: {model2_result[0]}\nEvidence:\n" + "\n".join([f"- {x}" for x in model2_result[1]]) + "\n"
    if model3_result:
        s += "\nModel 3 Plagiarism:\n" + f"Similarity: {model3_result[0]}\nEvidence:\n" + "\n".join([f"- {x}" for x in model3_result[1]]) + "\n"
    v = sum(1 for x in [a['l'] == "AI", model2_result and model2_result[0] == "S", model3_result and model3_result[0] == "S"] if x)
    s += "\nConclusion:\n" + ("- AI\n" if v >= 2 else "- Human\n")
    s += "\nRecommendations:\n"
    if a['l'] == "AI" or r or (model1_result and model1_result[0] == "S") or (model2_result and model2_result[0] == "S") or (model3_result and model3_result[0] == "S"):
        s += "- Review for AI/Plagiarism.\n- Verify.\n"
        if a['l'] == "AI":
            s += "- Discuss AI usage.\n"
        if (model1_result and model1_result[0] == "S") or (model2_result and model2_result[0] == "S") or (model3_result and model3_result[0] == "S"):
            s += "- Investigate plagiarism.\n"
    else:
        s += "- Human-authored.\n"
    return s

# ──────────────── ROUTES ────────────────
@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=JSONResponse)
async def analyze_code(payload: SingleCode):
    """Analyze a single code for AI/Human classification."""
    code_clean = strip_comments(payload.code)
    s, l, f, h = await ai_human_classify(payload.code, payload.handle)
    if not s:
        return {"handle": h, "error": l}
    a = {"l": l, "c": f}
    report = generate_report(a, [], h)
    return {
        "handle": h,
        "ai_probability": f,
        "label": l,
        "report": report,
        "message": "Analysis complete"
    }

@app.post("/compare", response_class=JSONResponse)
async def compare_codes(pair: CodePair):
    """Compare two codes for plagiarism."""
    verdict, evidence = lcs_judge(pair.code1, pair.code2)
    processed_code1 = process_code(pair.code1)
    processed_code2 = process_code(pair.code2)
    d = lcs(processed_code1, processed_code2)
    e, g = compare2(processed_code1, processed_code2)
    h = compare3(processed_code1, len(processed_code1), processed_code2, len(processed_code2))
    is_plagiarized = (d >= 1100 or 
                      e >= g / 2 + g / 5 or 
                      e >= 100 or 
                      (e >= 70 and d >= 700) or 
                      d >= len(processed_code1) / 2 + len(processed_code1) / 3 + g * 2 or 
                      h >= 240)
    return {
        "handle1": pair.handle1,
        "handle2": pair.handle2,
        "plagiarised": is_plagiarized,
        "lcs": evidence,
        "detailed_metrics": {
            "lcs_score": d,
            "hash_common": e,
            "hash_min": g,
            "compare3_score": h
        },
        "message": f"{pair.handle1} and {pair.handle2} {'plagiarized' if is_plagiarized else 'not plagiarized'}"
    }

@app.post("/plagiarism_batch_compare", response_class=JSONResponse)
async def plagiarism_batch_compare(payload: BatchCompareRequest):
    """Batch plagiarism checking."""
    if len(payload.codes) != len(payload.handles):
        return {"error": "Number of codes and handles must match"}
    
    processed_codes = [process_code(code) for code in payload.codes]
    similar_pairs = []
    cheaters = set()
    
    for i in range(len(processed_codes)):
        for j in range(i + 1, len(processed_codes)):
            d = lcs(processed_codes[i], processed_codes[j])
            e, g = compare2(processed_codes[i], processed_codes[j])
            h = compare3(processed_codes[i], len(processed_codes[i]), processed_codes[j], len(processed_codes[j]))
            
            if (d >= 1100 or 
                e >= g / 2 + g / 5 or 
                e >= 100 or 
                (e >= 70 and d >= 700) or 
                d >= len(processed_codes[i]) / 2 + len(processed_codes[i]) / 3 + g * 2 or 
                h >= 240):
                similar_pairs.append({
                    "handle1": payload.handles[i],
                    "handle2": payload.handles[j],
                    "lcs": d,
                    "hash_common": e,
                    "hash_min": g,
                    "compare3": h
                })
                cheaters.add(payload.handles[i])
                cheaters.add(payload.handles[j])
    
    return {
        "similar_pairs": similar_pairs,
        "num_similar_pairs": len(similar_pairs),
        "num_cheaters": len(cheaters),
        "cheaters": list(cheaters),
        "message": "Batch comparison complete"
    }

@app.post("/a", response_class=HTMLResponse)
async def analyze_form(request: Request, c: str = Form(...), h: str = Form("user1")):
    """Form-based single code analysis."""
    s, l, f, e = await ai_human_classify(c, h)
    if not s:
        return templates.TemplateResponse("index.html", {"request": request, "error": l})
    v, u = variable_check(c)
    d = [f"Var > {CODE_THRESHOLDS['v']} chars" if u else "", "Susp com" if suspicious_comment_block(c) else ""]
    d = [x for x in d if x]
    p, q, w = await check_structural_similarity(c)
    k = ("S" if p else "N", [w])
    b, q, w, m, v = await check_combined_similarity(c)
    o = ("S" if b else "N", [w] + v)
    a = {"l": l, "c": f}
    report = generate_report(a, d, h, model2_result=k, model3_result=o)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": {"handle": e, "report": report, "confidence": f, "label": l}
    })

@app.post("/c", response_class=HTMLResponse)
async def compare_batch_form(request: Request):
    """Form-based batch comparison."""
    f = await request.form()
    codes = []
    handles = []
    i = 1
    while f"code{i}" in f and f"handle{i}" in f:
        codes.append(f[f"code{i}"])
        handles.append(f[f"handle{i}"])
        i += 1
    if len(codes) < 2:
        raise HTTPException(status_code=400, detail="Need 2+ codes")
    u = []
    for i in range(0, len(codes), 2):
        if i + 1 >= len(codes):
            break
        s, p = codes[i], handles[i]
        o, q = codes[i + 1], handles[i + 1]
        k, e = model1_plagiarism(s, o, p, q)
        m, n = await model2_plagiarism(s, o, p, q)
        b, v = await model3_plagiarism(s, o, p, q)
        _, x = variable_check(s)
        _, y = variable_check(o)
        w = []
        if x:
            w.append(f"{p}: Var > {CODE_THRESHOLDS['v']} chars")
        if y:
            w.append(f"{q}: Var > {CODE_THRESHOLDS['v']} chars")
        if suspicious_comment_block(s):
            w.append(f"{p}: Susp com")
        if suspicious_comment_block(o):
            w.append(f"{q}: Susp com")
        a = {"l": "N/A", "c": 0.0}
        report = generate_report(a, w, f"{p} vs {q}", model1_result=k, model2_result=m, model3_result=b)
        u.append({
            "handle1": p,
            "handle2": q,
            "similarity": k[0],
            "message": f"{p} and {q} {'plagiarised' if any(x == 'S' for x in [k[0], m[0], b[0]]) else 'not plagiarised'}",
            "evidence": w + e + ["\nM2 Evidence:"] + n + ["\nM3 Evidence:"] + v,
            "report": report
        })
        await asyncio.sleep(2)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "batch_results": u,
        "num_similar_pairs": len(u),
        "num_cheaters": len(set([r["handle1"] for r in u] + [r["handle2"] for r in u])),
        "cheaters": list(set([r["handle1"] for r in u] + [r["handle2"] for r in u]))
    })

@app.post("/p", response_class=HTMLResponse)
async def prepare_form(request: Request, c: str = Form(...), h: str = Form(...)):
    """Prepare code submission."""
    prepare_code(c, h)
    return templates.TemplateResponse("index.html", {"request": request, "message": f"Code for {h} prepared"})

@app.post("/compare_lines", response_class=JSONResponse)
async def compare_lines(request: Request):
    """Compare two codes line-by-line."""
    data = await request.json()
    code1 = data.get("code1", "")
    code2 = data.get("code2", "")
    if not code1 or not code2:
        return {"error": "Both codes are required"}
    similar, message = lcs_judge(code1, code2)
    return {"similar": similar, "details": message}

# ──────────────── MAIN ────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)