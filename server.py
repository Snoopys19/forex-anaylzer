# === BEGIN ASTRAFX SERVER.PY ===
# AstraFX Backend v2.3 — caching + coalescing, static site, API unchanged

import os
import time
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set

import requests
from fastapi import APIRouter, Body, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, Response, FileResponse

# ------------------ Minimal email+password auth + gate ------------------
import base64 as _b64, hmac as _hmac, hashlib as _hashlib, datetime as _dt, time as _time
import os as _os

USERS_LOCAL_PATH = _os.environ.get("USERS_LOCAL_PATH", "data/users.csv")
_os.makedirs(_os.path.dirname(USERS_LOCAL_PATH) or ".", exist_ok=True)

def _hash_pw(password: str):
    salt = _os.urandom(16)
    dk = _hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return salt.hex(), dk.hex()

def _verify_pw(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    dk = _hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return _hmac.compare_digest(dk.hex(), hash_hex)

def _load_users():
    users = {}
    try:
        import csv
        with open(USERS_LOCAL_PATH, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                em = (row.get("email") or "").strip().lower()
                if em:
                    users[em] = (row.get("salt",""), row.get("hash",""))
    except FileNotFoundError:
        pass
    return users

def _save_user(email: str, salt_hex: str, hash_hex: str):
    header_needed = not _os.path.exists(USERS_LOCAL_PATH)
    import csv
    with open(USERS_LOCAL_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed: w.writerow(["timestamp","email","salt","hash"])
        ts = _dt.datetime.utcnow().isoformat()+"Z"
        w.writerow([ts, email, salt_hex, hash_hex])

def _commit_users_csv():
    try:
        with open(USERS_LOCAL_PATH, "rb") as f: data_bytes = f.read()
        return _github_commit("data/users.csv", data_bytes, "Update users.csv")
    except Exception as e:
        return {"error": str(e)}

SESSION_SECRET = _os.environ.get("SESSION_SECRET", "change-me-please").encode("utf-8")
COOKIE_NAME    = _os.environ.get("COOKIE_NAME", "astrafx_session")
COOKIE_SECURE  = _os.environ.get("COOKIE_SECURE", "1") != "0"

def _make_token(email: str, days: int = 7) -> str:
    exp = int(_time.time()) + days * 86400
    payload = f"{email}|{exp}"
    sig = _hmac.new(SESSION_SECRET, payload.encode("utf-8"), digestmod="sha256").hexdigest()
    raw = f"{payload}|{sig}".encode("utf-8")
    return _b64.urlsafe_b64encode(raw).decode("utf-8")

def _verify_token(token: str) -> str:
    try:
        raw = _b64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
        email, exp, sig = raw.split("|")
        if int(exp) < int(_time.time()): return ""
        exp_sig = _hmac.new(SESSION_SECRET, f"{email}|{exp}".encode("utf-8"), digestmod="sha256").hexdigest()
        if not _hmac.compare_digest(sig, exp_sig): return ""
        return email
    except Exception:
        return ""

def _set_session_cookie(response, email: str):
    token = _make_token(email)
    response.set_cookie(key=COOKIE_NAME, value=token, max_age=7*86400,
                        httponly=True, samesite="Lax", secure=COOKIE_SECURE, path="/")
from starlette.middleware.base import BaseHTTPMiddleware

# --------------------------------------------------------------------------------------
# App + CORS
# --------------------------------------------------------------------------------------
app = FastAPI(title="AstraFX Backend v2.3", version="2.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# In-memory TTL cache + request coalescing (+ optional stale-while-revalidate)
# --------------------------------------------------------------------------------------
CACHE_TTL_SEC: int = int(os.environ.get("CACHE_TTL_SEC", "45"))      # 0 disables cache
CACHE_SWR_SEC: int = int(os.environ.get("CACHE_SWR_SEC", "0"))       # e.g., 25 enables SWR
CACHE_PATHS: Set[str] = set(
    p.strip() for p in os.environ.get("CACHE_PATHS", "/api/ohlc,/api/scan").split(",") if p.strip()
)
CACHE_MAX_ENTRIES: Optional[int] = int(os.environ.get("CACHE_MAX_ENTRIES", "0")) or None

@dataclass
class _CacheEntry:
    body: bytes
    status_code: int
    headers: List[Tuple[str, str]]   # exclude content-length; we recompute
    media_type: Optional[str]
    expires_at: float

    def is_fresh(self, now: float) -> bool:
        return now < self.expires_at

class _HTTPCache:
    def __init__(self) -> None:
        self.entries: Dict[str, _CacheEntry] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        # metrics
        self.hits = 0
        self.misses = 0
        self.stale_served = 0
        self.revalidations = 0
        self.coalesced_waiters = 0

    def get_lock(self, key: str) -> asyncio.Lock:
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        return self.locks[key]

    def get(self, key: str) -> Optional[_CacheEntry]:
        return self.entries.get(key)

    def set(self, key: str, entry: _CacheEntry) -> None:
        self.entries[key] = entry
        if CACHE_MAX_ENTRIES and len(self.entries) > CACHE_MAX_ENTRIES:
            now = time.time()
            # prune expired first
            for k in [k for k, v in list(self.entries.items()) if not v.is_fresh(now)]:
                if len(self.entries) <= CACHE_MAX_ENTRIES:
                    break
                self.entries.pop(k, None)
            # still over? pop arbitrary
            while len(self.entries) > CACHE_MAX_ENTRIES:
                self.entries.pop(next(iter(self.entries)))

_cache = _HTTPCache()

class _CoalescingTTLMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, ttl: int, swr: int, paths: Set[str]) -> None:
        super().__init__(app)
        self.ttl = ttl
        self.swr = swr
        self.paths = paths

    @staticmethod
    def _fmt_utc(ts: float) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

    @staticmethod
    def _key_from_request(request) -> str:
        items = sorted(request.query_params.multi_items())
        qs = "&".join(f"{k}={v}" for k, v in items)
        return f"{request.url.path}?{qs}"

    def _build_response(self, entry: _CacheEntry, cache_status: str, expires_at: float) -> Response:
        headers = dict(entry.headers)
        headers["X-Cache"] = cache_status
        headers["X-Cache-Expires"] = self._fmt_utc(expires_at)
        return Response(
            content=entry.body,
            status_code=entry.status_code,
            headers=headers,
            media_type=entry.media_type,
        )

    async def _refresh(self, key, request, call_next):
        lock = _cache.get_lock(key)
        async with lock:
            response = await call_next(request)
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            if response.status_code == 200:
                headers = [(k, v) for k, v in response.headers.items() if k.lower() != "content-length"]
                _cache.revalidations += 1
                _cache.set(
                    key,
                    _CacheEntry(
                        body=body,
                        status_code=response.status_code,
                        headers=headers,
                        media_type=response.media_type,
                        expires_at=time.time() + self.ttl,
                    ),
                )

    async def dispatch(self, request, call_next):
        # Only GET + whitelisted paths + TTL>0
        if request.method != "GET" or request.url.path not in self.paths or self.ttl <= 0:
            return await call_next(request)

        key = self._key_from_request(request)
        now = time.time()
        entry = _cache.get(key)

        # fresh
        if entry and entry.is_fresh(now):
            _cache.hits += 1
            return self._build_response(entry, "HIT", entry.expires_at)

        # stale-while-revalidate
        if entry and self.swr > 0 and now < entry.expires_at + self.swr:
            _cache.stale_served += 1
            lock = _cache.get_lock(key)
            if not lock.locked():
                asyncio.create_task(self._refresh(key, request, call_next))
            return self._build_response(entry, "STALE", entry.expires_at)

        # coalesce
        lock = _cache.get_lock(key)
        if lock.locked():
            _cache.coalesced_waiters += 1
            async with lock:
                refreshed = _cache.get(key)
                if refreshed and refreshed.is_fresh(time.time()):
                    return self._build_response(refreshed, "COALESCED", refreshed.expires_at)

        async with lock:
            # double-check
            entry2 = _cache.get(key)
            if entry2 and entry2.is_fresh(time.time()):
                _cache.hits += 1
                return self._build_response(entry2, "HIT", entry2.expires_at)

            _cache.misses += 1
            response = await call_next(request)

            # clone body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk

            headers = [(k, v) for k, v in response.headers.items() if k.lower() != "content-length"]
            if response.status_code == 200:
                expires_at = time.time() + self.ttl
                _cache.set(
                    key,
                    _CacheEntry(
                        body=body,
                        status_code=response.status_code,
                        headers=headers,
                        media_type=response.media_type,
                        expires_at=expires_at,
                    ),
                )
                headers_dict = dict(headers)
                headers_dict["X-Cache"] = "MISS"
                headers_dict["X-Cache-Expires"] = self._fmt_utc(expires_at)
                return Response(content=body, status_code=response.status_code, headers=headers_dict, media_type=response.media_type)

            # non-200: don't cache
            return Response(content=body, status_code=response.status_code, headers=dict(headers), media_type=response.media_type)

# register middleware + metrics
app.add_middleware(_CoalescingTTLMiddleware, ttl=CACHE_TTL_SEC, swr=CACHE_SWR_SEC, paths=CACHE_PATHS)

_cache_router = APIRouter()

@_cache_router.get("/cache/metrics")
def cache_metrics():
    return {
        "cache": {
            "enabled_paths": sorted(CACHE_PATHS),
            "ttl_seconds": CACHE_TTL_SEC,
            "swr_seconds": CACHE_SWR_SEC,
            "entries": len(_cache.entries),
            "hits": _cache.hits,
            "misses": _cache.misses,
            "stale_served": _cache.stale_served,
            "revalidations": _cache.revalidations,
            "coalesced_waiters": _cache.coalesced_waiters,
        }
    }

app.include_router(_cache_router)

# --- Simple signup capture (stores email locally and optionally commits to GitHub) ---
from fastapi import Body
import hashlib, base64, json, datetime

SIGNUP_LOCAL_PATH = os.environ.get("SIGNUP_LOCAL_PATH", "data/signups.csv")
os.makedirs(os.path.dirname(SIGNUP_LOCAL_PATH), exist_ok=True)

def _append_local_signup(email: str):
    header_needed = not os.path.exists(SIGNUP_LOCAL_PATH)
    with open(SIGNUP_LOCAL_PATH, "a") as f:
        if header_needed:
            f.write("timestamp,email\n")
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        f.write(f"{ts},{email}\n")

def _github_commit(path: str, content_bytes: bytes, message: str):
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPO", "").strip()   # e.g. 'Snoopys19/forex-anaylzer'
    branch = os.environ.get("GITHUB_BRANCH", "main").strip()
    if not token or not repo:
        return {"skipped": True, "reason": "missing token or repo"}
    # Get current file SHA if exists
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    params = {"ref": branch}
    r = requests.get(api_url, headers=headers, params=params, timeout=20)
    sha = None
    if r.status_code == 200:
        js = r.json()
        sha = js.get("sha")
    # Prepare request
    put_body = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        put_body["sha"] = sha
    r2 = requests.put(api_url, headers=headers, json=put_body, timeout=20)
    return {"status": r2.status_code, "text": r2.text[:200]}

@app.post("/api/signup_email_capture")  # legacy email-only
def api_signup(payload: Dict[str, Any] = Body(...)):
    email = (payload.get("email") or "").strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email required")
    # 1) Local append
    _append_local_signup(email)
    # 2) Optional GitHub commit (append to data/signups.csv in repo)
    try:
        with open(SIGNUP_LOCAL_PATH, "rb") as f:
            data_bytes = f.read()
        gh = _github_commit("data/signups.csv", data_bytes, f"Add signup {email}")
    except Exception as e:
        gh = {"error": str(e)}
    return {"ok": True, "email": email, "github": gh}


# --------------------------------------------------------------------------------------
# Data providers
# --------------------------------------------------------------------------------------
DATA_PROVIDER = os.getenv("DATA_PROVIDER", "OANDA").upper()

def oanda_base_url() -> str:
    env = os.getenv("OANDA_ENV", "practice").lower()
    return "https://api-fxpractice.oanda.com" if env != "live" else "https://api-fxtrade.oanda.com"

def oanda_headers() -> Dict[str, str]:
    token = os.getenv("OANDA_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=500, detail="Missing OANDA_TOKEN env var")
    return {"Authorization": f"Bearer {token}"}

def map_pair_to_oanda(pair: str) -> str:
    pair = pair.upper().strip()
    if pair == "XAUUSD":
        return "XAU_USD"
    if len(pair) == 6:
        return pair[0:3] + "_" + pair[3:6]
    return pair.replace("/", "_")

def map_tf_to_oanda(tf: str) -> str:
    tf = tf.upper()
    return {
        "M5":"M5","M15":"M15","M30":"M30",
        "H1":"H1","H4":"H4","D1":"D","D":"D",
    }.get(tf, "H1")

def fetch_oanda_candles(pair: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    instrument = map_pair_to_oanda(pair)
    gran = map_tf_to_oanda(tf)
    url = f"{oanda_base_url()}/v3/instruments/{instrument}/candles"
    params = {"granularity": gran, "count": max(1, min(int(limit), 500)), "price": "M"}
    r = requests.get(url, headers=oanda_headers(), params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OANDA candles error: {r.status_code} {r.text[:180]}")
    data = r.json()
    out: List[Dict[str, Any]] = []
    for c in data.get("candles", []):
        if not c.get("complete", False):
            continue
        mid = c.get("mid", {})
        out.append({
            "time": c.get("time"),
            "open": float(mid.get("o")),
            "high": float(mid.get("h")),
            "low": float(mid.get("l")),
            "close": float(mid.get("c")),
        })
    return out

def fetch_twelvedata_ohlc(pair: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    key = os.getenv("TWELVEDATA_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="Missing TWELVEDATA_KEY")
    interval = {"M5":"5min","M15":"15min","M30":"30min","H1":"1h","H4":"4h","D1":"1day"}.get(tf.upper(),"1h")
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol": pair, "interval": interval, "outputsize": min(limit,500), "apikey": key}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"TwelveData error: {r.status_code} {r.text[:180]}")
    vals = list(reversed(r.json().get("values") or []))
    return [{
        "time": v.get("datetime"),
        "open": float(v["open"]),
        "high": float(v["high"]),
        "low": float(v["low"]),
        "close": float(v["close"]),
    } for v in vals]

def fetch_alphavantage_ohlc(pair: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    key = os.getenv("ALPHAVANTAGE_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="Missing ALPHAVANTAGE_KEY")
    func = {"H1":"FX_INTRADAY","H4":"FX_INTRADAY","D1":"FX_DAILY","M5":"FX_INTRADAY","M15":"FX_INTRADAY","M30":"FX_INTRADAY"}.get(tf.upper(),"FX_INTRADAY")
    interval = {"M5":"5min","M15":"15min","M30":"30min","H1":"60min","H4":"240min"}.get(tf.upper(),"60min")
    params = {"function": func, "from_symbol": pair[0:3], "to_symbol": pair[3:6], "apikey": key}
    if func == "FX_INTRADAY":
        params["interval"] = interval
    url = "https://www.alphavantage.co/query"
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"AlphaVantage error: {r.status_code} {r.text[:180]}")
    js = r.json()
    series_key = next((k for k in js.keys() if "Time Series" in k), None)
    if not series_key:
        raise HTTPException(status_code=502, detail=f"AlphaVantage no series: {list(js.keys())}")
    raw = js[series_key]
    rows = sorted(raw.items())
    return [{
        "time": t,
        "open": float(v["1. open"]),
        "high": float(v["2. high"]),
        "low": float(v["3. low"]),
        "close": float(v["4. close"]),
    } for t, v in rows[-limit:]]

def fetch_ohlc_router(pair: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    prov = os.getenv("DATA_PROVIDER", DATA_PROVIDER).upper()
    if prov == "OANDA":
        return fetch_oanda_candles(pair, tf, limit)
    if prov == "TWELVEDATA":
        return fetch_twelvedata_ohlc(pair, tf, limit)
    if prov == "ALPHAVANTAGE":
        return fetch_alphavantage_ohlc(pair, tf, limit)
    # MOCK fallback
    seed = 1.2345 if pair != "XAUUSD" else 2400.0
    out: List[Dict[str, Any]] = []
    val = seed
    for i in range(limit):
        o = val * (1 + (0.0005 - 0.001 * (i % 2)))
        h = o * (1 + 0.001)
        l = o * (1 - 0.001)
        c = o * (1 + (0.0002 - 0.0004 * (i % 2)))
        out.append({"time": i, "open": o, "high": h, "low": l, "close": c})
        val = c * (1 + (i - limit/2) * 1e-4)
    return out

# --------------------------------------------------------------------------------------
# TA helpers, patterns, strategies
# --------------------------------------------------------------------------------------
def ema(values: List[float], length: int) -> List[float]:
    if length <= 1 or not values:
        return values[:]
    k = 2.0 / (length + 1)
    out: List[float] = []
    prev = values[0]
    for v in values:
        prev = v * k + prev * (1 - k)
        out.append(prev)
    return out

def sma(values: List[float], length: int) -> List[float]:
    out: List[float] = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= length:
            s -= values[i - length]
        out.append(s / length if i >= length - 1 else values[i])
    return out

def stdev(values: List[float], length: int) -> List[float]:
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - length + 1)
        window = values[start:i+1]
        if len(window) < 2:
            out.append(0.0)
        else:
            m = sum(window) / len(window)
            var = sum((x - m) ** 2 for x in window) / (len(window) - 1)
            out.append(var ** 0.5)
    return out

def rsi(closes: List[float], length: int = 14) -> List[float]:
    if len(closes) < length + 1:
        return [50.0] * len(closes)
    gains = [0.0]; losses = [0.0]
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0.0)); losses.append(max(-ch, 0.0))
    avg_gain = sum(gains[1:length+1]) / length
    avg_loss = sum(losses[1:length+1]) / length
    rsis = [50.0] * len(closes)
    for i in range(length+1, len(closes)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rs = (avg_gain / avg_loss) if avg_loss != 0 else float("inf")
        rsis[i] = 100 - (100 / (1 + rs))
    return rsis

def atr(highs: List[float], lows: List[float], closes: List[float], length: int = 14) -> List[float]:
    if len(closes) < 2:
        return [0.0] * len(closes)
    trs = [0.0]
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    out = [0.0] * len(trs)
    if len(trs) <= length:
        return out
    out[length] = sum(trs[1:length+1]) / length
    for i in range(length+1, len(trs)):
        out[i] = (out[i-1] * (length - 1) + trs[i]) / length
    return out

def highest(values: List[float], lookback: int, end: Optional[int] = None) -> float:
    end = len(values) if end is None else end
    start = max(0, end - lookback)
    return max(values[start:end])

def lowest(values: List[float], lookback: int, end: Optional[int] = None) -> float:
    end = len(values) if end is None else end
    start = max(0, end - lookback)
    return min(values[start:end])

def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = int(max(0, min(len(xs) - 1, round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]

class Detected:
    def __init__(self, name: str, direction: str, entry: float, stop: float, score: float, tags: List[str]):
        self.name = name
        self.direction = direction
        self.entry = float(entry)
        self.stop = float(stop)
        self.score = float(score)
        self.tags = tags
    def to_signal(self, pair: str, tf: str) -> Dict[str, Any]:
        risk = abs(self.entry - self.stop)
        tps = [self.entry + (risk * r if self.direction == "long" else -risk * r) for r in (1.0, 2.0, 3.0)]
        tv_tf = {"M5": "5", "M15": "15", "M30": "30", "H1": "60", "H4": "240", "D1": "D"}.get(tf, "60")
        tv_sym = f"OANDA:{pair}"
        return {
            "pair": pair, "tf": tf, "score": int(round(self.score)), "setup": self.name,
            "dir": self.direction, "entry": self.entry, "stop": self.stop,
            "tps": [float(x) for x in tps], "tags": self.tags[:],
            "tv": f"https://www.tradingview.com/chart/?symbol={tv_sym}&interval={tv_tf}",
        }

def detect_pinbar(highs, lows, opens, closes) -> Optional[Detected]:
    i = -1
    o, h, l, c = opens[i], highs[i], lows[i], closes[i]
    body = abs(c - o)
    if body == 0:
        return None
    rng = h - l
    up = c > o
    upper = h - max(c, o)
    lower = min(c, o) - l
    tail = lower if up else upper
    tail_ratio = tail / (body + 1e-9)
    if tail_ratio < 2.0 or body / (rng + 1e-9) > 0.4:
        return None
    direction = "long" if up and lower > upper else "short" if (not up) and upper > lower else None
    if not direction:
        return None
    entry = c
    stop = l if direction == "long" else h
    score = min(100, 60 + tail_ratio * 8)
    return Detected("Pin Bar", direction, entry, stop, score, tags=["pinbar"])

def detect_engulfing(highs, lows, opens, closes) -> Optional[Detected]:
    if len(closes) < 2:
        return None
    o1, c1 = opens[-2], closes[-2]
    o2, c2 = opens[-1], closes[-1]
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    min1, max1 = min(o1, c1), max(o1, c1)
    min2, max2 = min(o2, c2), max(o2, c2)
    if not (min2 <= min1 and max2 >= max1 and body2 > body1 * 0.8):
        return None
    direction = "long" if c2 > o2 and c1 < o1 else "short" if c2 < o2 and c1 > o1 else None
    if not direction:
        return None
    entry = c2
    stop = lows[-2] if direction == "long" else highs[-2]
    score = min(100, 65 + (body2 / (body1 + 1e-9)) * 10)
    return Detected("Engulfing", direction, entry, stop, score, tags=["engulfing"])

def detect_insidebar(highs, lows, opens, closes) -> Optional[Detected]:
    if len(closes) < 2:
        return None
    if highs[-1] <= highs[-2] and lows[-1] >= lows[-2]:
        direction = "long" if closes[-1] >= opens[-1] else "short"
        entry = closes[-1]
        stop = lows[-2] if direction == "long" else highs[-2]
        compression = (highs[-1] - lows[-1]) / ((highs[-2] - lows[-2]) + 1e-9)
        score = min(100, 60 + (1 - compression) * 25)
        return Detected("Inside Bar Break", direction, entry, stop, score, tags=["insidebar"])
    return None

def detect_nr7(highs, lows, opens, closes) -> Optional[Detected]:
    if len(highs) < 7:
        return None
    rng = [h - l for h, l in zip(highs, lows)]
    if rng[-1] > min(rng[-7:]):
        return None
    direction = "long" if closes[-1] >= opens[-1] else "short"
    entry = closes[-1]
    stop = lows[-1] if direction == "long" else highs[-1]
    compress_pct = rng[-1] / (sum(rng[-7:]) / 7.0 + 1e-9)
    score = min(100, 58 + (1 - compress_pct) * 30)
    return Detected("NR7 Breakout", direction, entry, stop, score, tags=["NR7"])

def detect_donchian20(highs, lows, closes) -> Optional[Detected]:
    if len(closes) < 21:
        return None
    hi20 = highest(highs, 20)
    lo20 = lowest(lows, 20)
    c = closes[-1]
    if c > hi20:
        direction = "long"; entry = c; stop = lowest(lows, 10)
    elif c < lo20:
        direction = "short"; entry = c; stop = highest(highs, 10)
    else:
        return None
    ema50 = ema(closes, 50)
    slope = ema50[-1] - ema50[-5] if len(ema50) > 5 else 0.0
    atr14 = atr(highs, lows, closes, 14)
    dist = abs(c - (hi20 if direction == "long" else lo20))
    vol = atr14[-1] if atr14[-1] else 1e-9
    score = min(100, 70 + (dist / vol) * 10 + (5 if (direction == "long" and slope > 0) or (direction == "short" and slope < 0) else 0))
    return Detected("Donchian 20 Break", direction, entry, stop, score, tags=["donchian20"])

def detect_rsi_pullback(highs, lows, closes) -> Optional[Detected]:
    if len(closes) < 60:
        return None
    ema50 = ema(closes, 50)
    slope = ema50[-1] - ema50[-5] if len(ema50) > 5 else 0.0
    rs = rsi(closes, 14)
    c = closes[-1]
    recent = rs[-10:]
    if slope > 0 and min(recent) < 45 and rs[-1] > 50:
        direction = "long"; entry = c; stop = lowest(lows, 10)
        conf = (rs[-1] - 50) + (slope / (abs(c) + 1e-9)) * 1000
        score = min(100, 68 + conf * 0.2)
        return Detected("RSI Pullback (Uptrend)", direction, entry, stop, score, tags=["rsi"])
    if slope < 0 and max(recent) > 55 and rs[-1] < 50:
        direction = "short"; entry = c; stop = highest(highs, 10)
        conf = (50 - rs[-1]) + (-slope / (abs(c) + 1e-9)) * 1000
        score = min(100, 68 + conf * 0.2)
        return Detected("RSI Pullback (Downtrend)", direction, entry, stop, score, tags=["rsi"])
    return None

STRATEGY_TF_MAP = {
    "donchian": {"M15","M30","H1","H4","D1"},
    "bbsqueeze": {"M15","M30","H1"},
    "ema_pullback": {"M15","M30","H1","H4"},
    "ichimoku": {"M30","H1","H4","D1"},
    "orb_london": {"M5","M15","M30"},
    "rsi2": {"M15","H1"},
}

# --- AstraFX Proprietary Strategy #1 (confluence-based) ---
def _round_level_proximity(price: float, pair: str) -> bool:
    """Detect proximity to psychological round or mid levels (00/50 pips)."""
    if pair.endswith("JPY"):
        pip = 0.01
    elif pair == "XAUUSD":
        pip = 0.1
    else:
        pip = 0.0001
    pips = price / pip
    rem = abs(pips % 50)
    return (rem <= 2 or (50 - rem) <= 2)

def _swing_levels(highs, lows, lookback=50):
    hh = max(highs[-lookback:])
    ll = min(lows[-lookback:])
    return hh, ll

def _near(val: float, ref: float, tol: float) -> bool:
    return abs(val - ref) <= tol

def _fib_overlap(close: float, highs, lows, lookback=50, tol=0.002):
    """Check if close is near a Fib pocket (38.2-61.8) or mid-range."""
    hh, ll = _swing_levels(highs, lows, lookback)
    if hh == ll:
        return False
    mid = (hh + ll) * 0.5
    r382 = ll + 0.382 * (hh - ll)
    r618 = ll + 0.618 * (hh - ll)
    return (_near(close, r382, tol * hh) or _near(close, r618, tol * hh) or _near(close, mid, tol * hh))

def _liquidity_sweep(highs, lows, closes, lookback=10):
    """Detect sweep of prior highs or lows and return direction."""
    h_prev = max(highs[-(lookback + 1):-1])
    l_prev = min(lows[-(lookback + 1):-1])
    c = closes[-1]
    h = highs[-1]
    l = lows[-1]
    if h > h_prev and c < h_prev:
        return "short"
    if l < l_prev and c > l_prev:
        return "long"
    return None

def _fvg_context(highs, lows):
    """Check for a fair value gap (FVG) on the last three bars."""
    if len(highs) < 3:
        return False, None
    h3, h2, h1 = highs[-3], highs[-2], highs[-1]
    l3, l2, l1 = lows[-3], lows[-2], lows[-1]
    bull = l2 > h3
    bear = h2 < l3
    if bull:
        return True, "long"
    if bear:
        return True, "short"
    return False, None

def _stoch_rsi_kd(closes, length=14, stochLen=14, k_smooth=3, d_smooth=3):
    """Compute stochastic RSI %K and %D arrays with smoothing."""
    rs = rsi(closes, length)
    vals = rs[-(stochLen + 50):] if len(rs) >= stochLen + 50 else rs[:]
    stoch = []
    for i in range(len(vals)):
        lo = min(vals[max(0, i - stochLen + 1): i + 1])
        hi = max(vals[max(0, i - stochLen + 1): i + 1])
        stoch.append(50.0 if hi == lo else (vals[i] - lo) / (hi - lo) * 100.0)
    def sma(xs, n):
        if n <= 1:
            return xs[:]
        ret = []
        s = 0.0
        for i, x in enumerate(xs):
            s += x
            if i >= n:
                s -= xs[i - n]
            ret.append(s / min(i + 1, n))
        return ret
    K = sma(stoch, k_smooth)
    D_vals = sma(K, d_smooth)
    return K, D_vals, rs

def det_astra1_strategy(pair: str, tf: str, highs, lows, opens, closes, times) -> Optional[Detected]:
    """Proprietary AstraFX Strategy #1: requires ≥ 6 confluences."""
    if len(closes) < 60:
        return None
    # Calculate helpers
    a = atr(highs, lows, closes, 14)
    e20 = ema(closes, 20)
    e50 = ema(closes, 50)
    slope50 = e50[-1] - e50[-5] if len(e50) >= 5 else 0.0
    trend_dir = "long" if slope50 > 0 else "short" if slope50 < 0 else None
    hh, ll = _swing_levels(highs, lows, 50)
    tol = max(a[-1] * 0.25, closes[-1] * 0.0005)
    at_sr_long = _near(closes[-1], ll, tol)
    at_sr_short = _near(closes[-1], hh, tol)
    sweep = _liquidity_sweep(highs, lows, closes, 10)
    has_fvg, fvg_dir = _fvg_context(highs, lows)
    # Trigger from existing pattern detectors
    trig = None
    trig_dir = None
    d_pin = detect_pinbar(highs, lows, opens, closes)
    d_eng = detect_engulfing(highs, lows, opens, closes)
    if d_pin:
        trig, trig_dir = "pinbar", d_pin.direction
    elif d_eng:
        trig, trig_dir = "engulfing", d_eng.direction
    K, D_vals, RS_vals = _stoch_rsi_kd(closes, 14, 14, 3, 3)
    # Stoch signal
    k1 = K[-1] if K else 50.0
    d1 = D_vals[-1] if D_vals else 50.0
    k0 = K[-2] if len(K) > 1 else k1
    d0 = D_vals[-2] if len(D_vals) > 1 else d1
    stoch_long = k0 < 20 and k1 > 20 and k1 > d1 and k0 <= d0
    stoch_short = k0 > 80 and k1 < 80 and k1 < d1 and k0 >= d0
    # Session alignment using timestamp if available
    sess_ok = False
    try:
        t = str(times[-1])
        import re as _re
        m = _re.search(r'T(\d{2}):', t) or _re.search(r' (\d{2}):', t)
        hour = int(m.group(1)) if m else 12
        sess_ok = (7 <= hour <= 16) or (12 <= hour <= 21)
    except Exception:
        sess_ok = False
    # Determine direction
    direction = trig_dir or sweep or trend_dir
    if not direction:
        return None
    # Confluence scoring
    points = 0
    # Trend agreement and EMA stack
    if trend_dir == direction:
        if direction == "long" and closes[-1] > e20[-1] > e50[-1]:
            points += 2
        elif direction == "short" and closes[-1] < e20[-1] < e50[-1]:
            points += 2
    # Key level/zone
    if (direction == "long" and at_sr_long) or (direction == "short" and at_sr_short):
        points += 1
    # Location overlap: fib pocket or round number
    if _fib_overlap(closes[-1], highs, lows, 50, 0.001) or _round_level_proximity(closes[-1], pair):
        points += 1
    # Liquidity sweep
    if sweep == direction:
        points += 1
    # FVG context
    if has_fvg and (fvg_dir in (direction, None)):
        points += 1
    # Trigger candle
    if trig_dir == direction:
        points += 1
    # Session
    if sess_ok:
        points += 1
    # Volatility: ATR ratio
    if (a[-1] / max(1e-9, closes[-1])) > (0.0008 if not pair.endswith("JPY") and pair != "XAUUSD" else (0.0003 if pair.endswith("JPY") else 0.0005)):
        points += 1
    # Stoch RSI state change
    if (direction == "long" and stoch_long) or (direction == "short" and stoch_short):
        points += 1
    # Reward:Risk ≥ 1.5 vs nearest swing
    entry = closes[-1]
    stop = lows[-1] - a[-1] * 0.25 if direction == "long" else highs[-1] + a[-1] * 0.25
    risk = abs(entry - stop)
    target = hh if direction == "long" else ll
    if (abs(target - entry) / (risk + 1e-9)) >= 1.5:
        points += 1
    if points < 6:
        return None
    score = min(100, 62 + points * 3)
    return Detected("AstraFX Strategy #1", direction, entry, stop, score, tags=["Proprietary #1"])


# --- User Defined Confluence (multi-select, ALL conditions must hold) ---
def det_user_confluence(pair: str, tf: str, highs, lows, opens, closes, times, keys: List[str]) -> Optional[Detected]:
    if not keys:
        return None
    a = atr(highs, lows, closes, 14)
    e20 = ema(closes, 20); e50 = ema(closes, 50)
    slope50 = e50[-1] - e50[-5] if len(e50) >= 5 else 0.0
    hh, ll = _swing_levels(highs, lows, 50)
    tol = max(a[-1]*0.25, closes[-1]*0.0005)
    at_sr_long  = _near(closes[-1], ll, tol)
    at_sr_short = _near(closes[-1], hh, tol)
    sweep = _liquidity_sweep(highs, lows, closes, 10)
    has_fvg, fvg_dir = _fvg_context(highs, lows)
    d_pin = detect_pinbar(highs, lows, opens, closes)
    d_eng = detect_engulfing(highs, lows, opens, closes)

    # Prior day levels/pivots
    def _parse_hours(ts):
        try:
            import datetime as _dt
            if 'T' in ts:
                return _dt.datetime.fromisoformat(ts.replace('Z','+00:00'))
        except Exception:
            return None
        return None
    parsed = [_parse_hours(str(t)) for t in times]
    day_idx = {}
    for i, t in enumerate(parsed):
        if t: day_idx.setdefault(t.date(), []).append(i)
    prev_day_levels = None
    if parsed and parsed[-1]:
        today = parsed[-1].date()
        days = sorted([d for d in day_idx.keys() if d < today])
        if days:
            pd = days[-1]
            idxs = day_idx[pd]
            pd_high = max(highs[min(idxs):max(idxs)+1]); pd_low = min(lows[min(idxs):max(idxs)+1]); pd_close = closes[max(idxs)]
            P = (pd_high + pd_low + pd_close) / 3.0
            R1 = 2*P - pd_low; S1 = 2*P - pd_high
            prev_day_levels = (pd_high, pd_low, P, R1, S1)

    # Momentum via Stoch RSI exits
    K, D_vals, RS = _stoch_rsi_kd(closes, 14, 14, 3, 3)
    k0, d0 = (K[-2] if len(K)>1 else 50.0), (D_vals[-2] if len(D_vals)>1 else 50.0)
    k1, d1 = K[-1], D_vals[-1]
    mom_long  = k0<20 and k1>20 and k1>d1 and k0<=d0
    mom_short = k0>80 and k1<80 and k1<d1 and k0>=d0

    checks = {}
    checks['trend_structure_long']  = slope50 > 0
    checks['trend_structure_short'] = slope50 < 0
    e200 = ema(closes, 200) if len(closes) >= 200 else e50
    checks['d1_h4_bias_long']  = e50[-1] > e200[-1]
    checks['d1_h4_bias_short'] = e50[-1] < e200[-1]
    checks['ema2050_long']  = (e20[-1] > e50[-1]) and (slope50 > 0)
    checks['ema2050_short'] = (e20[-1] < e50[-1]) and (slope50 < 0)
    checks['location_any'] = _fib_overlap(closes[-1], highs, lows, 50, 0.001)
    checks['htf_sr_long']  = at_sr_long
    checks['htf_sr_short'] = at_sr_short
    checks['sd_long']  = at_sr_long
    checks['sd_short'] = at_sr_short
    checks['fib_overlap'] = _fib_overlap(closes[-1], highs, lows, 50, 0.001)
    checks['round_level'] = _round_level_proximity(closes[-1], pair)
    if prev_day_levels:
        pdh, pdl, P, R1, S1 = prev_day_levels
        checks['pdh_or_pdl'] = _near(closes[-1], pdh, tol) or _near(closes[-1], pdl, tol)
        checks['classic_pivots'] = _near(closes[-1], P, tol) or _near(closes[-1], R1, tol) or _near(closes[-1], S1, tol)
    else:
        checks['pdh_or_pdl'] = False
        checks['classic_pivots'] = False
    eq_highs = abs(highs[-1] - highs[-2]) <= tol if len(highs) >= 2 else False
    eq_lows  = abs(lows[-1] - lows[-2]) <= tol if len(lows) >= 2 else False
    checks['liq_imbalance'] = eq_highs or eq_lows or has_fvg
    checks['liq_sweep_long']  = sweep == 'long'
    checks['liq_sweep_short'] = sweep == 'short'
    checks['fvg_any'] = has_fvg
    checks['momentum_long']  = mom_long
    checks['momentum_short'] = mom_short
    checks['pinbar_long']  = bool(d_pin and d_pin.direction == 'long')
    checks['pinbar_short'] = bool(d_pin and d_pin.direction == 'short')
    checks['engulf_long']  = bool(d_eng and d_eng.direction == 'long')
    checks['engulf_short'] = bool(d_eng and d_eng.direction == 'short')
    checks['hammer']        = checks['pinbar_long']
    checks['shooting_star'] = checks['pinbar_short']
    ms = False; es = False
    if len(closes) >= 3:
        o1,o2,o3 = opens[-3], opens[-2], opens[-1]
        c1,c2,c3 = closes[-3], closes[-2], closes[-1]
        ms = (c1 < o1) and (abs(c2-o2) <= (abs(c1-o1)*0.4)) and (c3 > o3) and (c3 >= (o1 + (c1 - o1)*0.5))
        es = (c1 > o1) and (abs(c2-o2) <= (abs(c1-o1)*0.4)) and (c3 < o3) and (c3 <= (o1 + (c1 - o1)*0.5))
    checks['morning_star'] = ms
    checks['evening_star'] = es
    checks['hanging_man'] = checks['pinbar_short'] and slope50 > 0

    def key_to_check(k: str) -> bool:
        mapping = {
            "Trend and structure":"trend_structure_long",
            "D1/H4 bias":"d1_h4_bias_long",
            "20/50 EMA slope/stack":"ema2050_long",
            "Location":"location_any",
            "HTF S/R":"htf_sr_long",
            "Supply - Demand zones":"sd_long",
            "FIB overlap":"fib_overlap",
            "Round levels":"round_level",
            "Prior day high or low":"pdh_or_pdl",
            "Classic pivots":"classic_pivots",
            "Liquidity & Imbalance":"liq_imbalance",
            "Liquidity sweep":"liq_sweep_long",
            "Fair Value Gap":"fvg_any",
            "Momentum Confirmation":"momentum_long",
            "Pinbar":"pinbar_long",
            "Engulfing":"engulf_long",
            "Hammer / Shooting star":"hammer",
            "Morning star / Evening star":"morning_star",
            "Hanging man":"hanging_man",
        }
        return bool(checks.get(mapping.get(k, ""), False))

    if not all(key_to_check(k) for k in keys):
        return None

    pts = len(keys)
    score = min(95, 78 + 2*pts)
    direction = "long" if (checks['trend_structure_long'] or checks['pinbar_long'] or checks['engulf_long'] or checks['momentum_long']) else "short" if (checks['trend_structure_short'] or checks['pinbar_short'] or checks['engulf_short'] or checks['momentum_short']) else "long"
    entry = closes[-1]
    stop  = lows[-1] if direction == 'long' else highs[-1]
    return Detected("User Defined Confluence", direction, entry, stop, score, tags=["User Confluence"])
def strategy_label(name: str) -> str:
    order = ["M5","M15","M30","H1","H4","D1"]
    tf_list = sorted(list(STRATEGY_TF_MAP.get(name, set())), key=lambda x: order.index(x))
    pretty = {
        "donchian":"Donchian 20 Break",
        "bbsqueeze":"Bollinger Squeeze Breakout",
        "ema_pullback":"EMA(9/21) Pullback",
        "ichimoku":"Ichimoku Trend-Follow",
        "orb_london":"Opening-Range Breakout (London)",
        "rsi2":"RSI-2 Mean Reversion",
    }[name]
    return f"{pretty} (" + ", ".join(tf_list) + ")"

def det_bbsqueeze_strategy(highs, lows, closes) -> Optional[Detected]:
    if len(closes) < 40:
        return None
    mid = sma(closes, 20); sd = stdev(closes, 20)
    upper = [m + 2*s for m, s in zip(mid, sd)]
    lower = [m - 2*s for m, s in zip(mid, sd)]
    bandwidth = [(u - l) / (m if m != 0 else 1e-9) for u, l, m in zip(upper, lower, mid)]
    if bandwidth[-1] > percentile(bandwidth[-40:], 20):
        return None
    c = closes[-1]
    if c > upper[-1]:
        direction = "long"; entry = c; stop = mid[-1]
    elif c < lower[-1]:
        direction = "short"; entry = c; stop = mid[-1]
    else:
        return None
    atr14 = atr(highs, lows, closes, 14)
    dist = abs(c - (upper[-1] if direction == "long" else lower[-1]))
    vol = atr14[-1] if atr14[-1] else 1e-9
    score = min(100, 68 + (dist / vol) * 12)
    return Detected(strategy_label("bbsqueeze"), direction, entry, stop, score, tags=["bbsqueeze"])

def det_donchian_strategy(highs, lows, closes) -> Optional[Detected]:
    d = detect_donchian20(highs, lows, closes)
    if not d: return None
    d.name = strategy_label("donchian")
    return d

def det_ema_pullback_strategy(highs, lows, closes) -> Optional[Detected]:
    if len(closes) < 60:
        return None
    e9 = ema(closes, 9); e21 = ema(closes, 21)
    def crossed_up(ea, eb, look=10):
        for i in range(-look, 0):
            if ea[i-1] <= eb[i-1] and ea[i] > eb[i]:
                return True
        return False
    def crossed_down(ea, eb, look=10):
        for i in range(-look, 0):
            if ea[i-1] >= eb[i-1] and ea[i] < eb[i]:
                return True
        return False
    c = closes[-1]
    if e9[-1] > e21[-1] and (crossed_up(e9, e21) or c > e9[-1]):
        direction = "long"; entry = c; stop = lowest(lows, 10)
    elif e9[-1] < e21[-1] and (crossed_down(e9, e21) or c < e9[-1]):
        direction = "short"; entry = c; stop = highest(highs, 10)
    else:
        return None
    slope = (e21[-1] - e21[-5]) if len(e21) > 5 else 0.0
    score = min(100, 66 + abs(slope) * 2000.0 / (abs(c) + 1e-9))
    return Detected(strategy_label("ema_pullback"), direction, entry, stop, score, tags=["ema_pullback"])

def det_ichimoku_strategy(highs, lows, closes) -> Optional[Detected]:
    if len(closes) < 60:
        return None
    def hl_avg(H, L, n, i):
        start = max(0, i - n + 1)
        return (max(H[start:i+1]) + min(L[start:i+1])) / 2.0
    tenkan = [hl_avg(highs, lows, 9, i) for i in range(len(closes))]
    kijun  = [hl_avg(highs, lows, 26, i) for i in range(len(closes))]
    spanA  = [(t + k) / 2.0 for t, k in zip(tenkan, kijun)]
    spanB  = [hl_avg(highs, lows, 52, i) for i in range(len(closes))]
    cloud_top = [max(a, b) for a, b in zip(spanA, spanB)]
    cloud_bot = [min(a, b) for a, b in zip(spanA, spanB)]
    c = closes[-1]
    if c > cloud_top[-1] and tenkan[-1] > kijun[-1]:
        direction = "long"; entry = c; stop = min(kijun[-1], cloud_bot[-1])
    elif c < cloud_bot[-1] and tenkan[-1] < kijun[-1]:
        direction = "short"; entry = c; stop = max(kijun[-1], cloud_top[-1])
    else:
        return None
    score = 70
    return Detected(strategy_label("ichimoku"), direction, entry, stop, score, tags=["ichimoku"])

def det_orb_london_strategy(times: List[str], highs, lows, closes, tf: str) -> Optional[Detected]:
    if tf not in {"M5","M15","M30"}:
        return None
    import datetime as dt
    try:
        parsed = [dt.datetime.fromisoformat(t.replace("Z","+00:00")) for t in times]
    except Exception:
        return None
    today = parsed[-1].date()
    idx = [i for i,x in enumerate(parsed) if x.date() == today]
    if not idx: return None
    first = idx[0]
    window_bars = {"M5":6, "M15":2, "M30":1}[tf]
    end = min(len(highs), first + window_bars)
    or_high = max(highs[first:end]); or_low = min(lows[first:end])
    c = closes[-1]
    if c > or_high:
        direction = "long"; entry = c; stop = or_low
    elif c < or_low:
        direction = "short"; entry = c; stop = or_high
    else:
        return None
    score = 65
    return Detected(strategy_label("orb_london"), direction, entry, stop, score, tags=["orb_london"])

def det_rsi2_strategy(highs, lows, closes) -> Optional[Detected]:
    if len(closes) < 220:
        return None
    s200 = sma(closes, 200); r2 = rsi(closes, 2); c = closes[-1]
    if c > s200[-1] and r2[-1] < 10:
        direction = "long"; entry = c; stop = lowest(lows, 10)
    elif c < s200[-1] and r2[-1] > 90:
        direction = "short"; entry = c; stop = highest(highs, 10)
    else:
        return None
    score = 64 + (10 if abs(c - s200[-1]) / (abs(c) + 1e-9) > 0.001 else 0)
    return Detected(strategy_label("rsi2"), direction, entry, stop, score, tags=["rsi2"])

def gather_candidates(pair: str, tf: str, series: List[Dict[str, float]], which_pattern: str, which_strategy: str) -> Dict[str, List[Detected]]:
    opens  = [x["open"] for x in series]
    highs  = [x["high"] for x in series]
    lows   = [x["low"] for x in series]
    closes = [x["close"] for x in series]
    times  = [x.get("time","") for x in series]

    ws = (which_strategy or "all").lower()
    wp = (which_pattern or "all").lower()

    def tf_allowed(name_key: str) -> bool:
        return tf in STRATEGY_TF_MAP.get(name_key, set())

    strat: List[Detected] = []
    if ("donchian" in (ws, "all")) and tf_allowed("donchian"):
        d = det_donchian_strategy(highs, lows, closes);      strat += [d] if d else []
    if ("bbsqueeze" in (ws, "all")) and tf_allowed("bbsqueeze"):
        d = det_bbsqueeze_strategy(highs, lows, closes);     strat += [d] if d else []
    if ("ema_pullback" in (ws, "all")) and tf_allowed("ema_pullback"):
        d = det_ema_pullback_strategy(highs, lows, closes);  strat += [d] if d else []
    if ("ichimoku" in (ws, "all")) and tf_allowed("ichimoku"):
        d = det_ichimoku_strategy(highs, lows, closes);      strat += [d] if d else []
    if ("orb_london" in (ws, "all")) and tf_allowed("orb_london"):
        d = det_orb_london_strategy(times, highs, lows, closes, tf); strat += [d] if d else []
    if ("rsi2" in (ws, "all")) and tf_allowed("rsi2"):
        d = det_rsi2_strategy(highs, lows, closes);          strat += [d] if d else []

    pat: List[Detected] = []
    if wp in ("all","pinbar"):      (lambda d=detect_pinbar(highs, lows, opens, closes): pat.append(d) if d else None)()
    if wp in ("all","engulfing"):   (lambda d=detect_engulfing(highs, lows, opens, closes): pat.append(d) if d else None)()
    if wp in ("all","insidebar"):   (lambda d=detect_insidebar(highs, lows, opens, closes): pat.append(d) if d else None)()
    if wp in ("all","nr7","NR7"):   (lambda d=detect_nr7(highs, lows, opens, closes): pat.append(d) if d else None)()
    if wp in ("all","donchian20"):  (lambda d=detect_donchian20(highs, lows, closes): pat.append(d) if d else None)()
    if wp in ("all","rsi"):         (lambda d=detect_rsi_pullback(highs, lows, closes): pat.append(d) if d else None)()

    return {"strategy": strat, "pattern": pat}

def best_or_union(pair: str, tf: str, series: List[Dict[str, float]], which_pattern: str, which_strategy: str, min_score: int) -> List[Dict[str, Any]]:
    c = gather_candidates(pair, tf, series, which_pattern, which_strategy)
    ws = (which_strategy or "all").lower()
    wp = (which_pattern or "all").lower()

    if ws != "all" and wp != "all":
        out: List[Detected] = []
        if c["strategy"]:
            out.append(max(c["strategy"], key=lambda d: d.score))
        if c["pattern"]:
            out.append(max(c["pattern"], key=lambda d: d.score))
        uniq = {}
        for d in out:
            k = (d.name, d.direction, round(d.entry,6), round(d.stop,6))
            uniq[k] = d
        return [d.to_signal(pair, tf) for d in uniq.values() if d.score >= min_score]

    pool = c["strategy"] + c["pattern"]
    if not pool:
        return []
    best = max(pool, key=lambda d: d.score)
    return [best.to_signal(pair, tf)] if best.score >= min_score else []

# --------------------------------------------------------------------------------------
# API + static + aliases
# --------------------------------------------------------------------------------------
@app.get("/health")
def health():
    prov = os.getenv("DATA_PROVIDER", DATA_PROVIDER).upper()
    out = {"ok": True, "provider": prov, "version": "2.3"}
    if prov == "OANDA":
        out["env"] = os.getenv("OANDA_ENV", "practice")
    return out

@app.get("/api/ohlc")
def api_ohlc(pair: str = Query(...), tf: str = Query("H1"), limit: int = Query(200)):
    try:
        ohlc = fetch_ohlc_router(pair, tf, limit)
        return {"pair": pair, "tf": tf, "ohlc": ohlc}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scan")
def api_scan(
    pairs: str = Query("EURUSD,GBPUSD,USDJPY,XAUUSD"),
    tf: str = Query("H1"),
    min_score: int = Query(70),
    pattern: str = Query("all"),
    strategy: str = Query("all"),
    prop: str = Query("none"),
    udc: str = Query(""),
):
    """
    Scan for signals across one or more currency pairs and return high‑probability setups.

    Parameters:
      pairs: Comma‑separated list of currency symbols to scan (e.g. "EURUSD,GBPUSD").
      tf: Time frame (e.g. "M15", "H1").
      min_score: Minimum confidence score required to include a signal.
      pattern: Which candlestick pattern to evaluate ("pinbar", "engulfing", etc. or "all").
      strategy: Which built‑in strategy to evaluate (e.g. "donchian" or "all").
      prop: Optional proprietary strategy key (e.g. "astra1"); "none" disables proprietary checks.

    Returns:
      JSON with the time frame and a list of signal objects.
    """
    results: List[Dict[str, Any]] = []
    # Normalize proprietary key once for comparisons
    prop_key = (prop or "none").lower()
    for p in [x.strip() for x in pairs.split(",") if x.strip()]:
        try:
            series = fetch_ohlc_router(p, tf, 220)
            # Base strategy/pattern detections
            base = best_or_union(p, tf, series, which_pattern=pattern, which_strategy=strategy, min_score=min_score)
            if base:
                results.extend(base)
            # Proprietary strategy detection (append separately)
            if prop_key != "none":
                # Extract OHLC arrays for proprietary detection
                opens  = [x["open"] for x in series]
                highs  = [x["high"] for x in series]
                lows   = [x["low"] for x in series]
                closes = [x["close"] for x in series]
                times  = [x.get("time", "") for x in series]
                # Currently only one proprietary strategy is implemented: astra1
                if prop_key == "astra1":
                    d = det_astra1_strategy(p, tf, highs, lows, opens, closes, times)
                    if d and d.score >= min_score:
                        results.append(d.to_signal(p, tf))
        except Exception as e:
            print("scan error", p, e)
            continue
    return {"tf": tf, "signals": results}

# Serve the frontend from /

@app.post("/api/signup")
def api_signup_password(payload: Dict[str, Any] = Body(...)):
    email = (payload.get("email") or "").strip().lower()
    password = (payload.get("password") or "").strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email required")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    users = _load_users()
    if email in users:
        raise HTTPException(status_code=409, detail="Email already exists. Please log in.")
    salt_hex, hash_hex = _hash_pw(password)
    _save_user(email, salt_hex, hash_hex)
    gh = _commit_users_csv()
    return {"ok": True, "email": email, "committed": gh}

@app.post("/api/login")
def api_login(payload: Dict[str, Any] = Body(...)):
    email = (payload.get("email") or "").strip().lower()
    password = (payload.get("password") or "").strip()
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    users = _load_users()
    rec = users.get(email)
    if not rec or not _verify_pw(password, rec[0], rec[1]):
        raise HTTPException(status_code=403, detail="Invalid email or password")
    resp = Response(content='{"ok":true}', media_type="application/json")
    _set_session_cookie(resp, email)
    return resp

@app.get("/app", include_in_schema=False)
@app.get("/app/", include_in_schema=False)
def app_gate(request: Request):
    token = request.cookies.get(COOKIE_NAME, "")
    email = _verify_token(token) if token else ""
    if not email or email.lower() not in _load_users():
        return RedirectResponse(url="/?denied=1", status_code=302)
    return FileResponse("public/app/index.html")

app.mount("/", StaticFiles(directory="public", html=True), name="static")

# Legacy aliases
@app.get("/scan")
def _alias_scan(request: Request):
    q = ("?" + request.url.query) if request.url.query else ""
    return RedirectResponse(url="/api/scan" + q)

@app.get("/ohlc")
def _alias_ohlc(request: Request):
    q = ("?" + request.url.query) if request.url.query else ""
    return RedirectResponse(url="/api/ohlc" + q)

# Local run (Render uses gunicorn with `server:app`)
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
# === END ASTRAFX SERVER.PY ===
