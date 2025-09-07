# AstraFX Backend v2.4 — Email+Password auth + gate
import os, base64, hmac, time, datetime, hashlib, csv
from typing import List, Dict, Any, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query, Body, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse

app = FastAPI(title="AstraFX Backend v2.4", version="2.4")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    return {"M5":"M5","M15":"M15","M30":"M30","H1":"H1","H4":"H4","D1":"D","D":"D"}.get(tf, "H1")

def fetch_oanda_candles(pair: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    instrument = map_pair_to_oanda(pair); gran = map_tf_to_oanda(tf)
    url = f"{oanda_base_url()}/v3/instruments/{instrument}/candles"
    params = {"granularity": gran, "count": max(1, min(int(limit), 500)), "price": "M"}
    r = requests.get(url, headers=oanda_headers(), params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OANDA candles error: {r.status_code} {r.text[:180]}")
    data = r.json(); out = []
    for c in data.get("candles", []):
        if not c.get("complete", False): continue
        mid = c.get("mid", {})
        out.append({"time": c.get("time"), "open": float(mid.get("o")), "high": float(mid.get("h")),
                    "low": float(mid.get("l")), "close": float(mid.get("c"))})
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
    return [{"time": v.get("datetime"), "open": float(v["open"]), "high": float(v["high"]),
             "low": float(v["low"]), "close": float(v["close"])} for v in vals]

def fetch_alphavantage_ohlc(pair: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    key = os.getenv("ALPHAVANTAGE_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="Missing ALPHAVANTAGE_KEY")
    func = {"H1":"FX_INTRADAY","H4":"FX_INTRADAY","D1":"FX_DAILY","M5":"FX_INTRADAY","M15":"FX_INTRADAY","M30":"FX_INTRADAY"}.get(tf.upper(),"FX_INTRADAY")
    interval = {"M5":"5min","M15":"15min","M30":"30min","H1":"60min","H4":"240min"}.get(tf.upper(),"60min")
    params = {"function": func, "from_symbol": pair[0:3], "to_symbol": pair[3:6], "apikey": key}
    if func == "FX_INTRADAY": params["interval"] = interval
    url = "https://www.alphavantage.co/query"
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"AlphaVantage error: {r.status_code} {r.text[:180]}")
    js = r.json(); series_key = next((k for k in js.keys() if "Time Series" in k), None)
    if not series_key: raise HTTPException(status_code=502, detail=f"AlphaVantage no series: {list(js.keys())}")
    raw = js[series_key]; rows = sorted(raw.items())
    return [{"time": t, "open": float(v["1. open"]), "high": float(v["2. high"]), "low": float(v["3. low"]), "close": float(v["4. close"])}
            for t, v in rows[-limit:]]

def fetch_ohlc_router(pair: str, tf: str, limit: int = 200) -> List[Dict[str, Any]]:
    prov = os.getenv("DATA_PROVIDER", DATA_PROVIDER).upper()
    if prov == "OANDA": return fetch_oanda_candles(pair, tf, limit)
    if prov == "TWELVEDATA": return fetch_twelvedata_ohlc(pair, tf, limit)
    if prov == "ALPHAVANTAGE": return fetch_alphavantage_ohlc(pair, tf, limit)
    seed = 1.2345 if pair != "XAUUSD" else 2400.0
    out = []; val = seed
    for i in range(limit):
        o = val * (1 + (0.0005 - 0.001 * (i % 2)))
        h = o * (1 + 0.001); l = o * (1 - 0.001); c = o * (1 + (0.0002 - 0.0004 * (i % 2)))
        out.append({"time": i, "open": o, "high": h, "low": l, "close": c})
        val = c * (1 + (i - limit/2) * 1e-4)
    return out

def ema(values: List[float], length: int) -> List[float]:
    if length <= 1 or not values: return values[:]
    k = 2.0 / (length + 1); out = []; prev = values[0]
    for v in values:
        prev = v * k + prev * (1 - k); out.append(prev)
    return out

def sma(values: List[float], length: int) -> List[float]:
    out = []; s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= length: s -= values[i - length]
        out.append(s / length if i >= length - 1 else values[i])
    return out

def stdev(values: List[float], length: int) -> List[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - length + 1); window = values[start:i+1]
        if len(window) < 2: out.append(0.0)
        else:
            m = sum(window) / len(window)
            var = sum((x - m) ** 2 for x in window) / (len(window) - 1)
            out.append(var ** 0.5)
    return out

def rsi(closes: List[float], length: int = 14) -> List[float]:
    if len(closes) < length + 1: return [50.0] * len(closes)
    gains = [0.0]; losses = [0.0]
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        gains.append(max(ch, 0.0)); losses.append(max(-ch, 0.0))
    avg_gain = sum(gains[1:length+1]) / length; avg_loss = sum(losses[1:length+1]) / length
    rsis = [50.0] * len(closes)
    for i in range(length+1, len(closes)):
        avg_gain = (avg_gain * (length - 1) + gains[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i]) / length
        rs = (avg_gain / avg_loss) if avg_loss != 0 else float("inf")
        rsis[i] = 100 - (100 / (1 + rs))
    return rsis

def atr(highs: List[float], lows: List[float], closes: List[float], length: int = 14) -> List[float]:
    if len(closes) < 2: return [0.0] * len(closes)
    trs = [0.0]
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    out = [0.0] * len(trs)
    if len(trs) <= length: return out
    out[length] = sum(trs[1:length+1]) / length
    for i in range(length+1, len(trs)):
        out[i] = (out[i-1] * (length - 1) + trs[i]) / length
    return out

def highest(values: List[float], lookback: int, end: Optional[int] = None) -> float:
    end = len(values) if end is None else end; start = max(0, end - lookback)
    return max(values[start:end])

def lowest(values: List[float], lookback: int, end: Optional[int] = None) -> float:
    end = len(values) if end is None else end; start = max(0, end - lookback)
    return min(values[start:end])

def percentile(values: List[float], p: float) -> float:
    if not values: return 0.0
    xs = sorted(values)
    k = int(max(0, min(len(xs) - 1, round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]

USERS_LOCAL_PATH = os.environ.get("USERS_LOCAL_PATH", "data/users.csv")
os.makedirs(os.path.dirname(USERS_LOCAL_PATH) or ".", exist_ok=True)

def _github_commit(path: str, content_bytes: bytes, message: str):
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPO", "").strip()
    branch = os.environ.get("GITHUB_BRANCH", "main").strip()
    if not token or not repo: return {"skipped": True, "reason": "missing token or repo"}
    import requests
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    params = {"ref": branch}
    r = requests.get(api_url, headers=headers, params=params, timeout=20)
    sha = r.json().get("sha") if r.status_code == 200 else None
    put_body = {"message": message, "content": base64.b64encode(content_bytes).decode("utf-8"), "branch": branch}
    if sha: put_body["sha"] = sha
    r2 = requests.put(api_url, headers=headers, json=put_body, timeout=20)
    return {"status": r2.status_code, "text": r2.text[:200]}

def _hash_pw(password: str):
    salt = os.urandom(16)
    import hashlib
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return salt.hex(), dk.hex()

def _verify_pw(password: str, salt_hex: str, hash_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    import hashlib, hmac
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return hmac.compare_digest(dk.hex(), hash_hex)

def _load_users() -> Dict[str, tuple]:
    users = {}
    try:
        import csv
        with open(USERS_LOCAL_PATH, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                em = row.get("email","").strip().lower()
                if em:
                    users[em] = (row.get("salt",""), row.get("hash",""))
    except FileNotFoundError:
        pass
    return users

def _save_user(email: str, salt_hex: str, hash_hex: str):
    header_needed = not os.path.exists(USERS_LOCAL_PATH)
    with open(USERS_LOCAL_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed: w.writerow(["timestamp","email","salt","hash"])
        ts = datetime.datetime.utcnow().isoformat()+"Z"
        w.writerow([ts, email, salt_hex, hash_hex])

def _commit_users_csv():
    try:
        with open(USERS_LOCAL_PATH, "rb") as f: data_bytes = f.read()
        return _github_commit("data/users.csv", data_bytes, "Update users.csv")
    except Exception as e:
        return {"error": str(e)}

STRATEGY_TF_MAP = {
    "donchian": {"M15","M30","H1","H4","D1"},
    "bbsqueeze": {"M15","M30","H1"},
    "ema_pullback": {"M15","M30","H1","H4"},
    "ichimoku": {"M30","H1","H4","D1"},
    "orb_london": {"M5","M15","M30"},
    "rsi2": {"M15","H1"},
}
def strategy_label(name: str) -> str:
    order = ["M5","M15","M30","H1","H4","D1"]
    tf_list = sorted(list(STRATEGY_TF_MAP.get(name, set())), key=lambda x: order.index(x))
    pretty = {
        "donchian":"Donchian 20 Break","bbsqueeze":"Bollinger Squeeze Breakout","ema_pullback":"EMA(9/21) Pullback",
        "ichimoku":"Ichimoku Trend-Follow","orb_london":"Opening-Range Breakout (London)","rsi2":"RSI-2 Mean Reversion",
    }[name]
    return f"{pretty} (" + ", ".join(tf_list) + ")"

def det_bbsqueeze_strategy(highs, lows, closes): return None  # omitted here for brevity in this minimal server copy
def det_donchian_strategy(highs, lows, closes): return None
def det_ema_pullback_strategy(highs, lows, closes): return None
def det_ichimoku_strategy(highs, lows, closes): return None
def det_orb_london_strategy(times, highs, lows, closes, tf): return None
def det_rsi2_strategy(highs, lows, closes): return None

def gather_candidates(pair, tf, series, which_pattern, which_strategy): return {"strategy": [], "pattern": []}
def best_or_union(pair, tf, series, which_pattern, which_strategy, min_score): return []

@app.get("/health")
def health():
    prov = os.getenv("DATA_PROVIDER", DATA_PROVIDER).upper()
    out = {"ok": True, "provider": prov, "version": "2.4"}
    if prov == "OANDA": out["env"] = os.getenv("OANDA_ENV", "practice")
    return out

@app.get("/api/ohlc")
def api_ohlc(pair: str = Query(...), tf: str = Query("H1"), limit: int = Query(200)):
    try:
        ohlc = fetch_ohlc_router(pair, tf, limit)
        return {"pair": pair, "tf": tf, "ohlc": ohlc}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/scan")
def api_scan(pairs: str = Query("EURUSD,GBPUSD,USDJPY,XAUUSD"), tf: str = Query("H1"),
             min_score: int = Query(70), pattern: str = Query("all"), strategy: str = Query("all")):
    return {"tf": tf, "signals": []}

SESSION_SECRET = os.environ.get("SESSION_SECRET", "change-me-please").encode("utf-8")
COOKIE_NAME    = os.environ.get("COOKIE_NAME", "astrafx_session")
COOKIE_SECURE  = os.environ.get("COOKIE_SECURE", "1") != "0"

def _make_token(email: str, days: int = 7) -> str:
    exp = int(time.time()) + days * 86400
    payload = f"{email}|{exp}"
    sig = hmac.new(SESSION_SECRET, payload.encode("utf-8"), digestmod="sha256").hexdigest()
    raw = f"{payload}|{sig}".encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("utf-8")

def _verify_token(token: str) -> str:
    try:
        raw = base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
        email, exp, sig = raw.split("|")
        if int(exp) < int(time.time()): return ""
        exp_sig = hmac.new(SESSION_SECRET, f"{email}|{exp}".encode("utf-8"), digestmod="sha256").hexdigest()
        if not hmac.compare_digest(sig, exp_sig): return ""
        return email
    except Exception:
        return ""

def _set_session_cookie(response: Response, email: str):
    token = _make_token(email)
    response.set_cookie(key=COOKIE_NAME, value=token, max_age=7*86400,
                        httponly=True, samesite="Lax", secure=COOKIE_SECURE, path="/")

def _require_session(request: Request) -> str:
    token = request.cookies.get(COOKIE_NAME, "")
    if not token: return ""
    email = _verify_token(token)
    if not email: return ""
    users = _load_users()
    return email if email.lower() in users else ""

@app.post("/api/signup")
def api_signup(payload: Dict[str, Any] = Body(...)):
    email = (payload.get("email") or "").strip().lower()
    password = (payload.get("password") or "").strip()
    if not email or "@" not in email: raise HTTPException(status_code=400, detail="Valid email required")
    if len(password) < 6: raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    users = _load_users()
    if email in users: raise HTTPException(status_code=409, detail="Email already exists. Please log in.")
    salt_hex, hash_hex = _hash_pw(password)
    _save_user(email, salt_hex, hash_hex)
    gh = _commit_users_csv()
    return {"ok": True, "email": email, "committed": gh}

@app.post("/api/login")
def api_login(payload: Dict[str, Any] = Body(...), response: Response = None):
    email = (payload.get("email") or "").strip().lower()
    password = (payload.get("password") or "").strip()
    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")
    users = _load_users()
    rec = users.get(email)
    if not rec or not _verify_pw(password, rec[0], rec[1]):
        raise HTTPException(status_code=403, detail="Invalid email or password")
    if response is None:
        from fastapi import Response as _Resp
        response = _Resp()
    _set_session_cookie(response, email)
    return {"ok": True, "email": email}

@app.get("/app", include_in_schema=False)
@app.get("/app/", include_in_schema=False)
def app_gate(request: Request):
    email = _require_session(request)
    if not email: return RedirectResponse(url="/?denied=1", status_code=302)
    return FileResponse("public/app/index.html")

app.mount("/", StaticFiles(directory="public", html=True), name="static")

@app.get("/scan")
def _alias_scan(request: Request):
    q = ("?" + request.url.query) if request.url.query else ""
    return RedirectResponse(url="/api/scan" + q)

@app.get("/ohlc")
def _alias_ohlc(request: Request):
    q = ("?" + request.url.query) if request.url.query else ""
    return RedirectResponse(url="/api/ohlc" + q)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
