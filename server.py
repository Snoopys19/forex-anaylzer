# AstraFX Backend v2.3 -- consistent TF gating + union when both selected
# Providers via env:
#   DATA_PROVIDER = OANDA | TWELVEDATA | ALPHAVANTAGE | MOCK
#   OANDA_TOKEN, OANDA_ENV (practice|live)

import os
from typing import List, Dict, Any, Optional
import requests
from fastapi import FastAPI, HTTPException, Query, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

app = FastAPI(title="AstraFX Backend v2.3", version="2.3")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === PATCH: In-memory TTL cache + request coalescing (+ optional stale-while-revalidate) ===
import time, asyncio
from dataclasses import dataclass
from typing import Dict as _Dict, List as _List, Optional as _Optional, Tuple as _Tuple, Set as _Set
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Tunables
CACHE_TTL_SEC: int = int(os.environ.get("CACHE_TTL_SEC", "45"))      # 0 disables cache
CACHE_SWR_SEC: int = int(os.environ.get("CACHE_SWR_SEC", "0"))       # e.g., "25" to enable SWR
CACHE_PATHS: _Set[str] = set(
    p.strip() for p in os.environ.get("CACHE_PATHS", "/api/ohlc,/api/scan").split(",") if p.strip()
)
CACHE_MAX_ENTRIES: _Optional[int] = int(os.environ.get("CACHE_MAX_ENTRIES", "0")) or None

@dataclass
class _CacheEntry:
    body: bytes
    status_code: int
    headers: _List[_Tuple[str, str]]   # excluding content-length
    media_type: _Optional[str]
    expires_at: float
    def is_fresh(self, now: float) -> bool: return now < self.expires_at

class _HTTPCache:
    def __init__(self) -> None:
        self.entries: _Dict[str, _CacheEntry] = {}
        self.locks: _Dict[str, asyncio.Lock] = {}
        self.hits = self.misses = self.stale_served = self.revalidations = self.coalesced_waiters = 0
    def get_lock(self, key: str) -> asyncio.Lock:
        lock = self.locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self.locks[key] = lock
        return lock
    def get(self, key: str) -> _Optional[_CacheEntry]: return self.entries.get(key)
    def set(self, key: str, entry: _CacheEntry) -> None:
        self.entries[key] = entry
        if CACHE_MAX_ENTRIES and len(self.entries) > CACHE_MAX_ENTRIES:
            now = time.time()
            expired = [k for k, v in self.entries.items() if not v.is_fresh(now)]
            for k in expired:
                if len(self.entries) <= CACHE_MAX_ENTRIES: break
                self.entries.pop(k, None)
            while len(self.entries) > CACHE_MAX_ENTRIES:
                self.entries.pop(next(iter(self.entries)))

_cache = _HTTPCache()

class _CoalescingTTLMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, ttl: int, swr: int, paths: _Set[str]) -> None:
        super().__init__(app); self.ttl = ttl; self.swr = swr; self.paths = paths
    @staticmethod
    def _fmt_utc(ts: float) -> str: return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
    @staticmethod
    def _key_from_request(request) -> str:
        items = sorted(request.query_params.multi_items())
        qs = "&".join(f"{k}={v}" for k, v in items)
        return f"{request.url.path}?{qs}"
    def _build_response(self, entry: _CacheEntry, cache_status: str, expires_at: float) -> Response:
        headers = dict(entry.headers); headers["X-Cache"] = cache_status; headers["X-Cache-Expires"] = self._fmt_utc(expires_at)
        return Response(content=entry.body, status_code=entry.status_code, headers=headers, media_type=entry.media_type)
    async def _refresh(self, key, request, call_next):
        lock = _cache.get_lock(key)
        async with lock:
            response = await call_next(request)
            body = b""
            async for chunk in response.body_iterator: body += chunk
            if response.status_code == 200:
                headers = [(k, v) for k, v in response.headers.items() if k.lower() != "content-length"]
                _cache.revalidations += 1
                _cache.set(key, _CacheEntry(body, response.status_code, headers, response.media_type, time.time() + self.ttl))
    async def dispatch(self, request, call_next):
        if request.method != "GET" or request.url.path not in self.paths or self.ttl <= 0:
            return await call_next(request)
        key = self._key_from_request(request); now = time.time(); entry = _cache.get(key)
        if entry and entry.is_fresh(now):
            _cache.hits += 1; return self._build_response(entry, "HIT", entry.expires_at)
        if entry and self.swr > 0 and now < entry.expires_at + self.sw_
            status_code=entry.status_code,
            headers=headers,
            media_type=entry.media_type,
        )

    async def _refresh(self, key, request, call_next):
        """Background revalidation (used for SWR)."""
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
        # Only GET requests for whitelisted routes are cached
        if request.method != "GET" or request.url.path not in self.paths or self.ttl <= 0:
            return await call_next(request)

        key = self._key_from_request(request)
        now = time.time()
        entry = _cache.get(key)

        # Fresh hit
        if entry and entry.is_fresh(now):
            _cache.hits += 1
            return self._build_response(entry, "HIT", entry.expires_at)

        # Expired but SWR window valid -> serve stale immediately and kick off revalidation
        if entry and self.swr > 0 and now < entry.expires_at + self.swr:
            _cache.stale_served += 1
            lock = _cache.get_lock(key)
            if not lock.locked():  # avoid stampede
                asyncio.create_task(self._refresh(key, request, call_next))
            return self._build_response(entry, "STALE", entry.expires_at)

        # MISS / hard expired: coalesce concurrent waiters
        lock = _cache.get_lock(key)
        if lock.locked():
            _cache.coalesced_waiters += 1
            async with lock:
                refreshed = _cache.get(key)
                if refreshed and refreshed.is_fresh(time.time()):
                    return self._build_response(refreshed, "COALESCED", refreshed.expires_at)

        async with lock:
            # Double-check after acquiring
            entry2 = _cache.get(key)
            if entry2 and entry2.is_fresh(time.time()):
                _cache.hits += 1
                return self._build_response(entry2, "HIT", entry2.expires_at)
        o = val * (1 + (0.0005 - 0.001 * (i % 2)))
        h = o * (1 + 0.001)
        l = o * (1 - 0.001)
        c = o * (1 + (0.0002 - 0.0004 * (i % 2)))
        out.append({"time": i, "open": o, "high": h, "low": l, "close": c})
        val = c * (1 + (i - limit/2) * 1e-4)
    return out

# --------------------------- TA helpers ---------------------------

def ema(values: List[float], length: int) -> List[float]:
    if length <= 1 or not values:
        return values[:]
    k = 2.0 / (length + 1)
    out = []
    prev = values[0]
    for v in values:
        prev = v * k + prev * (1 - k)
        out.append(prev)
    return out

def sma(values: List[float], length: int) -> List[float]:
    out = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= length:
            s -= values[i - length]
        out.append(s / length if i >= length - 1 else values[i])
    return out

def stdev(values: List[float], length: int) -> List[float]:
    out = []
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

# ------------------------ Detected wrapper ------------------------

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

# ------------------------ Patterns ------------------------

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

# ------------------------ Strategies ------------------------

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
        "donchian":"Donchian 20 Break",
        "bbsqueeze":"Bollinger Squeeze Breakout",
        "ema_pullback":"EMA(9/21) Pullback",
        "ichimoku":"Ichimoku Trend-Follow",
        "orb_london":"Opening-Range Breakout (London)",
        "rsi2":"RSI-2 Mean Reversion",
    }[name]
    return f"{pretty} (" + ", ".join(tf_list) + ")"

# individual strategy detectors (same as v2.2)
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

# ------------------------ Orchestration ------------------------

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
    # Always gate by TF, regardless of ws
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

    # If BOTH specific, return up to two (best strategy + best pattern)
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

    # Otherwise: single best overall (strategy + pattern pool)
    pool = c["strategy"] + c["pattern"]
    if not pool:
        return []
    best = max(pool, key=lambda d: d.score)
    return [best.to_signal(pair, tf)] if best.score >= min_score else []

# --------------------------- API ---------------------------

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
):
    results: List[Dict[str, Any]] = []
    for p in [x.strip() for x in pairs.split(",") if x.strip()]:
        try:
            series = fetch_ohlc_router(p, tf, 220)
            results.extend(best_or_union(p, tf, series, which_pattern=pattern, which_strategy=strategy, min_score=min_score))
        except Exception as e:
            print("scan error", p, e)
            continue
    return {"tf": tf, "signals": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5057)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi import Request

# Serve the frontend from /
app.mount("/", StaticFiles(directory="public", html=True), name="static")

# Optional aliases so old paths still work
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
    port = int(os.getenv("PORT", 10000))  # Render gives PORT, default to 10000 for local
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
