# server.py — Drop‑in with Strategy Autoload Endpoints
# - Reuses your existing app from oldserver.py if present
# - Adds /api/strategies, /api/tester_json, /api/scan_json
# - Looks for strategies/ next to this file
# - Uses your fetch_ohlc_router() if available; otherwise a minimal CSV fallback in /data

import os, json, importlib.util
from typing import List, Dict, Any
try:
    # Prefer your original app & fetch function if available
    from oldserver import app as app, fetch_ohlc_router as fetch_ohlc_router
except Exception:
    from fastapi import FastAPI
    app = FastAPI()
    def fetch_ohlc_router(pair: str, timeframe: str, limit: int = 1500):
        # Fallback loader: tries CSV at data/{pair}_{timeframe}.csv with columns open,high,low,close
        import csv
        path = os.path.join(os.path.dirname(__file__), "data", f"{pair}_{timeframe}.csv")
        bars = []
        if os.path.exists(path):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    bars.append({
                        "open": float(row["open"]), "high": float(row["high"]),
                        "low": float(row["low"]), "close": float(row["close"])
                    })
        return bars[-limit:] if bars else []

from fastapi import Body

# ---------- Strategy auto-discovery helpers ----------
STRATEGY_DIR = os.path.join(os.path.dirname(__file__), "strategies")
os.makedirs(STRATEGY_DIR, exist_ok=True)

def _list_strategy_files() -> List[str]:
    try:
        return sorted([fn for fn in os.listdir(STRATEGY_DIR)])
    except Exception as e:
        return [f"<error: {e}>"]

def _load_strategy_meta() -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    for fn in _list_strategy_files():
        if fn.endswith(".json"):
            fp = os.path.join(STRATEGY_DIR, fn)
            try:
                with open(fp, "r") as f:
                    meta = json.load(f)
                    if isinstance(meta, dict): metas.append(meta)
            except Exception as e:
                print("[AstraFX] Failed to load", fp, e)
    return metas

def _load_runner_py(filename: str):
    py_path = os.path.join(STRATEGY_DIR, filename)
    if not os.path.exists(py_path): return None
    spec = importlib.util.spec_from_file_location(filename.replace(".py",""), py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# ---------- Endpoints ----------
@app.get("/api/strategies")
def api_strategies():
    metas = _load_strategy_meta()
    return metas

@app.get("/api/strategies/debug")
def api_strategies_debug():
    return {"strategy_dir": STRATEGY_DIR, "files": _list_strategy_files()}

@app.post("/api/tester_json")
def api_tester_json(payload: Dict[str, Any] = Body(...)):
    sid = payload.get("strategy_id")
    if not sid: return {"error": "strategy_id required"}
    metas = {m.get("id"): m for m in _load_strategy_meta() if m.get("id")}
    if sid not in metas:
        return {"error": f"strategy_id not found: {sid}", "known": list(metas.keys()), "files": _list_strategy_files()}
    meta = metas[sid]
    params = dict(meta.get("indicators", {})); params.update(payload.get("params", {}))

    runner_name = "runner_divergence_srsi.py" if sid == "divergence_srsi_v1" else f"runner_{sid}.py"
    mod = _load_runner_py(runner_name)
    if not mod or not hasattr(mod, "run_backtest"):
        return {"error": f"Runner not found or invalid for {sid}", "tried": runner_name, "files": _list_strategy_files()}

    pair = payload.get("pair", "EURUSD")
    tf = payload.get("timeframe", meta.get("timeframes", {}).get("entry_tf", "M15"))
    bars = fetch_ohlc_router(pair, tf, limit=5000)
    if not bars: return {"error": "No OHLC returned", "pair": pair, "timeframe": tf}
    ohlc = {"open":[b["open"] for b in bars], "high":[b["high"] for b in bars], "low":[b["low"] for b in bars], "close":[b["close"] for b in bars]}
    res = mod.run_backtest(ohlc, params)
    return {"strategy": sid, "params": params, **res}

@app.post("/api/scan_json")
def api_scan_json(payload: Dict[str, Any] = Body(...)):
    sid = payload.get("strategy_id")
    if not sid: return {"error": "strategy_id required"}
    metas = {m.get("id"): m for m in _load_strategy_meta() if m.get("id")}
    if sid not in metas:
        return {"error": f"strategy_id not found: {sid}", "known": list(metas.keys()), "files": _list_strategy_files()}
    meta = metas[sid]
    params = dict(meta.get("indicators", {})); params.update(payload.get("params", {}))

    runner_name = "runner_divergence_srsi.py" if sid == "divergence_srsi_v1" else f"runner_{sid}.py"
    mod = _load_runner_py(runner_name)
    if not mod or not hasattr(mod, "run_scan"):
        return {"error": f"Runner not found or invalid for {sid}", "tried": runner_name, "files": _list_strategy_files()}

    pair = payload.get("pair", "EURUSD")
    tf = payload.get("timeframe", meta.get("timeframes", {}).get("entry_tf", "M15"))
    bars = fetch_ohlc_router(pair, tf, limit=1500)
    if not bars: return {"error": "No OHLC returned", "pair": pair, "timeframe": tf}
    ohlc = {"open":[b["open"] for b in bars], "high":[b["high"] for b in bars], "low":[b["low"] for b in bars], "close":[b["close"] for b in bars]}
    res = mod.run_scan(ohlc, params)
    return {"strategy": sid, "params": params, **res}
