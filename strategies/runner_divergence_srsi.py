# strategies/runner_divergence_srsi.py
from typing import List, Dict, Any
import math

def ema(series: List[float], length: int) -> List[float]:
    if length <= 1 or len(series) == 0: return series[:]
    k = 2 / (length + 1)
    out = [series[0]]
    for i in range(1, len(series)):
        out.append(series[i]*k + out[-1]*(1-k))
    return out

def rsi(series: List[float], length: int) -> List[float]:
    gains, losses = [0.0], [0.0]
    for i in range(1, len(series)):
        ch = series[i] - series[i-1]
        gains.append(max(ch,0.0))
        losses.append(max(-ch,0.0))
    def sma(x, n):
        s, out = 0.0, []
        for i,v in enumerate(x):
            s += v
            if i>=n: s -= x[i-n]
            out.append(s/n if i>=n-1 else float('nan'))
        return out
    avg_gain = sma(gains, length); avg_loss = sma(losses, length)
    out = []
    for g,l in zip(avg_gain, avg_loss):
        if math.isnan(g) or math.isnan(l): out.append(float('nan')); continue
        rs = g / l if l != 0 else 1e9
        out.append(100 - (100 / (1 + rs)))
    return out

def stoch(series: List[float], k_len: int, d_len: int) -> (List[float], List[float]):
    k = []
    for i in range(len(series)):
        start = max(0, i-k_len+1)
        window = [v for v in series[start:i+1] if not math.isnan(v)]
        if len(window) < 1: k.append(float('nan')); continue
        lo, hi = min(window), max(window)
        if hi == lo: k.append(50.0)
        else: k.append(100*(series[i]-lo)/(hi-lo))
    d = []
    s = 0.0
    for i,v in enumerate(k):
        s += (0 if math.isnan(v) else v)
        if i>=d_len: s -= (0 if math.isnan(k[i-d_len]) else k[i-d_len])
        d.append(s/d_len if i>=d_len-1 else float('nan'))
    return k, d

def atr(high: List[float], low: List[float], close: List[float], length: int) -> List[float]:
    trs = []
    for i in range(len(close)):
        if i==0: trs.append(high[i]-low[i]); continue
        tr = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        trs.append(tr)
    out, s = [], 0.0
    for i,v in enumerate(trs):
        s += v
        if i>=length: s -= trs[i-length]
        out.append(s/length if i>=length-1 else float('nan'))
    return out

def sma(series: List[float], length: int) -> List[float]:
    s, out = 0.0, []
    for i,v in enumerate(series):
        s += (0 if math.isnan(v) else v)
        if i>=length:
            s -= (0 if math.isnan(series[i-length]) else series[i-length])
        out.append(s/length if i>=length-1 else float('nan'))
    return out

def find_swings(high: List[float], low: List[float], lookback: int):
    pivH, pivL = [], []
    n = len(high)
    for i in range(lookback, n-lookback):
        if all(high[i] >= high[j] for j in range(i-lookback, i+lookback+1)):
            pivH.append(i)
        if all(low[i] <= low[j] for j in range(i-lookback, i+lookback+1)):
            pivL.append(i)
    return pivH, pivL

def detect_divergence(close: List[float], srsi: List[float], pivH: List[int], pivL: List[int]):
    bull, bear = [], []
    for i in range(1, len(pivL)):
        a,b = pivL[i-1], pivL[i]
        if close[b] < close[a] and srsi[b] > srsi[a]: bull.append(b)
    for i in range(1, len(pivH)):
        a,b = pivH[i-1], pivH[i]
        if close[b] > close[a] and srsi[b] < srsi[a]: bear.append(b)
    return {"bull": bull, "bear": bear}

def last_break_of_structure(close: List[float], lookback: int):
    up, dn = [], []
    for i in range(lookback, len(close)):
        ref_lo = min(close[i-lookback:i])
        ref_hi = max(close[i-lookback:i])
        if close[i] > ref_hi: up.append(i)
        if close[i] < ref_lo: dn.append(i)
    return up, dn

def reversal_candle(open_: List[float], high: List[float], low: List[float], close: List[float], idx: int, direction: str) -> bool:
    body = abs(close[idx] - open_[idx])
    range_ = high[idx] - low[idx]
    if range_ == 0: return False
    wick_top = high[idx] - max(close[idx], open_[idx])
    wick_bot = min(close[idx], open_[idx]) - low[idx]
    if direction == "long":
        return (wick_bot > body and close[idx] > open_[idx]) or (idx>0 and close[idx] > close[idx-1] and open_[idx] < close[idx-1])
    else:
        return (wick_top > body and close[idx] < open_[idx]) or (idx>0 and close[idx] < close[idx-1] and open_[idx] > close[idx-1])

def run_scan(ohlc: Dict[str, List[float]], params: Dict[str, Any]) -> Dict[str, Any]:
    o, h, l, c = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
    ema_tr = ema(c, params.get("ema_trend",200))
    atr_v = atr(h, l, c, params.get("atr_len",14))
    atr_avg = sma(atr_v, params.get("atr_avg_len",20))
    r = rsi(c, params.get("srsi_len",14))
    k,d = stoch(r, params.get("srsi_k",3), params.get("srsi_d",3))
    pivH, pivL = find_swings(h, l, params.get("swing_lookback",20))
    div = detect_divergence(c, k, pivH, pivL)
    bos_up, bos_dn = last_break_of_structure(c, params.get("structure_lookback",10))

    signals = []
    for i in range(len(c)):
        score = 0
        if i in div["bull"]: score += 2
        if i in div["bear"]: score += 2
        if i in bos_up or i in bos_dn: score += 2
        if not math.isnan(ema_tr[i]): score += 1
        if not math.isnan(atr_v[i]) and not math.isnan(atr_avg[i]) and atr_v[i] > atr_avg[i]: score += 1
        if score >= 6:
            direction = "long" if i in div["bull"] else ("short" if i in div["bear"] else None)
            if direction:
                signals.append({"index": i, "direction": direction, "score": score})
    return {"signals": signals, "count": len(signals)}

def run_backtest(ohlc: Dict[str, List[float]], params: Dict[str, Any]) -> Dict[str, Any]:
    o, h, l, c = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
    ema_tr = ema(c, params.get("ema_trend",200))
    atr_v = atr(h, l, c, params.get("atr_len",14))
    atr_avg = sma(atr_v, params.get("atr_avg_len",20))
    r = rsi(c, params.get("srsi_len",14))
    k,d = stoch(r, params.get("srsi_k",3), params.get("srsi_d",3))
    pivH, pivL = find_swings(h, l, params.get("swing_lookback",20))
    div = detect_divergence(c, k, pivH, pivL)
    bos_up, bos_dn = last_break_of_structure(c, params.get("structure_lookback",10))

    trades = []
    tp1_rr = params.get("tp1_rr",1.2); tp2_rr=params.get("tp2_rr",2.0)
    for i in range(len(c)):
        direction = None
        if i in div["bull"] and c[i] >= ema_tr[i] and atr_v[i] > atr_avg[i]: direction = "long"
        if i in div["bear"] and c[i] <= ema_tr[i] and atr_v[i] > atr_avg[i]: direction = "short"
        if not direction: continue
        if direction == "long" and not any(j>=i-params.get("structure_lookback",10) and j<=i for j in bos_up): continue
        if direction == "short" and not any(j>=i-params.get("structure_lookback",10) and j<=i for j in bos_dn): continue
        if not reversal_candle(o,h,l,c,i,direction): continue
        if i+1>=len(c): break
        entry = o[i+1]
        sl_buffer = atr_v[i]
        if math.isnan(sl_buffer): continue
        if direction=="long":
            sl = min(l[i], min(l[max(0,i-2):i+1])) - sl_buffer
            tp1 = entry + (entry - sl)*tp1_rr
            tp2 = entry + (entry - sl)*tp2_rr
        else:
            sl = max(h[i], max(h[max(0,i-2):i+1])) + sl_buffer
            tp1 = entry - (sl - entry)*tp1_rr
            tp2 = entry - (sl - entry)*tp2_rr
        hit_tp1 = hit_tp2 = hit_sl = False; end=i+1; exit_price=c[i+1]
        for j in range(i+1, len(c)):
            hh,ll = h[j], l[j]
            if direction=="long":
                if ll <= sl: hit_sl=True; end=j; exit_price=sl; break
                if hh >= tp2: hit_tp2=True; end=j; exit_price=tp2; break
                if hh >= tp1: hit_tp1=True
            else:
                if hh >= sl: hit_sl=True; end=j; exit_price=sl; break
                if ll <= tp2: hit_tp2=True; end=j; exit_price=tp2; break
                if ll <= tp1: hit_tp1=True
        outcome = "tp2" if hit_tp2 else ("sl" if hit_sl else ("tp1" if hit_tp1 else "timeout"))
        trades.append({"entry_index": i+1, "exit_index": end, "direction": direction, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2, "outcome": outcome})
    return {"trades": trades, "n": len(trades)}
