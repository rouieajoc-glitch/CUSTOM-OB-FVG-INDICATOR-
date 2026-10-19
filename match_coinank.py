"""Compare OUR order block detector with CoinAnk's SMC order blocks (the answer key).
Usage: python match_coinank.py BTCUSDT 1h,4h,1d
Needs the COINANK_API_KEY secret. One-time calibration: the indicator itself needs no API.
Prints how well we match and the clues needed to tune our rules to theirs."""
import json
import os
import statistics
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone

from ob_fvg import fetch_candles, find_zones, atr_series

STEP = {"15m": 900000, "1h": 3600000, "4h": 14400000, "1d": 86400000}
SMC_URL = "https://open-api.coinank.com/api/indicator/smc"


def coinank(symbol, interval):
    key = os.environ.get("COINANK_API_KEY")
    if not key:
        raise RuntimeError("COINANK_API_KEY secret is missing in this repo")
    q = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "exchange": "Binance"})
    req = urllib.request.Request(f"{SMC_URL}?{q}", headers={"apikey": key, "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.loads(r.read())
    if not d.get("success"):
        raise RuntimeError(f"API error: {str(d)[:200]}")
    return d["data"]


def hm(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%m-%d %H:%M")


def near(a, b, tol):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-9)


def compare(symbol, iv, out):
    data = coinank(symbol, iv)
    lists = {k: v for k, v in data.items() if isinstance(v, list)}
    out.append(f"{symbol} {iv}: CoinAnk returned {{{', '.join(f'{k}: {len(v)}' for k, v in lists.items())}}}")
    for k, v in lists.items():
        if v:
            out.append(f"  sample {k}: {json.dumps(v[0])[:220]}")
    theirs = {"bull": data.get("bullishOB", []), "bear": data.get("bearishOB", [])}
    all_ts = [o["ts"] for lst in theirs.values() for o in lst if o.get("ts")]
    if not all_ts:
        out.append("  no order blocks returned")
        return
    step = STEP[iv]
    t0 = min(all_ts)
    need = min(6000, max(400, int((time.time() * 1000 - t0) / step) + 300))
    cs = fetch_candles(symbol, iv, need)
    idx = {c["t"]: i for i, c in enumerate(cs)}
    atr = atr_series(cs)
    ours = [z for z in find_zones(cs) if z["kind"] == "OB"]
    out.append(f"  window: {hm(t0)} to {hm(cs[-1]['t'])} UTC ({len(cs)} candles loaded)")
    for side, name in (("bull", "bullish"), ("bear", "bearish")):
        T = [o for o in theirs[side] if o.get("ts")]
        O = [z for z in ours if z["side"] == side and cs[z["src"]]["t"] >= t0]
        o_by_t = {cs[z["src"]]["t"]: z for z in O}
        offs, edge = Counter(), Counter()
        exact = price_ok = color_ok = beyond = 0
        disp, missed = [], []
        for o in T:
            ts = o["ts"]
            if O:
                best = min(((cs[z["src"]]["t"] - ts) / step for z in O), key=abs)
                if abs(best) <= 5:
                    offs[int(round(best))] += 1
            z = o_by_t.get(ts)
            if z:
                exact += 1
                if near(z["top"], o["h"], 5e-4) and near(z["bottom"], o["l"], 5e-4):
                    price_ok += 1
            else:
                missed.append(o)
            i = idx.get(ts)
            if i is None:
                continue
            c = cs[i]
            hi = "wick" if near(o["h"], c["h"], 1e-4) else "body" if near(o["h"], max(c["o"], c["c"]), 1e-4) else "other"
            lo = "wick" if near(o["l"], c["l"], 1e-4) else "body" if near(o["l"], min(c["o"], c["c"]), 1e-4) else "other"
            edge[f"high={hi}/low={lo}"] += 1
            if (c["c"] < c["o"]) == (side == "bull"):
                color_ok += 1
            if i + 3 < len(cs) and atr[i]:
                nxt = cs[i + 1:i + 4]
                if side == "bull":
                    d = max(x["h"] for x in nxt) - c["h"]
                    beyond += all(x["c"] > c["h"] for x in nxt)
                else:
                    d = c["l"] - min(x["l"] for x in nxt)
                    beyond += all(x["c"] < c["l"] for x in nxt)
                disp.append(d / atr[i])
        n = len(T)
        out.append(f"  {name}: CoinAnk {n} | ours in same window {len(O)} | same candle {exact}/{n} | "
                   f"same candle+price {price_ok}/{n}")
        out.append("    time offset ours-theirs (candles): " + (" ".join(f"{k:+d}:{v}" for k, v in sorted(offs.items())) or "none within 5"))
        out.append("    their zone edges: " + (", ".join(f"{k} x{v}" for k, v in edge.most_common()) or "n/a"))
        out.append(f"    their candle is opposite colour: {color_ok}/{n} | next 3 closes all beyond it: {beyond}/{len(disp)}")
        if disp:
            out.append(f"    move in next 3 candles (ATR): min {min(disp):.2f} median {statistics.median(disp):.2f} max {max(disp):.2f}")
        extra = [z for z in O if cs[z["src"]]["t"] not in {o['ts'] for o in T}]
        out.append(f"    we have, CoinAnk doesn't: {len(extra)}" + (" (e.g. " + ", ".join(hm(cs[z['src']]['t']) for z in extra[:4]) + ")" if extra else ""))
        for o in missed[:4]:
            i = idx.get(o["ts"])
            if i is None:
                out.append(f"    MISS {hm(o['ts'])}: candle not in our data")
                continue
            c = cs[i]
            flag = "".join(("Y" if (x["c"] > c["h"] if side == "bull" else x["c"] < c["l"]) else "n") for x in cs[i + 1:i + 4])
            out.append(f"    MISS {hm(o['ts'])} zone {o['l']:.1f}-{o['h']:.1f} | candle O{c['o']:.1f} H{c['h']:.1f} "
                       f"L{c['l']:.1f} C{c['c']:.1f} | next3 beyond: {flag}")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    intervals = (sys.argv[2] if len(sys.argv) > 2 else "1h,4h,1d").split(",")
    out = []
    for iv in intervals:
        try:
            compare(symbol, iv, out)
        except Exception as e:
            out.append(f"{symbol} {iv}: FAILED ({e})")
        out.append("")
    print("\n".join(out))
