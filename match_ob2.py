"""OB v2. Hypothesis from the last run: CoinAnk's order blocks ARE the swing candles
(bullishOB count == swingLows count, bearishOB count == swingHighs count in all 9 combos).
Bullish OB = the swing-LOW candle's full high-low range; bearish OB = the swing-HIGH candle's range.
This script checks it: (1) which pivot length reproduces their swings, (2) do their OBs sit on their swing
candles, (3) does our pivot candle's high/low match their zone, and how far apart are the two price feeds.
Usage: python match_ob2.py BTCUSDT,ETHUSDT,SOLUSDT 4h,12h,1d"""
import statistics
import sys
import time

from ob_fvg import fetch_candles
from match_coinank import coinank, STEP
from match_swing import pivots


def window(data, iv, symbol):
    step = STEP[iv]
    ref = next((o for k in ("swingHighs", "swingLows", "bullishOB", "bearishOB") for o in data.get(k, [])
                if "idx" in o and "ts" in o), None)
    ts0 = ref["ts"] - ref["idx"] * step
    need = min(6000, int((time.time() * 1000 - ts0) / step) + 5)
    cs = [c for c in fetch_candles(symbol, iv, need) if c["t"] >= ts0]
    return cs, {c["t"]: i for i, c in enumerate(cs)}


def pct(a, b):
    return (a - b) / b * 100 if b else 0.0


def run(symbol, iv, out):
    data = coinank(symbol, iv)
    cs, tidx = window(data, iv, symbol)
    last_ok = len(cs) - 6                     # newest candles cannot be confirmed pivots yet
    th = {tidx[o["ts"]] for o in data.get("swingHighs", []) if o["ts"] in tidx}
    tl = {tidx[o["ts"]] for o in data.get("swingLows", []) if o["ts"] in tidx}
    out.append(f"{symbol} {iv}: candles {len(cs)} | their swingHighs {len(th)} swingLows {len(tl)} | "
               f"bearOB {len(data.get('bearishOB', []))} bullOB {len(data.get('bullishOB', []))}")

    # (1) which pivot length reproduces their swings?
    best = None
    for L in (2, 3, 4, 5, 6, 7, 8):
        hi, lo = pivots(cs, L)
        oh, ol = set(hi), set(lo)
        tp = len(oh & th) + len(ol & tl)
        fp = len(oh - th) + len(ol - tl)
        fn = len(th - oh) + len(tl - ol)
        out.append(f"  pivot L={L}: match {tp}/{len(th) + len(tl)} | ours-only {fp} | theirs-only {fn}")
        if best is None or (tp - fp - fn) > best[0]:
            best = (tp - fp - fn, L)
    L = best[1]
    hi, lo = pivots(cs, L)
    out.append(f"  best pivot length: {L}")

    # (2)+(3) their OBs vs their swings vs our candle
    for side, key, sw, ours, hi_key in (("bull", "bullishOB", tl, set(lo), "l"), ("bear", "bearishOB", th, set(hi), "h")):
        obs = data.get(key, [])
        on_swing = on_ours = 0
        d = {"o": [], "h": [], "l": [], "c": []}
        edge = {0.05: 0, 0.1: 0, 0.2: 0, 0.5: 0}
        for o in obs:
            i = tidx.get(o["ts"])
            if i is None:
                continue
            on_swing += i in sw
            on_ours += i in ours
            c = cs[i]
            for k in d:
                d[k].append(pct(o[k], c[k]))
            worst = max(abs(pct(o["h"], c["h"])), abs(pct(o["l"], c["l"])))
            for t in edge:
                edge[t] += worst <= t
        n = len(obs)
        out.append(f"  {side}: their OB on their swing candle {on_swing}/{n} | on OUR pivot candle {on_ours}/{n}")
        if d["h"]:
            out.append("    their zone vs our candle (median % diff): "
                       + " ".join(f"{k}{statistics.median(v):+.3f}" for k, v in d.items()))
            out.append("    zone high AND low within: " + " ".join(f"{t}%:{v}/{n}" for t, v in edge.items()))
        # direction of OB candle (is it just the swing candle regardless of colour?)
        opp = sum((cs[tidx[o['ts']]]["c"] < cs[tidx[o['ts']]]["o"]) == (side == "bull") for o in obs if o["ts"] in tidx)
        out.append(f"    candle is opposite colour: {opp}/{n} (if ~50% the colour does not matter)")
        # swing candles in the last few bars (unconfirmed) are expected to be missing from ours
        miss = sorted(sw - ours)
        out.append(f"    their swing candles we do not pivot: {len(miss)} e.g. idx {miss[:6]}"
                   + (f" (last confirmable idx {last_ok})" if miss and max(miss) > last_ok else ""))


if __name__ == "__main__":
    symbols = (sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
    ivs = (sys.argv[2] if len(sys.argv) > 2 else "4h,12h,1d").split(",")
    out = []
    for sym in symbols:
        for iv in ivs:
            try:
                run(sym.strip(), iv.strip(), out)
            except Exception as e:
                out.append(f"{sym} {iv}: FAILED ({e})")
            out.append("")
            time.sleep(0.5)
    print("\n".join(out))
