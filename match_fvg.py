"""FVG diagnostic v3: find which candle fields CoinAnk's FVG high/low equal, and what size filter they use.
Usage: python match_fvg.py BTCUSDT 4h,1d   (same env as match_swing.py)"""
import statistics
import sys
import time
from collections import Counter

from ob_fvg import fetch_candles
from match_coinank import coinank, STEP, hm, near

TOL = 1e-3  # tighter than before: edges should equal a real OHLC value
FIELDS = ("o", "h", "l", "c")


def atr(cs, i, n=14):
    trs = []
    for k in range(max(1, i - n + 1), i + 1):
        trs.append(max(cs[k]["h"] - cs[k]["l"], abs(cs[k]["h"] - cs[k - 1]["c"]), abs(cs[k]["l"] - cs[k - 1]["c"])))
    return sum(trs) / len(trs) if trs else 0.0


def find_matches(cs, ti, price):
    """every (candle offset relative to their ts index, field) whose value is ~ price"""
    hits = []
    for off in range(-4, 3):
        j = ti + off
        if 0 <= j < len(cs):
            for f in FIELDS:
                if near(cs[j][f], price, TOL):
                    hits.append((off, f))
    return hits


def plain_gaps(cs):
    out = []
    for i in range(2, len(cs)):
        c1, m, c3 = cs[i - 2], cs[i - 1], cs[i]
        if c3["l"] > c1["h"]:
            out.append((i, 1, c1["h"], c3["l"]))
        if c3["h"] < c1["l"]:
            out.append((i, -1, c3["h"], c1["l"]))
    return out


def run(symbol, iv, out):
    data = coinank(symbol, iv)
    step = STEP[iv]
    ref = next((o for k in ("swingHighs", "swingLows", "bullishOB", "bearishOB") for o in data.get(k, [])
                if "idx" in o and "ts" in o), None)
    if ref is None:
        out.append(f"{symbol} {iv}: no reference idx")
        return
    ts0 = ref["ts"] - ref["idx"] * step
    need = min(6000, int((time.time() * 1000 - ts0) / step) + 5)
    cs = [c for c in fetch_candles(symbol, iv, need) if c["t"] >= ts0]
    theirs = data.get("fvgData", [])
    out.append(f"{symbol} {iv}: window {hm(cs[0]['t'])} to {hm(cs[-1]['t'])} ({len(cs)} candles) | CoinAnk FVGs {len(theirs)}")
    if theirs:
        out.append(f"  raw keys: {sorted(theirs[0].keys())}")
        out.append(f"  first raw: {theirs[0]}")

    lo_pat, hi_pat, both = Counter(), Counter(), Counter()
    gaps_pct, gaps_atr = [], []
    for o in theirs:
        ti = int((o["ts"] - ts0) // step)
        d = 1 if o.get("direction", 1) > 0 else -1
        lo_h, hi_h = find_matches(cs, ti, o["low"]), find_matches(cs, ti, o["high"])
        for h in lo_h:
            lo_pat[h] += 1
        for h in hi_h:
            hi_pat[h] += 1
        for a in lo_h:
            for b in hi_h:
                both[(a, b)] += 1
        g = (o["high"] - o["low"]) / o["low"] * 100
        gaps_pct.append(g)
        a = atr(cs, min(max(ti, 15), len(cs) - 1))
        gaps_atr.append((o["high"] - o["low"]) / a if a else 0)
        out.append(f"  {hm(o['ts'])} dir {d:+d} idx {ti} low {o['low']:.1f} high {o['high']:.1f} gap {g:.2f}% "
                   f"| low~{lo_h or '-'} | high~{hi_h or '-'} | fill {o.get('fillIdx')} bars {o.get('barCount')}")

    out.append("  most common LOW matches (candle offset vs their ts, field): " + str(lo_pat.most_common(5)))
    out.append("  most common HIGH matches: " + str(hi_pat.most_common(5)))
    out.append("  most common (low,high) pairs: " + str(both.most_common(4)))

    if theirs:
        out.append(f"  their gap %: min {min(gaps_pct):.2f} median {statistics.median(gaps_pct):.2f} max {max(gaps_pct):.2f}")
        out.append(f"  their gap / ATR14: min {min(gaps_atr):.2f} median {statistics.median(gaps_atr):.2f} max {max(gaps_atr):.2f}")
        pg = plain_gaps(cs)
        ours_pct = [(t - b) / b * 100 for _, _, b, t in pg]
        ours_atr = [(t - b) / atr(cs, i) for i, _, b, t in pg if atr(cs, i)]
        out.append(f"  our plain gaps {len(pg)} | >= their min gap%: {sum(g >= min(gaps_pct) for g in ours_pct)} "
                   f"| >= their min gap/ATR: {sum(g >= min(gaps_atr) for g in ours_atr)} (target {len(theirs)})")
        # how many of ours are bigger than their smallest, split by direction
        up = sum(1 for _, d, _, _ in pg if d > 0)
        out.append(f"  our gaps bull {up} bear {len(pg) - up} | theirs bull "
                   f"{sum(1 for o in theirs if o.get('direction', 1) > 0)} bear {sum(1 for o in theirs if o.get('direction', 1) <= 0)}")
        tis = [int((o['ts'] - ts0) // step) for o in theirs]
        out.append(f"  their ts index range: {min(tis)}..{max(tis)} of {len(cs)} (recent-only? unfilled-only? check)")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    out = []
    for iv in (sys.argv[2] if len(sys.argv) > 2 else "4h,1d").split(","):
        try:
            run(symbol, iv, out)
        except Exception as e:
            out.append(f"{symbol} {iv}: FAILED ({e})")
        out.append("")
    print("\n".join(out))
