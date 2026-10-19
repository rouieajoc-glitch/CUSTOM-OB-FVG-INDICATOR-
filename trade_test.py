"""Trade-style test: trend filter + rejection-candle confirmation, vs matched random controls.
Usage: python trade_test.py BTCUSDT,ETHUSDT,SOLUSDT 1h,4h,1d [null]

Rules (fixed in advance, NOT tuned on the data):
  Entry  : after a zone is known, wait for first touch, then enter at the close of the first
           'rejection candle' within 6 candles: bullish = touches the zone, closes back above it,
           and is green (bearish = mirror). If a candle closes through the zone first -> no trade.
  Trend  : variant 'rejection+trend' also needs close above EMA200 for bullish (below for bearish).
  Exit   : stop at the far edge of the zone; target 1R or 2R; max 24 candles (then close at market).
           If stop and target hit in the same candle, it counts as a loss. Fees NOT included.
  Control: for each real zone, 20 fake zones (same side/size/distance, random moment after a
           similar impulse) go through the exact same entry, trend and exit rules.
"""
import math
import random
import sys

from ob_fvg import fetch_candles, find_zones, HORIZON, MAX_WAIT, OB_MOVE, FVG_MIN
from backtest import (fake_candles, make_control, impulse_pools, day, TOTALS,
                      CONTROLS_PER_ZONE)

EMA_N = 200
TRADE_H = 24
Z_SIG = 2.5
VARIANTS = ["rejection", "rejection+trend"]
GROUPS = ["OB", "OB with FVG", "OB without FVG", "FVG"]


def ema_series(cs, n):
    k, out, e = 2 / (n + 1), [], None
    for c in cs:
        e = c["c"] if e is None else c["c"] * k + e * (1 - k)
        out.append(e)
    return out


def find_entry(cs, z):
    bull = z["side"] == "bull"
    top, bot, n = z["top"], z["bottom"], len(cs)
    start = z["known"] + 1
    for j in range(start, min(n, start + MAX_WAIT)):
        c = cs[j]
        if not ((c["l"] <= top) if bull else (c["h"] >= bot)):
            continue
        for k in range(j, min(n, j + HORIZON + 1)):
            x = cs[k]
            if bull:
                if x["c"] < bot:
                    return None
                if x["l"] <= top and x["c"] > top and x["c"] > x["o"]:
                    return k
            else:
                if x["c"] > top:
                    return None
                if x["h"] >= bot and x["c"] < bot and x["c"] < x["o"]:
                    return k
        return None
    return None


def trade_r(cs, k, z, mult):
    bull = z["side"] == "bull"
    entry = cs[k]["c"]
    stop = z["bottom"] if bull else z["top"]
    risk = (entry - stop) if bull else (stop - entry)
    if risk <= 0:
        return None
    tgt = entry + mult * risk if bull else entry - mult * risk
    end = min(len(cs) - 1, k + TRADE_H)
    for m in range(k + 1, end + 1):
        x = cs[m]
        if bull:
            if x["l"] <= stop:
                return -1.0
            if x["h"] >= tgt:
                return float(mult)
        else:
            if x["h"] >= stop:
                return -1.0
            if x["l"] <= tgt:
                return float(mult)
    r = (cs[end]["c"] - entry) / risk if bull else (entry - cs[end]["c"]) / risk
    return max(-1.0, min(float(mult), r))


def simulate(cs, z, ema):
    """-> {variant: (R at 1R target, R at 2R target)} for the variants that produce a trade."""
    k = find_entry(cs, z)
    if k is None:
        return {}
    r1, r2 = trade_r(cs, k, z, 1), trade_r(cs, k, z, 2)
    if r1 is None or r2 is None:
        return {}
    out = {"rejection": (r1, r2)}
    bull = z["side"] == "bull"
    if k >= EMA_N and ((cs[k]["c"] > ema[k]) if bull else (cs[k]["c"] < ema[k])):
        out["rejection+trend"] = (r1, r2)
    return out


def add(cnt, base, ctrl, res):
    for v in VARIANTS:
        c = cnt.setdefault(base + (v, ctrl), [0, 0, 0.0, 0.0, 0.0, 0.0])
        c[0] += 1
        if v in res:
            r1, r2 = res[v]
            c[1] += 1
            c[2] += r1
            c[3] += r1 * r1
            c[4] += r2
            c[5] += r2 * r2


def analyse(sym, iv, cnt):
    cs = fetch_candles(sym, iv, TOTALS.get(iv, 4000))
    n = len(cs)
    last = n - MAX_WAIT - HORIZON - TRADE_H - 2
    ema = ema_series(cs, EMA_N)
    zones = [z for z in find_zones(cs) if z["known"] <= last]
    bull, bear = impulse_pools(cs)
    fallback = list(range(20, last))
    pools = {"bull": [r for r in bull if r <= last] or fallback,
             "bear": [r for r in bear if r <= last] or fallback}
    rng = random.Random(7)
    cut = int(n * 0.7)
    for z in zones:
        groups = ["FVG"] if z["kind"] == "FVG" else ["OB", "OB with FVG" if z["fvg"] else "OB without FVG"]
        part = "older 70%" if z["known"] < cut else "newer 30%"
        res = simulate(cs, z, ema)
        ctrls = [simulate(cs, make_control(cs, z, rng.choice(pools[z["side"]])), ema)
                 for _ in range(CONTROLS_PER_ZONE)]
        for g in groups:
            add(cnt, (sym, iv, g, part), False, res)
            for cr in ctrls:
                add(cnt, (sym, iv, g, part), True, cr)
    return f"{sym} {iv}: {n} candles {day(cs[0]['t'])} to {day(cs[-1]['t'])}, {len(zones)} zones tested"


def agg(cnt, iv=None, grp=None, variant=None, part=None):
    out = {False: [0, 0, 0.0, 0.0, 0.0, 0.0], True: [0, 0, 0.0, 0.0, 0.0, 0.0]}
    for (s, i, g, p, v, ctrl), val in cnt.items():
        if (iv and i != iv) or (grp and g != grp) or (variant and v != variant) or (part and p != part):
            continue
        for k in range(6):
            out[ctrl][k] += val[k]
    return out[False], out[True]


def stat(real, ctrl, which):
    nr, nc = real[1], ctrl[1]
    s, ss = (2, 3) if which == 1 else (4, 5)
    mr, mc = real[s] / nr, ctrl[s] / nc
    vr = max(real[ss] / nr - mr * mr, 0)
    vc = max(ctrl[ss] / nc - mc * mc, 0)
    se = math.sqrt(vr / nr + vc / nc) or 1e-9
    return mr, mc, (mr - mc) / se


def line(label, real, ctrl):
    if real[1] < 2 or ctrl[1] < 2:
        return f"{label}: not enough trades"
    m1, c1, z1 = stat(real, ctrl, 1)
    m2, c2, z2 = stat(real, ctrl, 2)
    verdict = "BETTER" if min(z1, z2) >= Z_SIG else "WORSE" if max(z1, z2) <= -Z_SIG else "no edge"
    few = " (few!)" if real[1] < 30 else ""
    return (f"{label}: trades {real[1]}{few} | 1R: {m1:+.2f}R vs random {c1:+.2f}R (z={z1:+.1f}) | "
            f"2R: {m2:+.2f}R vs {c2:+.2f}R (z={z2:+.1f}) | {verdict}")


def run(symbols, intervals):
    cnt, heads = {}, []
    for s in symbols:
        for iv in intervals:
            try:
                heads.append(analyse(s, iv, cnt))
            except Exception as e:
                heads.append(f"{s} {iv}: FAILED ({e})")
    out = [f"entry: rejection candle at zone | trend: EMA{EMA_N} | stop = far edge of zone | targets 1R/2R | "
           f"max {TRADE_H} candles | fees NOT included | BETTER/WORSE needs both z >= {Z_SIG}",
           "avg R = average result per trade in units of risk (0 = break-even, +0.10 = gains 10% of risk per trade)",
           ""] + heads + [""]
    out.append("== EVERYTHING POOLED ==")
    for v in VARIANTS:
        for g in GROUPS:
            out.append(line(f"{v} | {g}", *agg(cnt, grp=g, variant=v)))
    out += ["", "== BY TIMEFRAME =="]
    for iv in intervals:
        for v in VARIANTS:
            for g in ("OB", "FVG"):
                out.append(line(f"{iv} {v} | {g}", *agg(cnt, iv=iv, grp=g, variant=v)))
    out += ["", "== OLDER vs NEWER DATA =="]
    for part in ("older 70%", "newer 30%"):
        for v in VARIANTS:
            for g in ("OB", "FVG"):
                out.append(line(f"{part} {v} | {g}", *agg(cnt, grp=g, variant=v, part=part)))
    out += ["", "Note: coins move together, so pooled samples are smaller than they look."]
    return "\n".join(out)


if __name__ == "__main__":
    symbols = (sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
    intervals = (sys.argv[2] if len(sys.argv) > 2 else "1h,4h,1d").split(",")
    if len(sys.argv) > 3 and sys.argv[3] == "null":
        fetch_candles = fake_candles
        print("NULL TEST: random-walk data, any 'edge' below is just the test's own bias\n")
    print(run(symbols, intervals))
