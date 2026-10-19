"""Backtest the OB/FVG rules against MATCHED RANDOM controls, across coins and timeframes.
Usage: python backtest.py BTCUSDT,ETHUSDT,SOLUSDT 1h,4h,1d

Control zone for every real zone: same side, same size, same distance from price, but placed
at a random moment that is ALSO right after a strong impulse (same market condition the real
zone is detected in). So the only difference is the location: the real OB/FVG candle vs a
random level. If real zones don't beat that, they add nothing.
"""
import math
import random
import sys
from datetime import datetime, timezone

from ob_fvg import (fetch_candles, find_zones, evaluate, atr_series,
                    OB_MOVE, FVG_MIN, HORIZON, MAX_WAIT)

TOTALS = {"15m": 12000, "1h": 8000, "4h": 4000, "1d": 2000}
CONTROLS_PER_ZONE = 20
Z_SIG = 2.5  # |z| needed to call a difference real (strict, because we look at many lines)
GROUPS = ["OB", "OB with FVG", "OB without FVG", "FVG"]


def fake_candles(sym, iv, total):
    """Pure random-walk candles (no real edge possible). Used by the 'null' check."""
    random.seed(sum(map(ord, sym + iv)))
    step = {"15m": 900000, "1h": 3600000, "4h": 14400000, "1d": 86400000}[iv]
    cs, p, vol = [], 100.0, 0.01
    for i in range(total):
        vol = max(0.003, min(0.03, vol * 0.97 + 0.03 * abs(random.gauss(0, 0.01)) + 0.0003))
        o = p
        c = o * (1 + random.gauss(0, vol))
        h = max(o, c) * (1 + abs(random.gauss(0, vol / 3)))
        l = min(o, c) * (1 - abs(random.gauss(0, vol / 3)))
        cs.append({"t": 1700000000000 + i * step, "o": o, "h": h, "l": l, "c": c})
        p = c
    return cs[:-1]


def day(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


def impulse_pools(cs):
    """Candle indexes that sit right after a strong rally / drop (same condition OBs live in)."""
    a = atr_series(cs)
    bull, bear = [], []
    for r in range(20, len(cs)):
        if not a[r]:
            continue
        d = cs[r]["c"] - cs[r - 3]["c"]
        if d >= OB_MOVE * a[r]:
            bull.append(r)
        elif -d >= OB_MOVE * a[r]:
            bear.append(r)
    return bull, bear


def make_control(cs, z, r):
    size = z["top"] - z["bottom"]
    k = z["known"]
    if z["side"] == "bull":
        gap = max(cs[k]["c"] - z["top"], 0)
        top = cs[r]["c"] - gap
        return {"side": "bull", "top": top, "bottom": top - size, "known": r}
    gap = max(z["bottom"] - cs[k]["c"], 0)
    bottom = cs[r]["c"] + gap
    return {"side": "bear", "top": bottom + size, "bottom": bottom, "known": r}


def add(cnt, key, z):
    c = cnt.setdefault(key, [0, 0, 0, 0])  # zones, touched, reacted, broke
    c[0] += 1
    if z["touch"] is not None:
        c[1] += 1
        c[2] += z["outcome"] == "reacted"
        c[3] += z["outcome"] == "broke"


def analyse(sym, iv, cnt):
    cs = fetch_candles(sym, iv, TOTALS.get(iv, 4000))
    n = len(cs)
    last = n - MAX_WAIT - HORIZON - 1  # need enough follow-up candles after a zone
    zones = [z for z in find_zones(cs) if z["known"] <= last]
    bull, bear = impulse_pools(cs)
    fallback = list(range(20, last))
    pools = {"bull": [r for r in bull if r <= last] or fallback,
             "bear": [r for r in bear if r <= last] or fallback}
    rng = random.Random(7)
    cut = int(n * 0.7)
    for z in zones:
        evaluate(cs, z)
        groups = ["FVG"] if z["kind"] == "FVG" else ["OB", "OB with FVG" if z["fvg"] else "OB without FVG"]
        part = "older 70%" if z["known"] < cut else "newer 30%"
        ctrls = []
        for _ in range(CONTROLS_PER_ZONE):
            c = make_control(cs, z, rng.choice(pools[z["side"]]))
            evaluate(cs, c)
            ctrls.append(c)
        for g in groups:
            add(cnt, (sym, iv, g, part, False), z)
            for c in ctrls:
                add(cnt, (sym, iv, g, part, True), c)
    return f"{sym} {iv}: {n} candles {day(cs[0]['t'])} to {day(cs[-1]['t'])}, {len(zones)} zones tested"


def agg(cnt, sym=None, iv=None, grp=None, part=None):
    out = {False: [0, 0, 0, 0], True: [0, 0, 0, 0]}
    for (s, i, g, p, ctrl), v in cnt.items():
        if (sym and s != sym) or (iv and i != iv) or (grp and g != grp) or (part and p != part):
            continue
        for k in range(4):
            out[ctrl][k] += v[k]
    return out[False], out[True]


def line(label, real, ctrl):
    nz, nt, rr, rb = real
    cz, ct, cr, cb = ctrl
    if nt == 0 or ct == 0:
        return f"{label}: not enough data"
    p, p0, b, b0 = rr / nt, cr / ct, rb / nt, cb / ct
    z = (p - p0) / math.sqrt(max(p0 * (1 - p0), 1e-9) / nt)
    verdict = "BETTER" if z >= Z_SIG else "WORSE" if z <= -Z_SIG else "no edge"
    few = " (few!)" if nt < 30 else ""
    return (f"{label}: n={nt}{few} | reacted {p:.0%} vs random {p0:.0%} | edge {100*(p-p0):+.0f}pt "
            f"z={z:+.1f} | broke {b:.0%} vs {b0:.0%} | {verdict}")


def run(symbols, intervals):
    cnt, heads = {}, []
    for s in symbols:
        for iv in intervals:
            try:
                heads.append(analyse(s, iv, cnt))
            except Exception as e:
                heads.append(f"{s} {iv}: FAILED ({e})")
    out = [f"rules: OB move >= {OB_MOVE} ATR, FVG >= {FVG_MIN} ATR | reaction = close 1x zone size away "
           f"within {HORIZON} candles | break = close through zone | first touch within {MAX_WAIT} candles",
           f"control: same side/size/distance, random moment after a similar impulse, "
           f"{CONTROLS_PER_ZONE} per zone | BETTER/WORSE needs |z| >= {Z_SIG}", ""]
    out += heads + [""]
    out.append("== EACH COIN + TIMEFRAME ==")
    for s in symbols:
        for iv in intervals:
            for g in ("OB", "FVG"):
                out.append(line(f"{s} {iv} {g}", *agg(cnt, sym=s, iv=iv, grp=g)))
    out += ["", "== ALL COINS, BY TIMEFRAME =="]
    for iv in intervals:
        for g in GROUPS:
            out.append(line(f"{iv} {g}", *agg(cnt, iv=iv, grp=g)))
    out += ["", "== EVERYTHING POOLED =="]
    for g in GROUPS:
        out.append(line(f"{g}", *agg(cnt, grp=g)))
    out += ["", "== POOLED: OLDER vs NEWER DATA (does it hold up later?) =="]
    for part in ("older 70%", "newer 30%"):
        for g in GROUPS:
            out.append(line(f"{part} {g}", *agg(cnt, grp=g, part=part)))
    out += ["", "Note: coins move together, so pooled samples are smaller than they look."]
    return "\n".join(out)


if __name__ == "__main__":
    symbols = (sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
    intervals = (sys.argv[2] if len(sys.argv) > 2 else "1h,4h,1d").split(",")
    if len(sys.argv) > 3 and sys.argv[3] == "null":
        fetch_candles = fake_candles
        print("NULL TEST: random-walk data, any 'edge' below is just the test's own bias\n")
    print(run(symbols, intervals))
