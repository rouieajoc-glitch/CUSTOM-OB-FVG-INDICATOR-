import json
import os
from collections import Counter

import compare_smc as base

SYMBOL, DUMP, STEP = base.SYMBOL, base.DUMP_DIR, base.STEP
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def my_fvgs(c):
    res = []
    for i in range(2, len(c)):
        a, b = c[i - 2], c[i]
        if b["l"] > a["h"]:
            res.append(dict(i0=i - 2, dir=0, low=a["h"], high=b["l"]))
        elif b["h"] < a["l"]:
            res.append(dict(i0=i - 2, dir=1, low=b["h"], high=a["l"]))
    return res


def pivot_n(c, i, key, hi, nmax=20):
    best = 0
    for n in range(1, nmax + 1):
        if i - n < 0 or i + n >= len(c):
            break
        if hi:
            ok = (all(c[i][key] > c[j][key] for j in range(i - n, i))
                  and all(c[i][key] >= c[j][key] for j in range(i + 1, i + n + 1)))
        else:
            ok = (all(c[i][key] < c[j][key] for j in range(i - n, i))
                  and all(c[i][key] <= c[j][key] for j in range(i + 1, i + n + 1)))
        if not ok:
            break
        best = n
    return best


def score(pred, kept):
    tp, fp, fn = len(pred & kept), len(pred - kept), len(kept - pred)
    return f1(tp, fp, fn), tp, fp, fn


def run(interval):
    path = f"{DUMP}/{SYMBOL}_{interval}.json"
    if not os.path.exists(path):
        log(f"[{interval}] missing")
        return
    d = json.load(open(path))
    step = STEP[interval]
    t0 = d["swingHighs"][0]["ts"] - d["swingHighs"][0]["idx"] * step
    src, c = base.fetch_candles(interval, t0, d["pdRange"]["lastIdx"] + 1)
    if not c:
        log(f"[{interval}] no candles")
        return
    idx_of = {x["ts"]: i for i, x in enumerate(c)}
    kept = {idx_of.get(f["ts"]) for f in d.get("fvgData", [])} - {None}
    fv = my_fvgs(c)
    events = [(e, k) for k in ("bos", "choch") for e in d.get(k, [])]
    log(f"===== {interval.upper()} ({src}) | kept {len(kept)} of {len(fv)} FVGs =====")

    # A) what is an event?
    fields = Counter()
    pivs = Counter()
    for e, _ in events:
        i = e["idx"]
        if not 0 <= i < len(c):
            continue
        for k in ("o", "h", "l", "c"):
            if abs(c[i][k] - e["price"]) < 0.05:
                fields[k] += 1
        pivs[pivot_n(c, i, "h" if e["dir"] == 1 else "l", e["dir"] == 1)] += 1
    log(f"event price == candle field at idx: {dict(fields)} of {len(events)}")
    log(f"event candle is a pivot with n (max) = {dict(sorted(pivs.items()))}")

    fams = {}

    def add(fam, name, pred):
        fams.setdefault(fam, []).append((score(pred, kept), name, pred))

    # B) event break bar (close / wick) -> FVG offset
    def brk(e, mode):
        for j in range(e["idx"] + 1, len(c)):
            if e["dir"] == 1:
                v = c[j]["c"] if mode == "close" else c[j]["h"]
                if v > e["price"]:
                    return j
            else:
                v = c[j]["c"] if mode == "close" else c[j]["l"]
                if v < e["price"]:
                    return j
        return None

    for mode in ("close", "wick"):
        bm = [(brk(e, mode), e["dir"]) for e, _ in events]
        for off in (-1, 0, 1, 2, 3):
            pred = {g["i0"] for g in fv for (b, dr) in bm
                    if b == g["i0"] + off and (dr == 1) == (g["dir"] == 0)}
            add("event break " + mode, f"break bar == i0{off:+d}", pred)
    # event idx itself
    for off in (-1, 0, 1, 2):
        pred = {g["i0"] for g in fv for e, _ in events
                if e["idx"] == g["i0"] + off and (e["dir"] == 1) == (g["dir"] == 0)}
        add("event idx", f"event idx == i0{off:+d}", pred)

    # C) FVG starts at a swing point (bullish FVG <-> swing low)
    for rev in (False, True):
        for off in range(-3, 4):
            pred = set()
            for g in fv:
                bull = (g["dir"] == 0) != rev
                sw = d["swingLows"] if bull else d["swingHighs"]
                if any(s["idx"] == g["i0"] + off for s in sw):
                    pred.add(g["i0"])
            add("swing point", f"swing idx == i0{off:+d}{' (reversed)' if rev else ''}", pred)

    # D) FVG overlaps a recent OB of the same direction
    for K in (3, 5, 10, 20, 50):
        pred = set()
        for g in fv:
            obs = d["bullishOB"] if g["dir"] == 0 else d["bearishOB"]
            if any(g["i0"] + 2 - K <= o["idx"] <= g["i0"] + 2 and g["low"] <= o["h"] and g["high"] >= o["l"]
                   for o in obs):
                pred.add(g["i0"])
        add("overlaps recent OB", f"OB within {K} bars", pred)

    # E) liquidity sweep bar
    for off in (0, 1, 2, 3):
        pred = set()
        for g in fv:
            sw = d.get("sweepLows", []) if g["dir"] == 0 else d.get("sweepHighs", [])
            if any(s["sweepIdx"] == g["i0"] + off for s in sw):
                pred.add(g["i0"])
        add("sweep", f"sweep idx == i0{off:+d}", pred)

    bests = []
    for fam, lst in fams.items():
        lst.sort(key=lambda x: -x[0][0])
        bests.append(lst[0])
        for (sc, tp, fp, fn), name, _ in lst[:2]:
            log(f"  [{fam}] {name}: F1 {sc:.2f} ({tp}/{fp}/{fn})")
    bests.sort(key=lambda x: -x[0][0])
    # union of the two best families
    if len(bests) >= 2:
        u = bests[0][2] | bests[1][2]
        sc, tp, fp, fn = score(u, kept)
        log(f"union of best two ({bests[0][1]} + {bests[1][1]}): F1 {sc:.2f} ({tp}/{fp}/{fn})")
    log()


def main():
    for iv in ("4h", "1d"):
        try:
            run(iv)
        except Exception as e:
            log(f"[{iv}] failed: {e}")
    os.makedirs(DUMP, exist_ok=True)
    with open(f"{DUMP}/compare5_report.txt", "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
