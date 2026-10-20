import json
import os
import statistics
from collections import Counter

import compare_smc as base

SYMBOL, DUMP, STEP = base.SYMBOL, base.DUMP_DIR, base.STEP
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def pivots(c, n, left_strict, right_strict):
    highs, lows = [], []
    for i in range(n, len(c) - n):
        h, l = c[i]["h"], c[i]["l"]
        lm = max(x["h"] for x in c[i - n:i])
        rm = max(x["h"] for x in c[i + 1:i + n + 1])
        if (h > lm if left_strict else h >= lm) and (h > rm if right_strict else h >= rm):
            highs.append(c[i]["ts"])
        lm = min(x["l"] for x in c[i - n:i])
        rm = min(x["l"] for x in c[i + 1:i + n + 1])
        if (l < lm if left_strict else l <= lm) and (l < rm if right_strict else l <= rm):
            lows.append(c[i]["ts"])
    return highs, lows


def in_range(mine, theirs):
    lo, hi = min(theirs), max(theirs)
    return {x for x in mine if lo <= x <= hi}


def swing_block(c, d, idx_of, n=5):
    th = {x["ts"]: x for x in d["swingHighs"]}
    tl = {x["ts"]: x for x in d["swingLows"]}
    log("-- swings (n=5) tie rules --")
    best = None
    for name, ls, rs in (("L strict,R loose", True, False), ("both strict", True, True),
                         ("both loose", False, False), ("L loose,R strict", False, True)):
        h, l = pivots(c, n, ls, rs)
        mh, ml = in_range(h, th), in_range(l, tl)
        tp = len(mh & set(th)) + len(ml & set(tl))
        fp = len(mh - set(th)) + len(ml - set(tl))
        fn = len(set(th) - mh) + len(set(tl) - ml)
        log(f"  {name}: {tp}/{fp}/{fn}")
        if best is None or f1(tp, fp, fn) > best[0]:
            best = (f1(tp, fp, fn), ls, rs)
    h, l = pivots(c, n, best[1], best[2])
    mh, ml = in_range(h, th), in_range(l, tl)
    for label, miss, extra, src in (
        ("high", set(th) - mh, mh - set(th), "h"), ("low", set(tl) - ml, ml - set(tl), "l")
    ):
        for tag, group in (("missing", miss), ("extra", extra)):
            for ts in sorted(group)[:3]:
                i = idx_of[ts]
                v = c[i][src]
                if src == "h":
                    L = max(x["h"] for x in c[i - n:i]) - v
                    R = max(x["h"] for x in c[i + 1:i + n + 1]) - v
                else:
                    L = v - min(x["l"] for x in c[i - n:i])
                    R = v - min(x["l"] for x in c[i + 1:i + n + 1])
                log(f"  {tag} {label} idx {i} {v}: left nb {L:+.1f}, right nb {R:+.1f}")


def first_fill(c, g, rule):
    lo, hi = g["low"], g["high"]
    mid = (lo + hi) / 2
    for j in range(g["i0"] + 3, len(c)):
        x = c[j]
        if g["dir"] == 0:  # bullish gap, price retraces down
            hit = {"touch": x["l"] <= hi, "mid": x["l"] <= mid,
                   "full_wick": x["l"] <= lo, "full_close": x["c"] <= lo}[rule]
        else:
            hit = {"touch": x["h"] >= lo, "mid": x["h"] >= mid,
                   "full_wick": x["h"] >= hi, "full_close": x["c"] >= hi}[rule]
        if hit:
            return j
    return None


def fvg_block(c, d, idx_of):
    log("-- FVG --")
    mine = {}
    for i in range(2, len(c)):
        a, b = c[i - 2], c[i]
        if b["l"] > a["h"]:
            mine[i - 2] = dict(i0=i - 2, dir=0, low=a["h"], high=b["l"])
        elif b["h"] < a["l"]:
            mine[i - 2] = dict(i0=i - 2, dir=1, low=b["h"], high=a["l"])
    kept = []
    for f in d.get("fvgData", []):
        i0 = idx_of.get(f["ts"])
        if i0 in mine:
            g = dict(mine[i0])
            g.update(fill=f.get("fillIdx"), bars=f.get("barCount"))
            kept.append(g)
    if not kept:
        log("  no FVG matched")
        return
    log(f"  kept {len(kept)}, my total {len(mine)}")
    log("  sample (i0, dir, barCount, fillIdx): " +
        ", ".join(f"({g['i0']},{g['dir']},{g['bars']},{g['fill']})" for g in kept[:5]))
    offs = Counter((g["fill"] - g["i0"]) - g["bars"] for g in kept if g["fill"] is not None)
    log(f"  (fillIdx - i0) - barCount: {dict(offs)}")
    for rule in ("touch", "mid", "full_wick", "full_close"):
        same = sum(1 for g in kept if first_fill(c, g, rule) == g["fill"])
        unf = sum(1 for g in kept if first_fill(c, g, rule) is None)
        log(f"  fill rule {rule}: fillIdx match {same}/{len(kept)}, rule says unfilled {unf}")
    lo, hi = min(g["i0"] for g in kept), max(g["i0"] for g in kept)
    ktab = {g["i0"] for g in kept}
    excl = [g for i, g in mine.items() if lo <= i <= hi and i not in ktab]

    def width(g):
        return (g["high"] - g["low"]) / g["low"] * 100

    def stats(lst):
        if not lst:
            return "none"
        w = [width(g) for g in lst]
        return f"{min(w):.3f}/{statistics.median(w):.3f}/{max(w):.3f}%"

    log(f"  i0 range {lo}..{hi}; excluded in range {len(excl)}")
    log(f"  width min/med/max kept {stats(kept)} | excluded {stats(excl)}")
    for rule in ("touch", "full_wick"):
        k = sum(1 for g in kept if first_fill(c, g, rule) is not None)
        e = sum(1 for g in excl if first_fill(c, g, rule) is not None)
        log(f"  filled ({rule}): kept {k}/{len(kept)}, excluded {e}/{len(excl)}")


def first_break(c, o, i0, side, rule):
    for j in range(i0 + 1, len(c)):
        x = c[j]
        if side == "bear":
            v = {"wick": x["h"], "close": x["c"], "body": max(x["o"], x["c"])}[rule]
            if v > o["h"]:
                return j
        else:
            v = {"wick": x["l"], "close": x["c"], "body": min(x["o"], x["c"])}[rule]
            if v < o["l"]:
                return j
    return None


def breaker_block(c, d, idx_of):
    log("-- breakers --")
    for okey, bkey, side in (("bearishOB", "breakerBearish", "bear"),
                             ("bullishOB", "breakerBullish", "bull")):
        obs = d.get(okey, [])
        brs = {x["ts"] for x in d.get(bkey, [])}
        obts = {x["ts"] for x in obs}
        log(f"  {bkey}: {len(brs)} | also in {okey}: {len(brs & obts)}")
        best = None
        for rule in ("wick", "close", "body"):
            pred = set()
            for o in obs:
                i0 = idx_of.get(o["ts"])
                if i0 is not None and first_break(c, o, i0, side, rule) is not None:
                    pred.add(o["ts"])
            tp, fp, fn = len(pred & brs), len(pred - brs), len(brs - pred)
            log(f"    rule {rule}: {tp}/{fp}/{fn}")
            if best is None or f1(tp, fp, fn) > best[0]:
                best = (f1(tp, fp, fn), rule, pred)
        _, rule, pred = best
        for tag, group in (("missing", brs - pred), ("extra", pred - brs)):
            for ts in sorted(group)[:3]:
                o = next((x for x in obs if x["ts"] == ts), None)
                if o is None:
                    continue
                i0 = idx_of[ts]
                bk = {r: first_break(c, o, i0, side, r) for r in ("wick", "close", "body")}
                log(f"    {tag} idx {i0}: first break wick/close/body {bk['wick']}/{bk['close']}/{bk['body']}")


def run(interval):
    path = f"{DUMP}/{SYMBOL}_{interval}.json"
    if not os.path.exists(path):
        log(f"[{interval}] missing {path}")
        return
    d = json.load(open(path))
    step = STEP[interval]
    sh = d["swingHighs"]
    t0 = sh[0]["ts"] - sh[0]["idx"] * step
    count = d["pdRange"]["lastIdx"] + 1
    src, c = base.fetch_candles(interval, t0, count)
    if not c:
        log(f"[{interval}] no candles")
        return
    idx_of = {x["ts"]: i for i, x in enumerate(c)}
    log(f"===== {interval.upper()} ({src}, {len(c)} candles) =====")
    swing_block(c, d, idx_of)
    fvg_block(c, d, idx_of)
    breaker_block(c, d, idx_of)
    log()

def pivots_edge(c, n, left_edge, right_edge):
    highs, lows = [], []
    N = len(c)
    lo = 0 if left_edge else n
    hi = N if right_edge else N - n
    for i in range(lo, hi):
        L = c[max(0, i - n):i]
        R = c[i + 1:i + n + 1]
        h, l = c[i]["h"], c[i]["l"]
        if all(h > x["h"] for x in L) and all(h >= x["h"] for x in R):
            highs.append(c[i]["ts"])
        if all(l < x["l"] for x in L) and all(l <= x["l"] for x in R):
            lows.append(c[i]["ts"])
    return highs, lows


def swing_block(c, d, idx_of, n=5):
    th = {x["ts"]: x for x in d["swingHighs"]}
    tl = {x["ts"]: x for x in d["swingLows"]}
    log("-- swings (n=5), edge handling --")
    best = None
    for name, le, re_ in (("no edges", False, False), ("start edge", True, False),
                          ("end edge", False, True), ("both edges", True, True)):
        h, l = pivots_edge(c, n, le, re_)
        mh, ml = in_range(h, th), in_range(l, tl)
        tp = len(mh & set(th)) + len(ml & set(tl))
        fp = len(mh - set(th)) + len(ml - set(tl))
        fn = len(set(th) - mh) + len(set(tl) - ml)
        log(f"  {name}: {tp}/{fp}/{fn}")
        if best is None or f1(tp, fp, fn) > best[0]:
            best = (f1(tp, fp, fn), le, re_)
    h, l = pivots_edge(c, n, best[1], best[2])
    mh, ml = in_range(h, th), in_range(l, tl)
    last = len(c) - 1
    for label, miss, extra, src in (
        ("high", set(th) - mh, mh - set(th), "h"), ("low", set(tl) - ml, ml - set(tl), "l")
    ):
        for tag, group in (("missing", miss), ("extra", extra)):
            for ts in sorted(group)[:4]:
                i = idx_of[ts]
                v = c[i][src]
                left = [x[src] for x in c[max(0, i - n):i]]
                right = [x[src] for x in c[i + 1:i + n + 1]]
                if src == "h":
                    L = max(left, default=None)
                    R = max(right, default=None)
                else:
                    L = min(left, default=None)
                    R = min(right, default=None)
                log(f"  {tag} {label} idx {i}/{last} {v}: left {L} right {R}")
def main():
    for iv in ("4h", "1d"):
        try:
            run(iv)
        except Exception as e:
            log(f"[{iv}] failed: {e}")
    os.makedirs(DUMP, exist_ok=True)
    with open(f"{DUMP}/compare2_report.txt", "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
