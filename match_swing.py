"""Calibration v2: reverse-engineer CoinAnk's SMC data (swings -> order blocks -> FVG).
Usage: python match_swing.py BTCUSDT 4h,1d      (needs COINANK_API_KEY; plan 1 returns 4h/1d only)
Facts from the first run: CoinAnk's swing counts equal its OB counts, and each OB sample sits on a
swing candle, so we test 'OB = the swing-pivot candle' and find the pivot size that matches."""
import statistics
import sys
import time
from collections import Counter

from ob_fvg import fetch_candles
from match_coinank import coinank, STEP, hm, near

TOL = 2e-3  # spot candles vs CoinAnk's (perp) prices differ a little


def pivots(cs, s):
    hi, lo = [], []
    for i in range(s, len(cs) - s):
        h, l = cs[i]["h"], cs[i]["l"]
        if all(h > cs[j]["h"] for j in range(i - s, i)) and all(h >= cs[j]["h"] for j in range(i + 1, i + s + 1)):
            hi.append(i)
        if all(l < cs[j]["l"] for j in range(i - s, i)) and all(l <= cs[j]["l"] for j in range(i + 1, i + s + 1)):
            lo.append(i)
    return hi, lo


def strength(cs, i, key, cap=40):
    v, n, left, right = cs[i][key], len(cs), 0, 0
    better = (lambda x: x < v) if key == "h" else (lambda x: x > v)
    for j in range(i - 1, max(-1, i - cap - 1), -1):
        if not better(cs[j][key]):
            break
        left += 1
    for j in range(i + 1, min(n, i + cap + 1)):
        if not better(cs[j][key]):
            break
        right += 1
    return left, right


def swing_scan(data, cs, by_t, out):
    th = {o["ts"]: o for o in data.get("swingHighs", [])}
    tl = {o["ts"]: o for o in data.get("swingLows", [])}
    rows = []
    for s in range(2, 21):
        hi, lo = pivots(cs, s)
        oh, ol = {cs[i]["t"] for i in hi}, {cs[i]["t"] for i in lo}
        tp = len(oh & th.keys()) + len(ol & tl.keys())
        fp = len(oh - th.keys()) + len(ol - tl.keys())
        fn = len(th.keys() - oh) + len(tl.keys() - ol)
        rows.append((2 * tp / max(2 * tp + fp + fn, 1), s, tp, fp, fn))
    rows.sort(reverse=True)
    out.append(f"  SWINGS: CoinAnk highs {len(th)} lows {len(tl)}. Best pivot sizes (bars each side):")
    for f1, s, tp, fp, fn in rows[:4]:
        out.append(f"    size {s}: match score {f1:.0%} | matched {tp} | we add {fp} | we miss {fn}")
    for name, T, key in (("highs", th, "h"), ("lows", tl, "l")):
        st = Counter()
        for ts in T:
            i = by_t.get(ts)
            if i is not None and 30 < i < len(cs) - 45:
                st[min(strength(cs, i, key))] += 1
        out.append(f"    their {name}: weakest side strength (bars each side that are beaten): "
                   + " ".join(f"{k}:{v}" for k, v in sorted(st.items())[:8]))
    return rows[0][1]


def ob_check(data, cs, by_t, s, out):
    th = {o["ts"] for o in data.get("swingHighs", [])}
    tl = {o["ts"] for o in data.get("swingLows", [])}
    for side, okey, bkey, piv in (("bull", "bullishOB", "breakerBullish", tl), ("bear", "bearishOB", "breakerBearish", th)):
        obs = data.get(okey, [])
        ts_set = {o["ts"] for o in obs}
        edge = Counter()
        for o in obs:
            i = by_t.get(o["ts"])
            if i is None:
                edge["candle not in our data"] += 1
                continue
            c = cs[i]
            if near(o["h"], c["h"], TOL) and near(o["l"], c["l"], TOL):
                edge["zone = candle high/low"] += 1
            elif near(o["h"], c["l"], TOL) and near(o["l"], c["h"], TOL):
                edge["zone = swapped (high-vol bar)"] += 1
            else:
                edge["other"] += 1
        out.append(f"  {okey}: {len(obs)} | on a swing candle: {len(ts_set & piv)}/{len(obs)} | "
                   + ", ".join(f"{k} x{v}" for k, v in edge.most_common()))
        bk = {o["ts"] for o in data.get(bkey, [])}
        other = {o["ts"] for o in data.get("bearishOB" if side == "bull" else "bullishOB", [])}
        out.append(f"    {bkey}: {len(bk)} | also in {okey}: {len(bk & ts_set)} | also in the opposite OB list: {len(bk & other)}")
        base = ts_set if len(bk & ts_set) >= len(bk & other) else other
        if not bk or not base:
            continue
        for rule in ("close", "wick"):
            mit, not_mit = [0, 0], [0, 0]  # [count, in breaker list]
            for o in obs:
                i = by_t.get(o["ts"])
                if i is None:
                    continue
                c = cs[i]
                later = cs[i + s + 1:]
                if side == "bull":
                    hit = any((x["c"] if rule == "close" else x["l"]) < c["l"] for x in later)
                else:
                    hit = any((x["c"] if rule == "close" else x["h"]) > c["h"] for x in later)
                bucket = mit if hit else not_mit
                bucket[0] += 1
                bucket[1] += o["ts"] in bk
            out.append(f"    if mitigated by {rule}: {mit[0]} OBs ({mit[1]} in breaker list) | not mitigated: "
                       f"{not_mit[0]} ({not_mit[1]} in breaker list)")


def fvg_candidates(cs):
    out, cum = [], 0.0
    for i in range(len(cs)):
        bd = (cs[i - 1]["c"] - cs[i - 1]["o"]) / cs[i - 1]["o"] if i else 0.0
        cum += abs(bd)
        thr = cum / i * 2 if i else 0.0
        if i < 2:
            continue
        c1, m, c3 = cs[i - 2], cs[i - 1], cs[i]
        if c3["l"] > c1["h"]:
            out.append({"dir": 1, "i": i, "top": c3["l"], "bottom": c1["h"], "close_ok": m["c"] > c1["h"],
                        "lux": m["c"] > c1["h"] and bd > thr, "gap": (c3["l"] - c1["h"]) / c1["h"] * 100,
                        "body": abs(bd) * 100})
        if c3["h"] < c1["l"]:
            out.append({"dir": -1, "i": i, "top": c1["l"], "bottom": c3["h"], "close_ok": m["c"] < c1["l"],
                        "lux": m["c"] < c1["l"] and -bd > thr, "gap": (c1["l"] - c3["h"]) / c1["l"] * 100,
                        "body": abs(bd) * 100})
    return out


def fill_idx(cs, f, rule):
    for j in range(f["i"] + 1, len(cs)):
        x = cs[j]
        if f["dir"] > 0:
            v = x["l"] if rule.startswith("wick") else x["c"]
            hit = v <= (f["bottom"] if rule.endswith("full") else f["top"])
        else:
            v = x["h"] if rule.startswith("wick") else x["c"]
            hit = v >= (f["top"] if rule.endswith("full") else f["bottom"])
        if hit:
            return j
    return -1


def fvg_check(data, cs, ts0, step, out):
    theirs = data.get("fvgData", [])
    cands = fvg_candidates(cs)
    out.append(f"  FVG: CoinAnk {len(theirs)} | our plain 3-candle gaps {len(cands)} | "
               f"+middle candle closes beyond: {sum(c['close_ok'] for c in cands)} | "
               f"+LuxAlgo size filter: {sum(c['lux'] for c in cands)}")
    pairs, offs, used = [], Counter(), set()
    for o in theirs:
        ti = int((o["ts"] - ts0) // step)
        d = 1 if o.get("direction", 1) > 0 else -1
        best = None
        for k, c in enumerate(cands):
            if c["dir"] != d or abs(c["i"] - ti) > 3:
                continue
            if near(c["bottom"], o["low"], TOL) and near(c["top"], o["high"], TOL):
                if best is None or abs(c["i"] - ti) < abs(cands[best]["i"] - ti):
                    best = k
        if best is not None:
            used.add(best)
            pairs.append((o, cands[best], ti))
            offs[cands[best]["i"] - ti] += 1
    out.append(f"    matched by direction+price: {len(pairs)}/{len(theirs)} | our 3rd-candle index minus their ts index: "
               + " ".join(f"{k:+d}:{v}" for k, v in sorted(offs.items())))
    if pairs:
        out.append(f"    of matched: middle candle closes beyond {sum(p[1]['close_ok'] for p in pairs)}/{len(pairs)} | "
                   f"passes size filter {sum(p[1]['lux'] for p in pairs)}/{len(pairs)}")
        res = {}
        for rule in ("wick-full", "wick-touch", "close-full", "close-touch"):
            same = sum(fill_idx(cs, c, rule) == (o.get("fillIdx") if (o.get("fillIdx") or -1) >= 0 else -1)
                       for o, c, _ in pairs)
            res[rule] = same
        out.append("    their fillIdx equals ours under: " + ", ".join(f"{k} {v}/{len(pairs)}" for k, v in res.items()))
        bc = sum(1 for o, c, ti in pairs if o.get("barCount") == (o.get("fillIdx") or 0) - ti)
        out.append(f"    barCount = fillIdx - start index: {bc}/{len(pairs)}")
    extras = [c for k, c in enumerate(cands) if k not in used]
    if theirs and extras:
        tg = [(o["high"] - o["low"]) / o["low"] * 100 for o in theirs]
        eg = [c["gap"] for c in extras]
        out.append(f"    gap size %: theirs min {min(tg):.2f} median {statistics.median(tg):.2f} | "
                   f"ours-not-theirs median {statistics.median(eg):.2f} max {max(eg):.2f}")
        if pairs:
            tb = [p[1]["body"] for p in pairs]
            out.append(f"    middle-candle body %: theirs min {min(tb):.2f} median {statistics.median(tb):.2f} | "
                       f"ours-not-theirs median {statistics.median(c['body'] for c in extras):.2f}")


def analyse(data, cs, ts0, iv, out):
    step = STEP[iv]
    by_t = {c["t"]: i for i, c in enumerate(cs)}
    out.append(f"  window: {hm(cs[0]['t'])} to {hm(cs[-1]['t'])} UTC ({len(cs)} candles)")
    s = swing_scan(data, cs, by_t, out)
    out.append(f"  using pivot size {s} for the checks below")
    ob_check(data, cs, by_t, s, out)
    fvg_check(data, cs, ts0, step, out)


def run(symbol, iv, out):
    data = coinank(symbol, iv)
    step = STEP[iv]
    ref = next((o for k in ("swingHighs", "swingLows", "bullishOB", "bearishOB") for o in data.get(k, [])
                if "idx" in o and "ts" in o), None)
    if ref is None:
        out.append("  no swing/OB data returned")
        return
    ts0 = ref["ts"] - ref["idx"] * step  # CoinAnk's idx 0 = this candle, so we copy their window
    need = min(6000, int((time.time() * 1000 - ts0) / step) + 5)
    cs = [c for c in fetch_candles(symbol, iv, need) if c["t"] >= ts0]
    out.append(f"{symbol} {iv}:")
    analyse(data, cs, ts0, iv, out)


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
