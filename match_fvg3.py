"""FVG v5. Geometry + fill rule are solved (see match_fvg2). Now find WHICH gaps CoinAnk keeps.
Method: for each feature take the tightest bound that keeps ALL their FVGs, count how many of our other
gaps stay inside; greedily stack features to shrink the leftovers. Usage: python match_fvg3.py BTCUSDT 4h,1d"""
import statistics
import sys
import time

from ob_fvg import fetch_candles
from match_coinank import coinank, STEP, near
from match_swing import pivots
from match_fvg2 import candidates, atr, fill_at, TOL


def add_features(cs, cands, swing_idx):
    has_v = "v" in cs[0]
    starts = [c["start"] for c in cands]
    open_gaps = []  # (bottom, top, fill_index)
    for k, c in enumerate(cands):
        i = c["i"]
        c1, m, c3 = cs[i - 2], cs[i - 1], cs[i]
        a = atr(cs, i - 1)
        r1, r3 = c1["h"] - c1["l"], c3["h"] - c3["l"]
        gap = c["top"] - c["bottom"]
        c["c1_body_pct"] = abs(c1["c"] - c1["o"]) / c1["o"] * 100
        c["c3_body_pct"] = abs(c3["c"] - c3["o"]) / c3["o"] * 100
        c["c1_range_atr"] = r1 / a
        c["c3_range_atr"] = r3 / a
        c["gap_vs_c1_range"] = gap / r1 if r1 else 0
        c["gap_vs_c3_range"] = gap / r3 if r3 else 0
        c["dist_swing"] = min((abs(p - (i - 1)) for p in swing_idx), default=99)
        c["mid_is_swing"] = int((i - 1) in swing_idx)
        c["since_prev_gap"] = (c["start"] - starts[k - 1]) if k else 99
        c["overlap_open"] = int(any(f > i and not (c["top"] < b or c["bottom"] > t) for b, t, f in open_gaps))
        c["up_flag"] = int(c["up"])
        c["pos_in_window"] = c["start"] / len(cs)
        if has_v and i >= 21:
            mv = statistics.mean(x["v"] for x in cs[i - 21:i - 1])
            c["mid_vol_vs_mean"] = m["v"] / mv if mv else 0
            c["c3_vol_vs_mean"] = c3["v"] / mv if mv else 0
        fi = fill_at(cs, c, "wick-full")
        open_gaps.append((c["bottom"], c["top"], fi if fi >= 0 else 10 ** 9))


def run(symbol, iv, out):
    data = coinank(symbol, iv)
    step = STEP[iv]
    ref = next((o for k in ("swingHighs", "swingLows", "bullishOB", "bearishOB") for o in data.get(k, [])
                if "idx" in o and "ts" in o), None)
    ts0 = ref["ts"] - ref["idx"] * step
    need = min(6000, int((time.time() * 1000 - ts0) / step) + 5)
    cs = [c for c in fetch_candles(symbol, iv, need) if c["t"] >= ts0]
    theirs = data.get("fvgData", [])
    cands = candidates(cs)
    hi, lo = pivots(cs, 5)
    add_features(cs, cands, set(hi) | set(lo))
    out.append(f"{symbol} {iv}: {len(cs)} candles | CoinAnk {len(theirs)} | our gaps {len(cands)} | volume in candles: {'v' in cs[0]}")

    pos, used = [], set()
    for o in theirs:
        ti = int((o["ts"] - ts0) // step)
        for k, c in enumerate(cands):
            if c["start"] == ti and near(c["bottom"], o["low"], TOL) and near(c["top"], o["high"], TOL):
                used.add(k)
                pos.append((o, c))
                break
    neg = [c for k, c in enumerate(cands) if k not in used]
    out.append(f"  matched {len(pos)}/{len(theirs)}")
    out.append("  their direction vs geometry: " + " ".join(
        f"{o.get('direction'):+d}{'U' if c['up'] else 'D'}" for o, c in pos) + "  (U=up-gap, D=down-gap)")
    if not pos:
        return

    skip = {"start", "i", "up", "bottom", "top", "label"}
    feats = [k for k, v in pos[0][1].items() if k not in skip and isinstance(v, (int, float))]
    bounds = {}
    for f in feats:
        pv = [c[f] for _, c in pos]
        bounds[f] = [(">=", min(pv)), ("<=", max(pv))]

    def inside(c, conds):
        return all((c[f] >= t) if op == ">=" else (c[f] <= t) for f, op, t in conds)

    conds, left = [], list(neg)
    for step_no in range(1, 5):
        best = None
        for f in feats:
            for op, t in bounds[f]:
                n = sum(inside(c, [(f, op, t)]) for c in left)
                if best is None or n < best[0]:
                    best = (n, f, op, t)
        if best is None or best[0] >= len(left):
            break
        conds.append((best[1], best[2], best[3]))
        left = [c for c in left if inside(c, [conds[-1]])]
        out.append(f"  step {step_no}: {best[1]} {best[2]} {best[3]:.3f} -> other gaps still inside: {len(left)} (of {len(neg)})")
        if not left:
            break
    if left:
        out.append("  leftover gaps (start idx): " + " ".join(str(c["start"]) for c in left[:15]))


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
