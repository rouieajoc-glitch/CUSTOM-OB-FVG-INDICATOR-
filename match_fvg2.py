"""FVG v4. Known: ts = FIRST candle of the pattern, wick-to-wick gap, their direction label is inverted
(-1 = up-gap, +1 = down-gap). Finds which feature separates their FVGs from our other plain gaps.
Usage: python match_fvg2.py BTCUSDT 4h,1d"""
import statistics
import sys
import time
from collections import Counter

from ob_fvg import fetch_candles
from match_coinank import coinank, STEP, hm, near

TOL = 1e-3


def atr(cs, i, n=14):
    trs = [max(cs[k]["h"] - cs[k]["l"], abs(cs[k]["h"] - cs[k - 1]["c"]), abs(cs[k]["l"] - cs[k - 1]["c"]))
           for k in range(max(1, i - n + 1), i + 1)]
    return sum(trs) / len(trs) if trs else 1.0


def candidates(cs):
    out, cum = [], 0.0
    for i in range(2, len(cs)):
        c1, m, c3 = cs[i - 2], cs[i - 1], cs[i]
        mb = abs(m["c"] - m["o"]) / m["o"]
        cum = sum(abs(cs[k]["c"] - cs[k]["o"]) / cs[k]["o"] for k in range(0, i)) / i  # mean body% before c3
        a = atr(cs, i - 1)
        for up in (True, False):
            if up and not c3["l"] > c1["h"]:
                continue
            if not up and not c3["h"] < c1["l"]:
                continue
            bot, top = (c1["h"], c3["l"]) if up else (c3["h"], c1["l"])
            gap = top - bot
            beyond = (m["c"] > c1["h"]) if up else (m["c"] < c1["l"])
            dirn = (m["c"] > m["o"]) == up  # middle candle colour agrees with gap direction
            out.append({
                "start": i - 2, "i": i, "up": up, "bottom": bot, "top": top, "label": -1 if up else 1,
                "gap_pct": gap / bot * 100,
                "gap_atr": gap / a,
                "mid_body_pct": mb * 100,
                "mid_body_vs_mean": mb / cum if cum else 0,
                "mid_body_atr": abs(m["c"] - m["o"]) / a,
                "mid_range_atr": (m["h"] - m["l"]) / a,
                "mid_body_ratio": abs(m["c"] - m["o"]) / (m["h"] - m["l"]) if m["h"] > m["l"] else 0,
                "gap_vs_mid_range": gap / (m["h"] - m["l"]) if m["h"] > m["l"] else 0,
                "gap_vs_mid_body": gap / abs(m["c"] - m["o"]) if m["c"] != m["o"] else 99,
                "beyond": int(beyond), "colour_ok": int(dirn),
            })
    return out


def fill_at(cs, f, rule):
    for j in range(f["i"] + 1, len(cs)):
        x = cs[j]
        wick = rule.startswith("wick")
        full = rule.endswith("full")
        if f["up"]:
            v = x["l"] if wick else x["c"]
            if v <= (f["bottom"] if full else f["top"]):
                return j
        else:
            v = x["h"] if wick else x["c"]
            if v >= (f["top"] if full else f["bottom"]):
                return j
    return -1


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
    out.append(f"{symbol} {iv}: {len(cs)} candles | CoinAnk {len(theirs)} | our plain gaps {len(cands)}")

    pos, used = [], set()
    for o in theirs:
        ti = int((o["ts"] - ts0) // step)
        hit = None
        for k, c in enumerate(cands):
            if c["start"] == ti and near(c["bottom"], o["low"], TOL) and near(c["top"], o["high"], TOL):
                hit = k
                break
        if hit is None:
            out.append(f"  NO MATCH: idx {ti} low {o['low']} high {o['high']} dir {o.get('direction')}")
        else:
            used.add(hit)
            pos.append((o, cands[hit]))
    out.append(f"  matched {len(pos)}/{len(theirs)} | label rule (-1=up-gap) holds: "
               f"{sum(1 for o, c in pos if o.get('direction') == c['label'])}/{len(pos)}")
    neg = [c for k, c in enumerate(cands) if k not in used]
    if not pos:
        return

    feats = [k for k in pos[0][1] if k not in ("start", "i", "up", "bottom", "top", "label")]
    out.append("  feature | theirs min/median/max | others median/max | best '>= threshold' (TP FP FN)")
    scored = []
    for f in feats:
        pv = sorted(c[f] for _, c in pos)
        nv = [c[f] for c in neg]
        best = None
        for t in sorted(set(pv)):
            tp = sum(v >= t for v in pv)
            fp = sum(v >= t for v in nv)
            fn = len(pv) - tp
            err = fp + fn
            if best is None or err < best[0]:
                best = (err, t, tp, fp, fn)
        scored.append((best[0], f, pv, nv, best))
    for err, f, pv, nv, b in sorted(scored)[:7]:
        out.append(f"   {f}: {pv[0]:.3f}/{statistics.median(pv):.3f}/{pv[-1]:.3f} | "
                   f"{statistics.median(nv):.3f}/{max(nv):.3f} | >= {b[1]:.3f} ({b[2]} {b[3]} {b[4]})")

    # fill rules
    for rule in ("wick-touch", "wick-full", "close-touch", "close-full"):
        exact, diffs = 0, Counter()
        for o, c in pos:
            fi = o.get("fillIdx")
            theirs_fill = fi if (fi is not None and fi >= 0) else -1
            ours = fill_at(cs, c, rule)
            exact += theirs_fill == ours
            if theirs_fill >= 0 and ours >= 0:
                diffs[ours - theirs_fill] += 1
        out.append(f"  fill {rule}: exact {exact}/{len(pos)} | offset(ours-theirs) {dict(diffs)}")
    bc = Counter()
    for o, c in pos:
        fi = o.get("fillIdx")
        if fi is not None and fi >= 0:
            bc["filled: barCount==fill-start" if o["barCount"] == fi - c["start"] else "filled: other"] += 1
        else:
            bc[f"unfilled: barCount {o['barCount']} vs last-start {len(cs) - 1 - c['start']}"] += 1
    out.append("  barCount: " + "; ".join(f"{k} x{v}" for k, v in bc.items()))
    # unselected recent gaps: are they just newer than some cutoff?
    last_theirs = max(c["start"] for _, c in pos)
    out.append(f"  their last start idx {last_theirs} of {len(cs)}; our other gaps after that: "
               f"{sum(c['start'] > last_theirs for c in neg)}")


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
