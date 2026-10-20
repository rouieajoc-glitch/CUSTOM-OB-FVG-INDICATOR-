import json
import os

import compare_smc as base
import compare3_smc as c3
import compare6_smc as c6
import compare7_smc as c7

SYMBOL, DUMP, STEP = base.SYMBOL, base.DUMP_DIR, base.STEP
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


def fill_bar(c, g, rule):
    lo, hi = g["low"], g["high"]
    for j in range(g["i0"] + 3, len(c)):
        x = c[j]
        if g["dir"] == 0:
            hit = x["l"] <= (hi if rule == "touch" else lo)
        else:
            hit = x["h"] >= (lo if rule == "touch" else hi)
        if hit:
            return j
    return None


def extra_features(c, g, evs, atr):
    i0 = g["i0"]
    a, m, b = c[i0], c[i0 + 1], c[i0 + 2]
    bull = g["dir"] == 0
    f = g["f"]
    f["mid candle color matches"] = 1 if ((m["c"] > m["o"]) == bull) else 0
    f["first candle color matches"] = 1 if ((a["c"] > a["o"]) == bull) else 0
    f["third candle color matches"] = 1 if ((b["c"] > b["o"]) == bull) else 0
    f["mid close beyond first extreme"] = 1 if ((m["c"] > a["h"]) if bull else (m["c"] < a["l"])) else 0
    f["third close beyond mid extreme"] = 1 if ((b["c"] > m["h"]) if bull else (b["c"] < m["l"])) else 0
    ff, ft = fill_bar(c, g, "full_wick"), fill_bar(c, g, "touch")
    f["bars to full fill (999=never)"] = (ff - i0) if ff is not None else 999
    f["bars to touch (999=never)"] = (ft - i0) if ft is not None else 999
    d = 1 if bull else -1
    for n, ev in evs.items():
        f[f"break of n={n} swing at mid"] = 1 if (i0 + 1, d) in ev else 0
        f[f"break of n={n} swing at third"] = 1 if (i0 + 2, d) in ev else 0
    rng = m["h"] - m["l"]
    for k in (5, 10, 20):
        prev = [x["h"] - x["l"] for x in c[max(0, i0 + 1 - k):i0 + 1]]
        f[f"mid range / max range prev {k}"] = rng / max(prev) if prev and max(prev) else 0.0
    f["first body / ATR14"] = abs(a["c"] - a["o"]) / atr[i0]
    f["third body / ATR14"] = abs(b["c"] - b["o"]) / atr[i0 + 2]


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
    fv = c6.my_fvgs(c)
    c3.add_features(c, fv)
    atr = c3.atr_rma(c)
    evs = {n: c7.events_info(c, n) for n in (3, 4, 5)}
    for g in fv:
        extra_features(c, g, evs, atr)
    pop = [g for g in fv if g["f"]["gap / mid body"] >= 0.5]
    lab = [g["i0"] in kept for g in pop]
    log(f"===== {interval.upper()} ({src}) =====")
    log(f"population: gap >= 0.5 x mid body: {len(pop)} of {len(fv)} FVGs; kept {sum(lab)} of {len(kept)}")
    if sum(lab) < 2 or len(pop) - sum(lab) < 2:
        log()
        return
    rows = []
    for nm in pop[0]["f"].keys():
        vals = [g["f"][nm] for g in pop]
        sc, t, tp, fp, fn = c3.best_thr(vals, lab)
        kmin = min(v for v, l in zip(vals, lab) if l)
        emax = max((v for v, l in zip(vals, lab) if not l), default=0.0)
        rows.append((sc, nm, t, tp, fp, fn, kmin, emax))
    rows.sort(reverse=True)
    for sc, nm, t, tp, fp, fn, kmin, emax in rows[:8]:
        log(f"  {nm}: F1 {sc:.2f} thr {t:.3f} ({tp}/{fp}/{fn}) kept>={kmin:.3f} excl<={emax:.3f}")
    top = [r[1] for r in rows[:5]]
    pairs = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            va = [g["f"][top[a]] for g in pop]
            vb = [g["f"][top[b]] for g in pop]
            best = (-1, 0, 0, 0, 0, 0)
            for ta in c3.thresholds(va):
                sub = [(vb[i], lab[i]) for i in range(len(pop)) if va[i] >= ta]
                if not sub:
                    continue
                fn_out = sum(1 for i in range(len(pop)) if va[i] < ta and lab[i])
                for tb in c3.thresholds([x[0] for x in sub]):
                    tp = sum(1 for v, l in sub if v >= tb and l)
                    fp = sum(1 for v, l in sub if v >= tb and not l)
                    fn = fn_out + sum(1 for v, l in sub if v < tb and l)
                    sc = c3.f1(tp, fp, fn)
                    if sc > best[0]:
                        best = (sc, ta, tb, tp, fp, fn)
            pairs.append((best[0], top[a], top[b], best))
    pairs.sort(key=lambda x: -x[0])
    for sc, na, nb, b in pairs[:3]:
        log(f"  {na} >= {b[1]:.3f} AND {nb} >= {b[2]:.3f}: F1 {sc:.2f} ({b[3]}/{b[4]}/{b[5]})")
    log()


def main():
    for iv in ("4h", "1d"):
        try:
            run(iv)
        except Exception as e:
            log(f"[{iv}] failed: {e}")
    os.makedirs(DUMP, exist_ok=True)
    with open(f"{DUMP}/compare8_report.txt", "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
