import json
import os

import compare_smc as base
import compare3_smc as c3  # feature helpers (also installs the volume-aware archive parser)
import compare6_smc as c6

SYMBOL, DUMP, STEP = base.SYMBOL, base.DUMP_DIR, base.STEP
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


def events_info(c, n):
    """Close-break structure events with swing length n: {(idx, dir): (kind, swing_idx)}"""
    N = len(c)
    ph, pl = {}, {}
    for p in range(n, N - n):
        h, l = c[p]["h"], c[p]["l"]
        if h > max(x["h"] for x in c[p - n:p]) and h >= max(x["h"] for x in c[p + 1:p + n + 1]):
            ph[p + n] = (p, h)
        if l < min(x["l"] for x in c[p - n:p]) and l <= min(x["l"] for x in c[p + 1:p + n + 1]):
            pl[p + n] = (p, l)
    top = btm = None
    top_x = btm_x = False
    trend = 0
    ev = {}
    for j in range(N):
        if j in ph:
            top, top_x = ph[j], False
        if j in pl:
            btm, btm_x = pl[j], False
        x = c[j]
        pv = c[j - 1] if j else x
        if top is not None and not top_x and x["c"] > top[1] and pv["c"] <= top[1]:
            ev[(j, 1)] = ("choch" if trend == -1 else "bos", top[0])
            trend, top_x = 1, True
        if btm is not None and not btm_x and x["c"] < btm[1] and pv["c"] >= btm[1]:
            ev[(j, -1)] = ("choch" if trend == 1 else "bos", btm[0])
            trend, btm_x = -1, True
    return ev


def letter(k):
    return {"bos": "B", "choch": "C"}.get(k, "-")


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
    ca = {}
    for k in ("bos", "choch"):
        for e in d.get(k, []):
            ca[(e["idx"], e["dir"])] = k
    log(f"===== {interval.upper()} ({src}) =====")

    # part 1: n=5 structure vs CoinAnk, event by event
    mine = events_info(c, 5)
    keys = sorted(set(mine) | set(ca))
    cells = []
    for kk in keys:
        m = mine.get(kk)
        cells.append(f"{kk[0]}{'U' if kk[1] == 1 else 'D'} {letter(m[0]) if m else '-'}/{letter(ca.get(kk))}"
                     + (f" s{m[1]}" if m else ""))
    log("n=5 close events: idx dir mine/CoinAnk (B=bos C=choch -=none) s=swing idx broken")
    for i in range(0, len(cells), 3):
        log("  " + " | ".join(cells[i:i + 3]))

    # part 2: FVGs at short-swing breaks -> what separates kept?
    fv = c6.my_fvgs(c)
    c3.add_features(c, fv)
    for n in (3, 4):
        ev = events_info(c, n)
        pop = [g for g in fv if (g["i0"] + 1, 1 if g["dir"] == 0 else -1) in ev]
        lab = [g["i0"] in kept for g in pop]
        log(f"-- FVGs at n={n} close breaks: {len(pop)}, kept among them {sum(lab)} of {len(kept)} --")
        if sum(lab) < 2 or len(pop) - sum(lab) < 2:
            continue
        names = list(pop[0]["f"].keys())
        rows = []
        for nm in names:
            vals = [g["f"][nm] for g in pop]
            sc, t, tp, fp, fn = c3.best_thr(vals, lab)
            kmin = min(v for v, l in zip(vals, lab) if l)
            emax = max((v for v, l in zip(vals, lab) if not l), default=0.0)
            rows.append((sc, nm, t, tp, fp, fn, kmin, emax))
        rows.sort(reverse=True)
        for sc, nm, t, tp, fp, fn, kmin, emax in rows[:5]:
            log(f"  {nm}: F1 {sc:.2f} thr {t:.3f} ({tp}/{fp}/{fn}) kept>={kmin:.3f} excl<={emax:.3f}")
        top = [r[1] for r in rows[:4]]
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
        for sc, na, nb, b in pairs[:2]:
            log(f"  {na} >= {b[1]:.3f} AND {nb} >= {b[2]:.3f}: F1 {sc:.2f} ({b[3]}/{b[4]}/{b[5]})")
    log()


def main():
    for iv in ("4h", "1d"):
        try:
            run(iv)
        except Exception as e:
            log(f"[{iv}] failed: {e}")
    os.makedirs(DUMP, exist_ok=True)
    with open(f"{DUMP}/compare7_report.txt", "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
