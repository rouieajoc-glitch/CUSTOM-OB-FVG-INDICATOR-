import json
import os

import compare_smc as base

SYMBOL, DUMP, STEP = base.SYMBOL, base.DUMP_DIR, base.STEP
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


# use futures archive parser that also keeps volume
def _rows_v(url):
    import io
    import zipfile
    try:
        r = base.requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        text = z.read(z.namelist()[0]).decode()
    except Exception:
        return None
    rows = []
    for line in text.splitlines():
        p = line.split(",")
        if not p or not p[0].isdigit():
            continue
        ts = int(p[0])
        if ts > 10**14:
            ts //= 1000
        rows.append({"ts": ts, "o": float(p[1]), "h": float(p[2]), "l": float(p[3]),
                     "c": float(p[4]), "v": float(p[5]) if len(p) > 5 else 0.0})
    return rows


base._archive_rows = _rows_v


def atr_rma(c, n=14):
    out, tr, prev = [], [], None
    for i, x in enumerate(c):
        t = x["h"] - x["l"] if i == 0 else max(
            x["h"] - x["l"], abs(x["h"] - c[i - 1]["c"]), abs(x["l"] - c[i - 1]["c"]))
        tr.append(t)
        if i < n - 1:
            out.append(sum(tr) / len(tr))
        elif i == n - 1:
            prev = sum(tr) / n
            out.append(prev)
        else:
            prev = (prev * (n - 1) + t) / n
            out.append(prev)
    return out


def my_fvgs(c):
    res = []
    for i in range(2, len(c)):
        a, b = c[i - 2], c[i]
        if b["l"] > a["h"]:
            res.append(dict(i0=i - 2, dir=0, low=a["h"], high=b["l"]))
        elif b["h"] < a["l"]:
            res.append(dict(i0=i - 2, dir=1, low=b["h"], high=a["l"]))
    return res


def add_features(c, fvgs):
    atr = atr_rma(c)
    cum, s = [], 0.0
    for i, x in enumerate(c):
        s += abs(x["c"] - x["o"]) / x["o"] * 100
        cum.append(s / (i + 1))
    for g in fvgs:
        i0 = g["i0"]
        a, m = c[i0], c[i0 + 1]
        w = g["high"] - g["low"]
        body = abs(m["c"] - m["o"])
        rng = max(m["h"] - m["l"], 1e-9)
        body_pct = body / m["o"] * 100
        vols = [x.get("v", 0.0) for x in c[max(0, i0 + 1 - 20):i0 + 1]]
        allv = [x.get("v", 0.0) for x in c[:i0 + 2]]
        mv = m.get("v", 0.0)
        sb = (m["c"] - m["o"]) / m["o"] * 100
        if g["dir"] == 0:
            ok = m["c"] > a["h"]
            lux_full = sb / cum[i0 + 2] if ok and sb > 0 else 0.0
        else:
            ok = m["c"] < a["l"]
            lux_full = -sb / cum[i0 + 2] if ok and sb < 0 else 0.0
        g["f"] = {
            "gap % of price": w / g["low"] * 100,
            "gap / ATR14": w / atr[i0 + 2],
            "mid body %": body_pct,
            "mid body / ATR14": body / atr[i0 + 1],
            "mid range / ATR14": rng / atr[i0 + 1],
            "mid body / range": body / rng,
            "gap / mid body": w / max(body, 1e-9),
            "mid vol / avg20": mv / (sum(vols) / len(vols)) if vols and sum(vols) else 0.0,
            "mid vol / avg all": mv / (sum(allv) / len(allv)) if allv and sum(allv) else 0.0,
            "lux: body% / cum mean": body_pct / cum[i0 + 2],
            "lux full (dir+close beyond)": lux_full,
        }


def thresholds(vals):
    u = sorted(set(vals))
    return [u[0] - 1e-9] + [(u[i] + u[i + 1]) / 2 for i in range(len(u) - 1)]


def best_thr(vals, lab):
    best = (-1, None, 0, 0, 0)
    for t in thresholds(vals):
        tp = sum(1 for v, l in zip(vals, lab) if v >= t and l)
        fp = sum(1 for v, l in zip(vals, lab) if v >= t and not l)
        fn = sum(1 for v, l in zip(vals, lab) if v < t and l)
        sc = f1(tp, fp, fn)
        if sc > best[0]:
            best = (sc, t, tp, fp, fn)
    return best


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
    kept_i0 = {idx_of.get(f["ts"]) for f in d.get("fvgData", [])}
    fv = my_fvgs(c)
    add_features(c, fv)
    lo, hi = min(kept_i0), max(kept_i0)
    pop = [g for g in fv if lo <= g["i0"] <= hi]
    lab = [g["i0"] in kept_i0 for g in pop]
    tail = sum(1 for g in fv if g["i0"] > hi)
    log(f"===== {interval.upper()} ({src}) =====")
    log(f"population i0 {lo}..{hi}: {len(pop)} FVGs, kept {sum(lab)}; after last kept: {tail}")
    if sum(lab) < 2:
        return
    names = list(pop[0]["f"].keys())
    rows = []
    for nme in names:
        vals = [g["f"][nme] for g in pop]
        sc, t, tp, fp, fn = best_thr(vals, lab)
        kmin = min(v for v, l in zip(vals, lab) if l)
        emax = max((v for v, l in zip(vals, lab) if not l), default=0.0)
        rows.append((sc, nme, t, tp, fp, fn, kmin, emax))
    rows.sort(reverse=True)
    log("single feature (best F1, threshold, tp/fp/fn, kept min, excluded max):")
    for sc, nme, t, tp, fp, fn, kmin, emax in rows:
        log(f"  {nme}: F1 {sc:.2f} thr {t:.3f} ({tp}/{fp}/{fn}) kept>={kmin:.3f} excl<={emax:.3f}")
    # pairs among the top 5
    top = [r[1] for r in rows[:5]]
    pairs = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            va = [g["f"][top[a]] for g in pop]
            vb = [g["f"][top[b]] for g in pop]
            best = (-1, None, None, 0, 0, 0)
            for ta in thresholds(va):
                sub = [(vb[i], lab[i]) for i in range(len(pop)) if va[i] >= ta]
                if not sub:
                    continue
                fn_out = sum(1 for i in range(len(pop)) if va[i] < ta and lab[i])
                for tb in thresholds([x[0] for x in sub]):
                    tp = sum(1 for v, l in sub if v >= tb and l)
                    fp = sum(1 for v, l in sub if v >= tb and not l)
                    fn = fn_out + sum(1 for v, l in sub if v < tb and l)
                    sc = f1(tp, fp, fn)
                    if sc > best[0]:
                        best = (sc, ta, tb, tp, fp, fn)
            pairs.append((best[0], top[a], top[b], best))
    pairs.sort(key=lambda x: -x[0])
    log("best AND of two (F1, thr1, thr2, tp/fp/fn):")
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
    with open(f"{DUMP}/compare3_report.txt", "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
