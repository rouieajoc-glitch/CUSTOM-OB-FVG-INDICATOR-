"""FVG v6. Same matching as match_fvg3, but over many symbols/timeframes, pooled.
Answers: (1) is the kept set a top-N by some feature? (2) what do leftovers look like?
(3) does a small decision tree generalize to a combo it never saw (leave-one-combo-out)?
Usage: python match_fvg4.py BTCUSDT,ETHUSDT,SOLUSDT 1h,4h,12h,1d
Writes fvg_rows.csv (every gap, kept=1/0) - add it to your upload-artifact step."""
import csv
import statistics
import sys
import time

from ob_fvg import fetch_candles
from match_coinank import coinank, STEP, near
from match_swing import pivots
from match_fvg2 import candidates, TOL
from match_fvg3 import add_features

META = {"sym", "iv", "start", "i", "up", "bottom", "top", "label", "kept", "age", "n_cs", "has_v", "their_dir"}
# not comparable across timeframes / window artifacts -> kept out of the pooled tree
POOL_SKIP = {"pos_in_window", "since_prev_gap", "c1_body_pct", "c3_body_pct"}


def collect(symbol, iv):
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
    kept = {}
    for o in theirs:
        ti = int((o["ts"] - ts0) // step)
        for k, c in enumerate(cands):
            if k not in kept and c["start"] == ti and near(c["bottom"], o["low"], TOL) and near(c["top"], o["high"], TOL):
                kept[k] = o.get("direction")
                break
    rows = []
    for k, c in enumerate(cands):
        r = dict(c)
        r.update(sym=symbol, iv=iv, kept=int(k in kept), their_dir=kept.get(k), n_cs=len(cs),
                 age=len(cs) - c["start"], has_v=int("v" in cs[0]))
        rows.append(r)
    return rows, len(theirs)


def feats_of(rows):
    return sorted({k for r in rows for k, v in r.items()
                   if k not in META and isinstance(v, (int, float)) and not isinstance(v, bool)})


def topn_test(rows, feats, out, top=3):
    """For each feature+direction: how many gaps sit at-or-beyond the weakest kept gap? == P means perfect top-N."""
    pos = [r for r in rows if r["kept"]]
    P = len(pos)
    res = []
    for f in feats:
        pv = [r[f] for r in pos if f in r]
        if len(pv) < P:
            continue
        for op, t in ((">=", min(pv)), ("<=", max(pv))):
            n = sum((r[f] >= t) if op == ">=" else (r[f] <= t) for r in rows if f in r)
            res.append((n, f, op, t))
    res.sort()
    for n, f, op, t in res[:top]:
        out.append(f"    top-N test: {f} {op} {t:.3f} -> {n} gaps qualify (kept {P}){'  <== PERFECT' if n == P else ''}")


def leftovers(rows, feats, out):
    pos = [r for r in rows if r["kept"]]
    if not pos:
        return
    key = [f for f in ("gap_atr", "gap_vs_mid_body", "mid_body_atr", "gap_vs_c1_range", "mid_body_ratio") if f in feats]
    # tightest box around kept gaps on all key features -> who else is inside?
    lo = {f: min(r[f] for r in pos) for f in key}
    inside = [r for r in rows if not r["kept"] and all(r[f] >= lo[f] for f in key)]
    out.append(f"    box on {key} (min of kept): {len(inside)} non-kept gaps inside")
    out.append("    kept   (start, age, up, " + ", ".join(key) + ")")
    for r in pos:
        out.append("      K %4d %4d %s " % (r["start"], r["age"], "U" if r["up"] else "D") + " ".join(f"{r[f]:.2f}" for f in key))
    out.append("    leftovers (inside box, not kept)")
    for r in inside[:12]:
        out.append("      L %4d %4d %s " % (r["start"], r["age"], "U" if r["up"] else "D") + " ".join(f"{r[f]:.2f}" for f in key))


def age_check(rows, out):
    pos = [r["age"] / r["n_cs"] for r in rows if r["kept"]]
    neg = [r["age"] / r["n_cs"] for r in rows if not r["kept"]]
    if pos and neg:
        out.append(f"  age (0=newest,1=oldest): kept median {statistics.median(pos):.2f} vs others {statistics.median(neg):.2f}")


def pooled_tree(rows, out):
    try:
        import numpy as np
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError:
        out.append("pooled tree skipped: pip install scikit-learn numpy")
        return
    feats = [f for f in feats_of(rows) if f not in POOL_SKIP and all(f in r for r in rows)]
    X = np.array([[r[f] for f in feats] for r in rows], float)
    y = np.array([r["kept"] for r in rows])
    combos = np.array([f"{r['sym']}|{r['iv']}" for r in rows])
    mk = lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=0)
    tp = fp = fn = 0
    for g in sorted(set(combos)):
        te = combos == g
        if y[~te].sum() == 0 or y[te].sum() == 0:
            continue
        p = mk().fit(X[~te], y[~te]).predict(X[te])
        tp += int(((p == 1) & (y[te] == 1)).sum())
        fp += int(((p == 1) & (y[te] == 0)).sum())
        fn += int(((p == 0) & (y[te] == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    out.append(f"leave-one-combo-out: precision {prec:.2f} recall {rec:.2f} (tp {tp} fp {fp} fn {fn})")
    out.append("  (precision/recall near 1.0 = real rule; low = not explained by these features)")
    out.append(export_text(mk().fit(X, y), feature_names=feats))


if __name__ == "__main__":
    symbols = (sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
    ivs = (sys.argv[2] if len(sys.argv) > 2 else "1h,4h,12h,1d").split(",")
    out, allrows = [], []
    for sym in symbols:
        for iv in ivs:
            try:
                rows, n_theirs = collect(sym.strip(), iv.strip())
                P = sum(r["kept"] for r in rows)
                out.append(f"{sym} {iv}: candles {rows[0]['n_cs'] if rows else 0} | CoinAnk {n_theirs} | ours {len(rows)} | "
                           f"matched {P}/{n_theirs} | volume {bool(rows and rows[0]['has_v'])}")
                if P:
                    fe = feats_of(rows)
                    topn_test(rows, fe, out)
                    age_check(rows, out)
                    leftovers(rows, fe, out)
                    allrows += rows
            except Exception as e:
                out.append(f"{sym} {iv}: FAILED ({e})")
            time.sleep(0.5)
    out.append("")
    out.append(f"=== POOLED: {len(allrows)} gaps, {sum(r['kept'] for r in allrows)} kept ===")
    if allrows:
        pooled_tree(allrows, out)
        cols = ["sym", "iv", "kept", "their_dir"] + sorted({k for r in allrows for k in r} - {"sym", "iv", "kept", "their_dir"})
        with open("fvg_rows.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(allrows)
    print("\n".join(out))
