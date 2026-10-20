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


def my_fvgs(c):
    res = []
    for i in range(2, len(c)):
        a, b = c[i - 2], c[i]
        if b["l"] > a["h"]:
            res.append(dict(i0=i - 2, dir=0, low=a["h"], high=b["l"]))
        elif b["h"] < a["l"]:
            res.append(dict(i0=i - 2, dir=1, low=b["h"], high=a["l"]))
    return res


def structure_events(c, n, mode):
    """Swing pivots of length n, confirmed n bars later; break by close or wick."""
    N = len(c)
    ph, pl = {}, {}
    for p in range(n, N - n):
        h, l = c[p]["h"], c[p]["l"]
        if h > max(x["h"] for x in c[p - n:p]) and h >= max(x["h"] for x in c[p + 1:p + n + 1]):
            ph[p + n] = h
        if l < min(x["l"] for x in c[p - n:p]) and l <= min(x["l"] for x in c[p + 1:p + n + 1]):
            pl[p + n] = l
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
        up, up_prev = (x["c"], pv["c"]) if mode == "close" else (x["h"], pv["h"])
        dn, dn_prev = (x["c"], pv["c"]) if mode == "close" else (x["l"], pv["l"])
        if top is not None and not top_x and up > top and up_prev <= top:
            ev[(j, 1)] = "choch" if trend == -1 else "bos"
            trend, top_x = 1, True
        if btm is not None and not btm_x and dn < btm and dn_prev >= btm:
            ev[(j, -1)] = "choch" if trend == 1 else "bos"
            trend, btm_x = -1, True
    return ev


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
    ca = {}
    for k in ("bos", "choch"):
        for e in d.get(k, []):
            ca[(e["idx"], e["dir"])] = k
    log(f"===== {interval.upper()} ({src}) | CoinAnk events {len(ca)}, kept FVGs {len(kept)} =====")
    rows_s, rows_f = [], []
    for mode in ("close", "wick"):
        for n in range(3, 51):
            mine = structure_events(c, n, mode)
            tp = len(set(mine) & set(ca))
            fp = len(set(mine) - set(ca))
            fn = len(set(ca) - set(mine))
            agree = sum(1 for k in set(mine) & set(ca) if mine[k] == ca[k])
            rows_s.append((f1(tp, fp, fn), mode, n, tp, fp, fn, agree))
            pred = {g["i0"] for g in fv
                    if (g["i0"] + 1, 1 if g["dir"] == 0 else -1) in mine}
            ftp, ffp, ffn = len(pred & kept), len(pred - kept), len(kept - pred)
            rows_f.append((f1(ftp, ffp, ffn), mode, n, ftp, ffp, ffn))
    rows_s.sort(reverse=True)
    rows_f.sort(reverse=True)
    log("structure events vs CoinAnk bos/choch (F1, tp/fp/fn, bos-vs-choch label agrees):")
    for sc, mode, n, tp, fp, fn, ag in rows_s[:6]:
        log(f"  swing n={n} {mode}: F1 {sc:.2f} ({tp}/{fp}/{fn}) label {ag}/{tp}")
    log("FVG at my structure break (middle candle) vs kept FVGs:")
    for sc, mode, n, tp, fp, fn in rows_f[:6]:
        log(f"  swing n={n} {mode}: F1 {sc:.2f} ({tp}/{fp}/{fn})")
    log()


def main():
    for iv in ("4h", "1d"):
        try:
            run(iv)
        except Exception as e:
            log(f"[{iv}] failed: {e}")
    os.makedirs(DUMP, exist_ok=True)
    with open(f"{DUMP}/compare6_report.txt", "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
