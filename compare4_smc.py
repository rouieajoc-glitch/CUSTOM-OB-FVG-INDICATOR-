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
    events = [(e, "bos") for e in d.get("bos", [])] + [(e, "choch") for e in d.get("choch", [])]
    log(f"===== {interval.upper()} ({src}) =====")
    log(f"FVG kept {len(kept)} of {len(fv)} | structure events: bos {len(d.get('bos', []))}, choch {len(d.get('choch', []))}")

    # what does an event idx mean?
    ok_idx = [(e, k) for e, k in events if 0 < e["idx"] < len(c)]
    ts_ok = sum(1 for e, _ in ok_idx if c[e["idx"]]["ts"] == e["ts"])
    close_beyond = sum(1 for e, _ in ok_idx
                       if (c[e["idx"]]["c"] > e["price"]) == (e["dir"] == 1)
                       and (c[e["idx"]]["c"] != e["price"]))
    prev_inside = sum(1 for e, _ in ok_idx
                      if (c[e["idx"] - 1]["c"] <= e["price"]) == (e["dir"] == 1))
    log(f"event idx: ts matches {ts_ok}/{len(ok_idx)}, close beyond level {close_beyond}/{len(ok_idx)}, prev close not beyond {prev_inside}/{len(ok_idx)}")
    ex = []
    for e, k in ok_idx:
        lst = d["swingHighs"] if e["dir"] == 1 else d["swingLows"]
        m = [s for s in lst if abs(s["price"] - e["price"]) < 0.05 and s["idx"] < e["idx"]]
        if m:
            ex.append(f"{k} idx {e['idx']} <- swing idx {m[-1]['idx']}")
    log(f"event level equals a swing price: {len(ex)}/{len(ok_idx)}; e.g. " + "; ".join(ex[:3]))

    def brk_idx(e):
        for j in range(e["idx"] + 1, len(c)):
            if (c[j]["c"] > e["price"]) if e["dir"] == 1 else (c[j]["c"] < e["price"]):
                return j
        return None

    refs = {"idx": {id(e): e["idx"] for e, _ in events},
            "break-after-idx": {id(e): brk_idx(e) for e, _ in events}}
    best = None
    for rname, rmap in refs.items():
        for dirmode in ("same", "ignore"):
            for off in (-1, 0, 1, 2):
                pred = set()
                for g in fv:
                    for e, _ in events:
                        if rmap[id(e)] != g["i0"] + off:
                            continue
                        same = (e["dir"] == 1) == (g["dir"] == 0)
                        if dirmode == "ignore" or same:
                            pred.add(g["i0"])
                tp, fp, fn = len(pred & kept), len(pred - kept), len(kept - pred)
                sc = f1(tp, fp, fn)
                log(f"  {rname}, dir {dirmode}, == i0{off:+d}: F1 {sc:.2f} ({tp}/{fp}/{fn})")
                if best is None or sc > best[0]:
                    best = (sc, rname, dirmode, off, pred)
    sc, rname, dirmode, off, pred = best
    log(f"best: {rname}, dir {dirmode}, offset {off:+d}, F1 {sc:.2f}")
    for tag, grp in (("kept FVG not explained", sorted(kept - pred)), ("predicted but not kept", sorted(pred - kept))):
        if grp:
            log(f"  {tag} (i0): {grp[:8]}")
    rmap = refs[rname]
    unmatched = [(rmap[id(e)], k) for e, k in events if not any(
        g["i0"] + off == rmap[id(e)] for g in fv)]
    log(f"  events with no FVG at that spot: {len(unmatched)} e.g. {unmatched[:5]}")
    log()


def main():
    for iv in ("4h", "1d"):
        try:
            run(iv)
        except Exception as e:
            log(f"[{iv}] failed: {e}")
    os.makedirs(DUMP, exist_ok=True)
    with open(f"{DUMP}/compare4_report.txt", "w") as f:
        f.write("\n".join(OUT))


if __name__ == "__main__":
    main()
