"""FVG filter backtest. No CoinAnk needed: uses long Binance history on any timeframe.
For every FVG (our exact geometry), test: does price react when it first returns to the gap?
Compares filters: all gaps vs gap_vs_mid_body >= 0.5 vs 0.5 + size cutoffs.
Usage: python backtest_fvg.py BTCUSDT,ETHUSDT,SOLUSDT 1h,4h,12h,1d
No lookahead: a gap is usable only after its 3rd candle closed; first touch is searched afterwards.

Two reaction measures (closes only, within HORIZON candles after first touch):
  size  : close >= 1x gap size away from the zone  (ob_fvg.evaluate; harder for BIG gaps)
  atr   : close >= 1 ATR away from the zone         (fair across gap sizes)
broke = close beyond the far side of the gap first."""
import sys
import time

from ob_fvg import fetch_candles, atr_series, evaluate, HORIZON, MAX_WAIT
from match_fvg2 import candidates

TOTAL = {"1h": 4000, "4h": 3000, "12h": 2000, "1d": 2000}


def atr_outcome(cs, z, a_ref):
    """First touch after known, then reacted/broke measured in ATR units."""
    bull = z["side"] == "bull"
    for j in range(z["known"] + 1, min(len(cs), z["known"] + 1 + MAX_WAIT)):
        c = cs[j]
        if (c["l"] <= z["top"]) if bull else (c["h"] >= z["bottom"]):
            for k in range(j, min(j + HORIZON + 1, len(cs))):
                x = cs[k]
                if bull:
                    if x["c"] < z["bottom"]:
                        return "broke"
                    if x["c"] >= z["top"] + a_ref:
                        return "reacted"
                else:
                    if x["c"] > z["top"]:
                        return "broke"
                    if x["c"] <= z["bottom"] - a_ref:
                        return "reacted"
            return "no_reaction"
    return "untouched"


def build(symbol, iv):
    cs = fetch_candles(symbol, iv, TOTAL.get(iv, 2000))
    a = atr_series(cs)
    zones = []
    for c in candidates(cs):
        i = c["i"]
        m = cs[i - 1]
        at = a[i - 1] or a[i]
        body = abs(m["c"] - m["o"])
        gap = c["top"] - c["bottom"]
        if not at or body <= 0 or gap <= 0:
            continue
        z = {"kind": "FVG", "side": "bull" if c["up"] else "bear", "top": c["top"], "bottom": c["bottom"],
             "known": i, "src": i - 2, "sym": symbol, "iv": iv,
             "gvmb": gap / body, "gap_atr": gap / at, "mid_body_atr": body / at}
        evaluate(cs, z)                       # sets z["outcome"] (size-based)
        z["out_atr"] = atr_outcome(cs, z, at)
        zones.append(z)
    return zones


FILTERS = [
    ("all gaps", lambda z: True),
    ("gvmb>=0.5", lambda z: z["gvmb"] >= 0.5),
    ("0.5 + gap_atr>=0.75", lambda z: z["gvmb"] >= 0.5 and z["gap_atr"] >= 0.75),
    ("0.5 + gap_atr>=1.0", lambda z: z["gvmb"] >= 0.5 and z["gap_atr"] >= 1.0),
    ("0.5 + gap_atr>=1.5", lambda z: z["gvmb"] >= 0.5 and z["gap_atr"] >= 1.5),
    ("0.5 + body_atr>=1.5", lambda z: z["gvmb"] >= 0.5 and z["mid_body_atr"] >= 1.5),
    ("gvmb<0.5 (rejected)", lambda z: z["gvmb"] < 0.5),
]


def table(zs, title, out):
    out.append(title)
    out.append("  filter                 n   touched | reacted% broke% (size) | reacted% broke% (ATR) | edge(ATR)")
    for name, f in FILTERS:
        sel = [z for z in zs if f(z)]
        t = [z for z in sel if z["outcome"] != "untouched"]
        if not sel:
            continue
        nt = len(t) or 1
        rs = sum(z["outcome"] == "reacted" for z in t) / nt * 100
        bs = sum(z["outcome"] == "broke" for z in t) / nt * 100
        ra = sum(z["out_atr"] == "reacted" for z in t) / nt * 100
        ba = sum(z["out_atr"] == "broke" for z in t) / nt * 100
        out.append(f"  {name:20s} {len(sel):5d} {len(t):6d} | {rs:6.1f} {bs:6.1f}        | {ra:6.1f} {ba:6.1f}       | {ra - ba:+6.1f}"
                   + ("  (small sample)" if len(t) < 40 else ""))


if __name__ == "__main__":
    symbols = (sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
    ivs = (sys.argv[2] if len(sys.argv) > 2 else "1h,4h,12h,1d").split(",")
    out, allz = [], []
    by_iv = {}
    for iv in ivs:
        for sym in symbols:
            try:
                zs = build(sym.strip(), iv.strip())
                by_iv.setdefault(iv, []).extend(zs)
                allz += zs
                out.append(f"{sym} {iv}: {len(zs)} gaps")
            except Exception as e:
                out.append(f"{sym} {iv}: FAILED ({e})")
            time.sleep(0.3)
    out.append("")
    for iv, zs in by_iv.items():
        table(zs, f"=== {iv} pooled over {len(set(z['sym'] for z in zs))} symbols ===", out)
        out.append("")
    if allz:
        table(allz, "=== ALL timeframes pooled ===", out)
        # consistency: does the 0.5 gate help on every symbol/timeframe?
        out.append("")
        out.append("gate consistency (ATR edge: gvmb>=0.5 minus gvmb<0.5, touched gaps only)")
        for iv, zs in by_iv.items():
            for sym in sorted(set(z["sym"] for z in zs)):
                g = [z for z in zs if z["sym"] == sym and z["outcome"] != "untouched"]
                hi = [z for z in g if z["gvmb"] >= 0.5]
                lo = [z for z in g if z["gvmb"] < 0.5]
                if len(hi) < 10 or len(lo) < 10:
                    continue
                e = lambda L: (sum(z["out_atr"] == "reacted" for z in L) - sum(z["out_atr"] == "broke" for z in L)) / len(L) * 100
                out.append(f"  {sym} {iv}: kept {len(hi)} edge {e(hi):+.1f} | rejected {len(lo)} edge {e(lo):+.1f}")
    print("\n".join(out))
