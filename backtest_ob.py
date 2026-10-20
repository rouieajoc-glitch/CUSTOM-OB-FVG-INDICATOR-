"""OB backtest. CoinAnk's OB = every pivot-5 swing candle (full high-low range), so showing all of them is easy;
the real question is WHICH ones price respects. For each swing-candle OB we test whether price reacts or breaks
when it first returns to the zone, and compare simple filters (colour, size, impulse after the swing, FVG next to it).
No lookahead: an OB becomes usable only once its pivot is confirmed (5 candles after the swing candle).
Also prints the edge on the first vs second half of history, so a filter that only worked in one regime shows up.
Usage: python backtest_ob.py BTCUSDT,ETHUSDT,SOLUSDT 4h,12h,1d"""
import sys
import time

from ob_fvg import fetch_candles, atr_series, evaluate
from match_fvg2 import candidates
from match_swing import pivots
from backtest_fvg import atr_outcome, TOTAL

L = 5


def build(symbol, iv):
    cs = fetch_candles(symbol, iv, TOTAL.get(iv, 2000))
    a = atr_series(cs)
    hi, lo = pivots(cs, L)
    up_fvg, dn_fvg = set(), set()                 # FVGs that pass the 0.5 mid-body gate
    for c in candidates(cs):
        m = cs[c["i"] - 1]
        body = abs(m["c"] - m["o"])
        if body > 0 and (c["top"] - c["bottom"]) / body >= 0.5:
            (up_fvg if c["up"] else dn_fvg).add(c["start"])
    zones = []
    for side, idxs in (("bull", lo), ("bear", hi)):
        for i in idxs:
            at = a[i]
            if not at or i + L + 1 >= len(cs):
                continue
            c = cs[i]
            sign = 1 if side == "bull" else -1
            disp = sign * (cs[i + 3]["c"] - c["c"]) / at          # impulse away from the swing, known at i+3
            fvg = any(s in (up_fvg if side == "bull" else dn_fvg) for s in (i - 1, i, i + 1))
            z = {"kind": "OB", "side": side, "top": c["h"], "bottom": c["l"], "known": i + L, "src": i,
                 "sym": symbol, "iv": iv, "size_atr": (c["h"] - c["l"]) / at, "disp": disp, "fvg": fvg,
                 "opp": (c["c"] < c["o"]) if side == "bull" else (c["c"] > c["o"]),
                 "half": 0 if i < len(cs) / 2 else 1}
            evaluate(cs, z)
            z["out_atr"] = atr_outcome(cs, z, at)
            zones.append(z)
    return zones


FILTERS = [
    ("all swing OBs", lambda z: True),
    ("opposite colour", lambda z: z["opp"]),
    ("same colour", lambda z: not z["opp"]),
    ("size_atr>=1.0", lambda z: z["size_atr"] >= 1.0),
    ("size_atr>=1.5", lambda z: z["size_atr"] >= 1.5),
    ("size_atr<1.0", lambda z: z["size_atr"] < 1.0),
    ("impulse>=1.0 ATR", lambda z: z["disp"] >= 1.0),
    ("impulse>=2.0 ATR", lambda z: z["disp"] >= 2.0),
    ("impulse<0.5 ATR", lambda z: z["disp"] < 0.5),
    ("FVG beside swing", lambda z: z["fvg"]),
    ("FVG + impulse>=1", lambda z: z["fvg"] and z["disp"] >= 1.0),
]


def edge(L_):
    t = [z for z in L_ if z["outcome"] != "untouched"]
    if not t:
        return None, 0, 0, 0
    r = sum(z["out_atr"] == "reacted" for z in t) / len(t) * 100
    b = sum(z["out_atr"] == "broke" for z in t) / len(t) * 100
    return r - b, r, b, len(t)


def table(zs, title, out):
    out.append(title)
    out.append("  filter                n  touched | reacted% broke% edge | edge 1st half | 2nd half")
    for name, f in FILTERS:
        sel = [z for z in zs if f(z)]
        e, r, b, nt = edge(sel)
        if e is None:
            continue
        e1 = edge([z for z in sel if z["half"] == 0])
        e2 = edge([z for z in sel if z["half"] == 1])
        s1 = f"{e1[0]:+6.1f}({e1[3]})" if e1[0] is not None else "   n/a"
        s2 = f"{e2[0]:+6.1f}({e2[3]})" if e2[0] is not None else "   n/a"
        out.append(f"  {name:19s} {len(sel):5d} {nt:6d} | {r:6.1f} {b:6.1f} {e:+6.1f} | {s1:>12s} | {s2:>12s}"
                   + ("  (small)" if nt < 40 else ""))


if __name__ == "__main__":
    symbols = (sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
    ivs = (sys.argv[2] if len(sys.argv) > 2 else "4h,12h,1d").split(",")
    out, allz, by_iv = [], [], {}
    for iv in ivs:
        for sym in symbols:
            try:
                zs = build(sym.strip(), iv.strip())
                by_iv.setdefault(iv, []).extend(zs)
                allz += zs
                out.append(f"{sym} {iv}: {len(zs)} swing OBs")
            except Exception as e:
                out.append(f"{sym} {iv}: FAILED ({e})")
            time.sleep(0.3)
    out.append("")
    for iv, zs in by_iv.items():
        table(zs, f"=== {iv} pooled ===", out)
        out.append("")
    if allz:
        table(allz, "=== ALL timeframes pooled ===", out)
    print("\n".join(out))
