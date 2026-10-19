"""Backtest the OB/FVG rules in ob_fvg.py.  Usage: python backtest.py BTCUSDT 4h 2200"""
import sys
from datetime import datetime, timezone

from ob_fvg import fetch_candles, find_zones, evaluate, OB_MOVE, FVG_MIN, HORIZON


def stats(label, zs):
    n = len(zs)
    if not n:
        return f"{label}: no zones"
    t = [z for z in zs if z["touch"] is not None]
    if not t:
        return f"{label}: {n} zones, none touched yet"
    r = sum(z["outcome"] == "reacted" for z in t)
    b = sum(z["outcome"] == "broke" for z in t)
    wait = sum(z["touch"] - z["known"] for z in t) / len(t)
    warn = "  (small sample!)" if len(t) < 30 else ""
    return (f"{label}: {n} zones | touched {len(t)} ({len(t)/n:.0%}) | reacted {r/len(t):.0%} | "
            f"broke {b/len(t):.0%} | no reaction {(len(t)-r-b)/len(t):.0%} | "
            f"avg wait {wait:.0f} candles{warn}")


def report(title, zs):
    groups = [
        ("OB all", lambda z: z["kind"] == "OB"),
        ("OB with FVG", lambda z: z["kind"] == "OB" and z["fvg"]),
        ("OB without FVG", lambda z: z["kind"] == "OB" and not z["fvg"]),
        ("FVG all", lambda z: z["kind"] == "FVG"),
    ]
    lines = [f"== {title} =="]
    lines += [stats(lbl, [z for z in zs if f(z)]) for lbl, f in groups]
    return "\n".join(lines)


def day(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    interval = sys.argv[2] if len(sys.argv) > 2 else "4h"
    total = int(sys.argv[3]) if len(sys.argv) > 3 else 2200
    cs = fetch_candles(symbol, interval, total)
    zones = find_zones(cs)
    for z in zones:
        evaluate(cs, z)
    cut = int(len(cs) * 0.7)
    out = [
        f"{symbol} {interval} | {len(cs)} candles | {day(cs[0]['t'])} to {day(cs[-1]['t'])}",
        f"rules: OB move >= {OB_MOVE} ATR, FVG >= {FVG_MIN} ATR, reaction = close 1x zone size away "
        f"within {HORIZON} candles, break = close through the zone",
        "",
        report("ALL DATA", zones),
        "",
        report("FIRST 70% (older)", [z for z in zones if z["known"] < cut]),
        "",
        report("LAST 30% (newer, closest to out-of-sample)", [z for z in zones if z["known"] >= cut]),
        "",
        "Read: if 'reacted' is not clearly above what a random price level gives, the zones add nothing.",
    ]
    print("\n".join(out))
