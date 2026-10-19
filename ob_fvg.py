"""Custom Order Block (OB) + Fair Value Gap (FVG) detector. Standard library only.

Rules (v1):
  Bullish OB : last red candle i, next candle green, next 3 closes all above
               red candle's high, and rally >= OB_MOVE x ATR. Zone = low..high of candle i.
  Bearish OB : mirror image.
  Bullish FVG: candle3.low > candle1.high, gap >= FVG_MIN x ATR. Zone = candle1.high..candle3.low.
  Bearish FVG: mirror image.
A zone is only "known" when its last confirming candle has closed (no lookahead).
"""
import json
import urllib.parse
import urllib.request

OB_MOVE = 1.5
FVG_MIN = 0.5
CONFIRM = 3
HORIZON = 6  # candles to wait for a reaction after first touch


# ---------- data ----------
def _get(url, params):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{q}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def _c(r):
    return {"t": int(r[0]), "o": float(r[1]), "h": float(r[2]), "l": float(r[3]), "c": float(r[4])}


def _binance(host, symbol, interval, total):
    out, end = [], None
    while len(out) < total:
        p = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end:
            p["endTime"] = end
        rows = _get(host + "/api/v3/klines", p)
        if not rows:
            break
        out = [_c(r) for r in rows] + out
        end = rows[0][0] - 1
        if len(rows) < 1000:
            break
    return out[-total:]


def _bybit(symbol, interval, total):
    iv = {"15m": "15", "1h": "60", "4h": "240", "1d": "D"}[interval]
    out, end = [], None
    while len(out) < total:
        p = {"category": "linear", "symbol": symbol, "interval": iv, "limit": 1000}
        if end:
            p["end"] = end
        rows = _get("https://api.bybit.com/v5/market/kline", p)["result"]["list"]
        if not rows:
            break
        rows = [_c(r) for r in rows][::-1]
        out = rows + out
        end = rows[0]["t"] - 1
        if len(rows) < 1000:
            break
    return out[-total:]


def fetch_candles(symbol="BTCUSDT", interval="4h", total=2200):
    """Tries Binance, then Binance's data mirror, then Bybit. Drops the unfinished last candle."""
    sources = [
        ("binance", lambda: _binance("https://api.binance.com", symbol, interval, total)),
        ("binance-data", lambda: _binance("https://data-api.binance.vision", symbol, interval, total)),
        ("bybit", lambda: _bybit(symbol, interval, total)),
    ]
    for name, fn in sources:
        try:
            cs = fn()
            if len(cs) > 100:
                print(f"data source: {name} ({len(cs)} candles)")
                return cs[:-1]
        except Exception as e:  # blocked region, rate limit, etc.
            print(f"{name} failed: {e}")
    raise RuntimeError("all data sources failed")


# ---------- detection ----------
def atr_series(cs, n=14):
    tr = []
    for i, c in enumerate(cs):
        if i == 0:
            tr.append(c["h"] - c["l"])
        else:
            pc = cs[i - 1]["c"]
            tr.append(max(c["h"] - c["l"], abs(c["h"] - pc), abs(c["l"] - pc)))
    atr = [None] * len(cs)
    for i in range(n - 1, len(cs)):
        atr[i] = sum(tr[i - n + 1:i + 1]) / n
    return atr


def find_zones(cs, ob_move=OB_MOVE, fvg_min=FVG_MIN, confirm=CONFIRM):
    a = atr_series(cs)
    zones, n = [], len(cs)
    for i in range(n):
        c = cs[i]
        if not a[i]:
            continue
        if i + confirm < n:
            nxt = cs[i + 1:i + 1 + confirm]
            if (c["c"] < c["o"] and cs[i + 1]["c"] > cs[i + 1]["o"]
                    and all(x["c"] > c["h"] for x in nxt)
                    and max(x["h"] for x in nxt) - c["h"] >= ob_move * a[i]):
                zones.append({"kind": "OB", "side": "bull", "top": c["h"], "bottom": c["l"],
                              "src": i, "known": i + confirm})
            if (c["c"] > c["o"] and cs[i + 1]["c"] < cs[i + 1]["o"]
                    and all(x["c"] < c["l"] for x in nxt)
                    and c["l"] - min(x["l"] for x in nxt) >= ob_move * a[i]):
                zones.append({"kind": "OB", "side": "bear", "top": c["h"], "bottom": c["l"],
                              "src": i, "known": i + confirm})
        if i >= 2:
            c1 = cs[i - 2]
            if c["l"] > c1["h"] and c["l"] - c1["h"] >= fvg_min * a[i]:
                zones.append({"kind": "FVG", "side": "bull", "top": c["l"], "bottom": c1["h"],
                              "src": i - 2, "known": i})
            if c["h"] < c1["l"] and c1["l"] - c["h"] >= fvg_min * a[i]:
                zones.append({"kind": "FVG", "side": "bear", "top": c1["l"], "bottom": c["h"],
                              "src": i - 2, "known": i})
    # tag OBs that have a same-side FVG formed inside their displacement
    fvgs = [z for z in zones if z["kind"] == "FVG"]
    for z in zones:
        if z["kind"] == "OB":
            z["fvg"] = any(f["side"] == z["side"] and f["src"] >= z["src"] and f["known"] <= z["known"]
                           for f in fvgs)
    return zones


def evaluate(cs, z, horizon=HORIZON):
    """First touch after the zone is known, then did price react (close away by 1x zone size)
    or break (close through the zone) within `horizon` candles? Uses closes only."""
    size = z["top"] - z["bottom"]
    bull = z["side"] == "bull"
    z["touch"], z["outcome"], z["filled"] = None, "untouched", False
    for j in range(z["known"] + 1, len(cs)):
        c = cs[j]
        if (c["l"] <= z["top"]) if bull else (c["h"] >= z["bottom"]):
            z["touch"], z["outcome"] = j, "no_reaction"
            for k in range(j, min(j + horizon + 1, len(cs))):
                x = cs[k]
                if bull:
                    if x["l"] <= z["bottom"]:
                        z["filled"] = True
                    if x["c"] < z["bottom"]:
                        z["outcome"] = "broke"
                        break
                    if x["c"] >= z["top"] + size:
                        z["outcome"] = "reacted"
                        break
                else:
                    if x["h"] >= z["top"]:
                        z["filled"] = True
                    if x["c"] > z["top"]:
                        z["outcome"] = "broke"
                        break
                    if x["c"] <= z["bottom"] - size:
                        z["outcome"] = "reacted"
                        break
            break
    return z
