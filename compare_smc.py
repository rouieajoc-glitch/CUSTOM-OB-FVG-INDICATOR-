import json
import os
from collections import Counter

import requests

SYMBOL = "BTCUSDT"
DUMP_DIR = "coinank_dump"
INTERVALS = ["4h", "1d"]
STEP = {"4h": 4 * 3600 * 1000, "1d": 24 * 3600 * 1000}
SOURCES = [
    ("binance-futures", "https://fapi.binance.com/fapi/v1/klines"),
    ("binance-spot", "https://data-api.binance.vision/api/v3/klines"),
    ("binance-spot-2", "https://api.binance.com/api/v3/klines"),
]
TOL = 0.2  # price tolerance when comparing

REPORT = []


def log(s=""):
    print(s)
    REPORT.append(s)


def fetch_candles(interval, start_ms, count):
    """Return (source_name, candles) trying each source until one works."""
    for name, url in SOURCES:
        try:
            out, t = [], start_ms
            while len(out) < count:
                want = min(1000, count - len(out))
                r = requests.get(
                    url,
                    params={"symbol": SYMBOL, "interval": interval,
                            "startTime": t, "limit": want},
                    timeout=20,
                )
                rows = r.json()
                if not isinstance(rows, list) or not rows:
                    raise Exception(str(rows)[:120])
                out.extend(rows)
                t = rows[-1][0] + STEP[interval]
                if len(rows) < want:
                    break
            candles = [
                {"ts": int(x[0]), "o": float(x[1]), "h": float(x[2]),
                 "l": float(x[3]), "c": float(x[4])}
                for x in out
            ]
            return name, candles
        except Exception as e:
            log(f"  source {name} failed: {e}")
    return None, []


# ---------- swing detectors ----------
def swings_pivot(c, n):
    highs, lows = [], []
    for i in range(n, len(c) - n):
        left_h = max(x["h"] for x in c[i - n:i])
        right_h = max(x["h"] for x in c[i + 1:i + n + 1])
        if c[i]["h"] > left_h and c[i]["h"] >= right_h:
            highs.append(c[i]["ts"])
        left_l = min(x["l"] for x in c[i - n:i])
        right_l = min(x["l"] for x in c[i + 1:i + n + 1])
        if c[i]["l"] < left_l and c[i]["l"] <= right_l:
            lows.append(c[i]["ts"])
    return highs, lows


def swings_state(c, length):
    """WeloTrades / LuxAlgo style swing detection."""
    highs, lows, t = [], [], None
    for i in range(length, len(c)):
        j = i - length
        up = max(x["h"] for x in c[j + 1:i + 1])
        lo = min(x["l"] for x in c[j + 1:i + 1])
        new = 0 if c[j]["h"] > up else 1 if c[j]["l"] < lo else t
        if new == 0 and t != 0:
            highs.append(c[j]["ts"])
        if new == 1 and t != 1:
            lows.append(c[j]["ts"])
        t = new
    return highs, lows


def score(mine, theirs):
    lo, hi = min(theirs), max(theirs)
    m = {x for x in mine if lo <= x <= hi}
    t = set(theirs)
    return len(m & t), len(m - t), len(t - m)


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def check_interval(interval):
    path = f"{DUMP_DIR}/{SYMBOL}_{interval}.json"
    if not os.path.exists(path):
        log(f"[{interval}] missing {path}")
        return
    d = json.load(open(path))
    sh, sl = d.get("swingHighs", []), d.get("swingLows", [])
    step = STEP[interval]
    log(f"===== {interval.upper()} =====")
    if not sh or not sl:
        log("no swings in dump")
        return

    t0 = sh[0]["ts"] - sh[0]["idx"] * step
    aligned = sum(1 for x in sh + sl if x["ts"] == t0 + x["idx"] * step)
    last_idx = d.get("pdRange", {}).get("lastIdx") or max(x["idx"] for x in sh + sl)
    count = last_idx + 1
    log(f"bars: {count}, idx->time aligned: {aligned}/{len(sh) + len(sl)}")

    src, candles = fetch_candles(interval, t0, count)
    if not candles:
        log("could not fetch candles")
        return
    cmap = {x["ts"]: x for x in candles}
    log(f"candle source: {src} ({len(candles)} candles)")

    ok_h = sum(1 for x in sh if x["ts"] in cmap and abs(cmap[x["ts"]]["h"] - x["price"]) <= TOL)
    ok_l = sum(1 for x in sl if x["ts"] in cmap and abs(cmap[x["ts"]]["l"] - x["price"]) <= TOL)
    log(f"swing price = candle high/low: highs {ok_h}/{len(sh)}, lows {ok_l}/{len(sl)}")

    # swings
    th = [x["ts"] for x in sh]
    tl = [x["ts"] for x in sl]
    results = []
    for n in range(2, 13):
        h, l = swings_pivot(candles, n)
        a, b = score(h, th), score(l, tl)
        tot = [a[i] + b[i] for i in range(3)]
        results.append((f1(*tot), f"pivot n={n}", a, b))
    for n in range(3, 16):
        h, l = swings_state(candles, n)
        a, b = score(h, th), score(l, tl)
        tot = [a[i] + b[i] for i in range(3)]
        results.append((f1(*tot), f"state len={n}", a, b))
    results.sort(key=lambda x: -x[0])
    log("best swing methods (match/extra/missing):")
    for sc, name, a, b in results[:5]:
        log(f"  {name}: F1 {sc:.2f} | highs {a[0]}/{a[1]}/{a[2]} | lows {b[0]}/{b[1]}/{b[2]}")

    # order blocks = swing candle?
    for key, swings, field in (("bearishOB", sh, "h"), ("bullishOB", sl, "l")):
        obs = d.get(key, [])
        sts = {x["ts"] for x in swings}
        on_swing = sum(1 for o in obs if o["ts"] in sts)
        full = sum(
            1 for o in obs
            if o["ts"] in cmap
            and abs(cmap[o["ts"]]["h"] - o["h"]) <= TOL
            and abs(cmap[o["ts"]]["l"] - o["l"]) <= TOL
        )
        log(f"{key}: {len(obs)} | on swing candle {on_swing} | h/l = candle h/l {full}")

    # fvg
    mine = {}
    for i in range(2, len(candles)):
        a, b = candles[i - 2], candles[i]
        if b["l"] > a["h"]:
            mine[(round(a["h"], 1), round(b["l"], 1))] = (i, 1)
        elif b["h"] < a["l"]:
            mine[(round(b["h"], 1), round(a["l"], 1))] = (i, -1)
    fvgs = d.get("fvgData", [])
    found, offs, dirs = 0, Counter(), Counter()
    for f in fvgs:
        k = (round(f["low"], 1), round(f["high"], 1))
        if k in mine:
            found += 1
            i, dr = mine[k]
            offs[round((f["ts"] - candles[i - 2]["ts"]) / step)] += 1
            dirs[(dr, f["direction"])] += 1
    log(f"fvgData: {len(fvgs)} | standard 3-candle FVG found {found}")
    if found:
        log(f"  ts offset from 1st candle (0/1/2): {dict(offs)}")
        log(f"  (my dir, coinank dir): {dict(dirs)}")
    log()

import io
import zipfile
from datetime import datetime, timezone, timedelta

_spot_fetch = fetch_candles  # old version, kept as fallback


def _archive_rows(url):
    try:
        r = requests.get(url, timeout=30)
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
        rows.append({"ts": ts, "o": float(p[1]), "h": float(p[2]),
                     "l": float(p[3]), "c": float(p[4])})
    return rows


def fetch_candles(interval, start_ms, count):
    """Binance FUTURES candles from the public data.binance.vision archive."""
    base = "https://data.binance.vision/data/futures/um"
    end_ms = start_ms + (count - 1) * STEP[interval]
    start = datetime.fromtimestamp(start_ms / 1000, timezone.utc)
    end = datetime.fromtimestamp(end_ms / 1000, timezone.utc)
    today = datetime.now(timezone.utc).date()
    got = {}
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        name = f"BTCUSDT-{interval}-{y}-{m:02d}"
        rows = None
        if (y, m) < (today.year, today.month):
            rows = _archive_rows(f"{base}/monthly/klines/BTCUSDT/{interval}/{name}.zip")
        if rows is None:
            d = datetime(y, m, 1).date()
            while d.month == m:
                if d < today:
                    url = f"{base}/daily/klines/BTCUSDT/{interval}/BTCUSDT-{interval}-{d.isoformat()}.zip"
                    for x in _archive_rows(url) or []:
                        got[x["ts"]] = x
                d += timedelta(days=1)
        else:
            for x in rows:
                got[x["ts"]] = x
        m += 1
        if m == 13:
            y, m = y + 1, 1
    candles = [got[t] for t in sorted(got) if start_ms <= t <= end_ms]
    if len(candles) >= count * 0.9:
        return "binance-futures-archive", candles
    log(f"  futures archive incomplete ({len(candles)}/{count}), using spot")
    return _spot_fetch(interval, start_ms, count)
def main():
    for iv in INTERVALS:
        try:
            check_interval(iv)
        except Exception as e:
            log(f"[{iv}] failed: {e}")
    os.makedirs(DUMP_DIR, exist_ok=True)
    with open(f"{DUMP_DIR}/compare_report.txt", "w") as f:
        f.write("\n".join(REPORT))


if __name__ == "__main__":
    main()
