"""Draw candles + active OB / FVG zones as a PNG and send it to Telegram.

Rules (all calibrated in the earlier runs):
  swings  : pivot length 5 (matches CoinAnk ~98%)
  OB      : every swing candle, full high-low range. bull OB on swing lows, bear OB on swing highs.
            hidden if price moved < 0.5 ATR away within 3 candles (consistently worst in the backtest);
            "strong" if >= 1.0 ATR. Dropped once a candle CLOSES beyond its far side.
  FVG     : 3-candle gap, gap >= 0.5 x middle-candle body, and gap >= 1.0 ATR on 4h and up.
            Not drawn on 1h or lower (no edge found there). Dropped once a wick fully covers it.

Usage : python chart_zones.py BTCUSDT,ETHUSDT 4h,1d [bars]
Env   : TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (without them the PNG is only saved to ./charts)
"""
import datetime as dt
import os
import sys
import uuid
import urllib.error
import urllib.request

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ob_fvg import fetch_candles, atr_series
from match_fvg2 import candidates
from match_swing import pivots

L = 5
FETCH = 700                      # history used to find zones that are still active
MAX_PER_KIND = 5                 # nearest zones drawn per kind
FVG_MIN_ATR = {"4h": 1.0, "12h": 1.0, "1d": 1.0}     # intervals not listed here get no FVGs
BG, FG, GRID = "#0e1117", "#c9d1d9", "#1f2630"
UP, DN = "#26a69a", "#ef5350"


def fmt(p):
    return f"{p:,.0f}" if p >= 1000 else (f"{p:.2f}" if p >= 10 else f"{p:.4f}")


def find_zones(cs, iv):
    n = len(cs)
    a = atr_series(cs)
    hi, lo = pivots(cs, L)
    zones = []
    for side, idxs in (("bull", lo), ("bear", hi)):
        sign = 1 if side == "bull" else -1
        for i in idxs:
            at = a[i]
            if not at or i + L > n - 1:
                continue
            c = cs[i]
            disp = sign * (cs[i + 3]["c"] - c["c"]) / at
            if disp < 0.5:
                continue
            later = cs[i + 1:]
            if side == "bull" and any(x["c"] < c["l"] for x in later):
                continue
            if side == "bear" and any(x["c"] > c["h"] for x in later):
                continue
            touched = any((x["l"] <= c["h"]) if side == "bull" else (x["h"] >= c["l"]) for x in cs[i + L + 1:])
            zones.append({"kind": "OB", "side": side, "bottom": c["l"], "top": c["h"], "start": i,
                          "strong": disp >= 1.0, "touched": touched, "info": f"{disp:.1f}ATR"})
    min_atr = FVG_MIN_ATR.get(iv)
    if min_atr is not None:
        for c in candidates(cs):
            i = c["i"]
            m = cs[i - 1]
            body = abs(m["c"] - m["o"])
            at = a[i - 1] or a[i]
            gap = c["top"] - c["bottom"]
            if not at or body <= 0 or gap <= 0 or gap / body < 0.5 or gap / at < min_atr:
                continue
            later = cs[i + 1:]
            if c["up"] and any(x["l"] <= c["bottom"] for x in later):
                continue
            if (not c["up"]) and any(x["h"] >= c["top"] for x in later):
                continue
            touched = any((x["l"] <= c["top"]) if c["up"] else (x["h"] >= c["bottom"]) for x in later)
            zones.append({"kind": "FVG", "side": "bull" if c["up"] else "bear", "bottom": c["bottom"],
                          "top": c["top"], "start": c["start"], "strong": gap / at >= 1.5, "touched": touched,
                          "info": f"{gap / at:.1f}ATR"})
    return zones


def pick(zones, price, ylo, yhi):
    """Zones that overlap the visible price range, nearest to the current price first, capped per kind."""
    vis = [z for z in zones if z["top"] >= ylo and z["bottom"] <= yhi]
    out = []
    for kind in ("OB", "FVG"):
        k = [z for z in vis if z["kind"] == kind]
        k.sort(key=lambda z: min(abs(price - z["top"]), abs(price - z["bottom"])) if not (z["bottom"] <= price <= z["top"]) else 0)
        out += k[:MAX_PER_KIND]
    return out


def draw(cs, zones, symbol, iv, bars, path):
    win = cs[-bars:]
    off = len(cs) - len(win)
    price = win[-1]["c"]
    lo_, hi_ = min(c["l"] for c in win), max(c["h"] for c in win)
    pad = (hi_ - lo_) * 0.06
    ylo, yhi = lo_ - pad, hi_ + pad
    shown = pick(zones, price, ylo, yhi)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    right = len(win) + 1            # zones stop just after the last candle; labels sit in the right margin
    for z in shown:
        x0 = max(0, z["start"] - off)
        col = (UP if z["side"] == "bull" else DN) if z["kind"] == "OB" else ("#58a6ff" if z["side"] == "bull" else "#e3b341")
        alpha = (0.30 if z["strong"] else 0.17) * (0.65 if z["touched"] else 1.0)
        ax.add_patch(Rectangle((x0, z["bottom"]), right - x0, z["top"] - z["bottom"], facecolor=col, alpha=alpha,
                               edgecolor=col, linewidth=0.8, linestyle="--" if z["touched"] else "-", zorder=1))
        tag = f"{'bull' if z['side'] == 'bull' else 'bear'} {z['kind']}{' *' if z['strong'] else ''} {z['info']}"
        ax.text(right + 0.6, (z["top"] + z["bottom"]) / 2, tag, color=col, fontsize=6.5, ha="left", va="center", zorder=4)
    for k, c in enumerate(win):
        col = UP if c["c"] >= c["o"] else DN
        ax.vlines(k, c["l"], c["h"], color=col, linewidth=0.8, zorder=3)
        ax.add_patch(Rectangle((k - 0.3, min(c["o"], c["c"])), 0.6, max(abs(c["c"] - c["o"]), (hi_ - lo_) * 0.0015),
                               facecolor=col, edgecolor=col, linewidth=0.5, zorder=3))
    ax.axhline(price, color=FG, linewidth=0.7, linestyle=":", zorder=2)
    ax.text(right + 0.6, price, f" {fmt(price)}", color=BG, fontsize=7, va="center", ha="left",
            bbox=dict(facecolor=FG, edgecolor="none", pad=1.5), zorder=5)
    ax.set_xlim(-1, right + 17)
    ax.set_ylim(ylo, yhi)
    step = max(1, len(win) // 7)
    ticks = list(range(0, len(win), step))
    fmt_t = "%m-%d" if iv in ("1d", "12h") else "%m-%d %H:%M"
    ax.set_xticks(ticks)
    ax.set_xticklabels([dt.datetime.fromtimestamp(win[t]["t"] / 1000, dt.timezone.utc).strftime(fmt_t) for t in ticks],
                       color=FG, fontsize=7)
    ax.tick_params(axis="y", colors=FG, labelsize=7)
    ax.yaxis.tick_right()
    ax.grid(color=GRID, linewidth=0.5)
    for s in ax.spines.values():
        s.set_color(GRID)
    n_ob = sum(z["kind"] == "OB" for z in shown)
    n_fvg = sum(z["kind"] == "FVG" for z in shown)
    ax.set_title(f"{symbol} {iv}   close {fmt(price)}   |   {n_ob} OB  {n_fvg} FVG   (* = strong, dashed = already touched)",
                 color=FG, fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return shown, price


def caption(symbol, iv, shown, price):
    def line(z):
        return f"{z['side']} {z['kind']}{'*' if z['strong'] else ''} {fmt(z['bottom'])}-{fmt(z['top'])}"
    above = sorted([z for z in shown if z["bottom"] > price], key=lambda z: z["bottom"])[:3]
    below = sorted([z for z in shown if z["top"] < price], key=lambda z: -z["top"])[:3]
    inside = [z for z in shown if z["bottom"] <= price <= z["top"]]
    t = [f"{symbol} {iv}  close {fmt(price)}"]
    if inside:
        t.append("INSIDE: " + "; ".join(line(z) for z in inside))
    if above:
        t.append("above: " + "; ".join(line(z) for z in above))
    if below:
        t.append("below: " + "; ".join(line(z) for z in below))
    return "\n".join(t)[:1000]


def send_photo(path, text):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("telegram: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set, image saved only")
        return
    b = uuid.uuid4().hex
    parts = []
    for k, v in (("chat_id", chat), ("caption", text)):
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    with open(path, "rb") as fh:
        parts.append((f'--{b}\r\nContent-Disposition: form-data; name="photo"; filename="{os.path.basename(path)}"\r\n'
                      f"Content-Type: image/png\r\n\r\n").encode() + fh.read() + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendPhoto", data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"telegram: sent {os.path.basename(path)} (HTTP {r.status})")
    except urllib.error.HTTPError as e:          # never print the URL: it contains the bot token
        print(f"telegram: FAILED HTTP {e.code} {e.read().decode(errors='replace')[:200]}")
    except Exception as e:
        print(f"telegram: FAILED {type(e).__name__}")


if __name__ == "__main__":
    arg = lambda i, d: sys.argv[i] if len(sys.argv) > i and sys.argv[i].strip() else d   # empty (scheduled run) -> default
    symbols = arg(1, "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
    ivs = arg(2, "4h,1d").split(",")
    bars = int(arg(3, "150"))
    os.makedirs("charts", exist_ok=True)
    for sym in symbols:
        for iv in ivs:
            sym, iv = sym.strip(), iv.strip()
            try:
                cs = fetch_candles(sym, iv, FETCH)
                zones = find_zones(cs, iv)
                path = f"charts/{sym}_{iv}.png"
                shown, price = draw(cs, zones, sym, iv, bars, path)
                print(f"{sym} {iv}: {len(zones)} active zones, {len(shown)} drawn -> {path}")
                send_photo(path, caption(sym, iv, shown, price))
            except Exception as e:
                print(f"{sym} {iv}: FAILED ({type(e).__name__}: {e})")
