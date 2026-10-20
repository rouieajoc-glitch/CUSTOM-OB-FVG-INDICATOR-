import json
import os
import time
import requests

API_KEY = os.environ["COINANK_API_KEY"]
SYMBOL = "BTCUSDT"
INTERVALS = ["4h", "1d"]
OUT_DIR = "coinank_dump"

HEADERS = {"apikey": API_KEY}
SMC_URL = "https://open-api.coinank.com/api/indicator/smc"


def fetch(interval):
    params = {"symbol": SYMBOL, "interval": interval, "exchange": "Binance"}
    r = requests.get(SMC_URL, headers=HEADERS, params=params, timeout=15)
    return r.json()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, interval in enumerate(INTERVALS):
        if i:
            time.sleep(2)
        resp = fetch(interval)
        if not resp.get("success"):
            print(interval, "API Error:", resp)
            continue

        data = resp["data"]
        path = f"{OUT_DIR}/{SYMBOL}_{interval}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"=== {interval} -> {path}")
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, list):
                    print(f"{key}: {len(val)} items")
                    if val:
                        print("  sample:", json.dumps(val[0]))
                else:
                    print(f"{key}: {json.dumps(val)[:200]}")
        else:
            print("data type:", type(data).__name__)


if __name__ == "__main__":
    main()
