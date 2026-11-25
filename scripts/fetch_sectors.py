"""Fetches the sector label for every ticker in the universe (yfinance `.info`) and caches it to
data/sectors.csv. Resumable: tickers already in the file are skipped, so a dropped connection is safe.

CAVEAT to carry into the README: this is TODAY's sector classification applied to history. That is a
far milder problem than today's index membership, because a company rarely changes sector, but it is
not point-in-time and must be disclosed.

Usage: python scripts/fetch_sectors.py --universe data/nifty500_constituents.csv"""
import sys, os, time
sys.path.insert(0, "src")
import pandas as pd
from bt.data import load_config, read_universe, _p

cfg = load_config()
if "--universe" in sys.argv: cfg["data"]["universe_file"] = sys.argv[sys.argv.index("--universe") + 1]
out = _p(cfg, "data/sectors.csv")
done = {}
if os.path.exists(out):
    done = dict(pd.read_csv(out).dropna(subset=["ticker"]).fillna("")[["ticker", "sector"]].values)
    print(f"{len(done)} tickers already cached in {out}")

import yfinance as yf
tickers = read_universe(cfg)
todo = [t for t in tickers if t not in done]
print(f"fetching {len(todo)} of {len(tickers)}")
for i, t in enumerate(todo, 1):
    try:
        done[t] = yf.Ticker(t).info.get("sector") or "Unknown"
    except Exception:
        done[t] = "Unknown"
    if i % 25 == 0:
        pd.DataFrame(sorted(done.items()), columns=["ticker", "sector"]).to_csv(out, index=False)
        print(f"  {i}/{len(todo)}", flush=True); time.sleep(0.2)
df = pd.DataFrame(sorted(done.items()), columns=["ticker", "sector"])
df.to_csv(out, index=False)
print(f"\nwrote {out}")
print(df.sector.value_counts().to_string())
