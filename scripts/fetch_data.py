"""Usage:  python scripts/fetch_data.py   (run from repo root, with internet)
Downloads Nifty 50 constituents + index + ETF proxy, then prints:
  1. RAW quality report (what Yahoo gave us, flags kept visible)
  2. what clean_prices() removed, per ticker."""
import sys
sys.path.insert(0, "src")
import pandas as pd
from bt.data import load_config, read_universe, download, load_prices, quality_report, clean_prices, quarantine_reason

cfg = load_config()
if "--universe" in sys.argv: cfg["data"]["universe_file"] = sys.argv[sys.argv.index("--universe") + 1]
QUIET = "--quiet" in sys.argv
tickers = read_universe(cfg) + [cfg["data"]["index_ticker"], cfg["data"]["benchmark_tr_proxy"]]
if "--skip-download" not in sys.argv:
    download(tickers, cfg)

print("\n--- RAW quality report (dev window only) ---")
for t in tickers:
    try:
        raw = load_prices(t, cfg, clean=False)
    except FileNotFoundError:
        print(t, "MISSING"); continue
    q = quality_report(raw)
    if q["rows"] == 0:
        print(t, "NO DEV-WINDOW DATA (listed after 2023?) -> excluded"); continue
    flag = "" if not (q["dup_dates"] or q["nan_close"] or q["nonpositive"] or q["jumps"]) else "  <-- REVIEW"
    if flag or not QUIET: print(t, q["first"], q["last"], q["rows"], q["jumps"], flag)

print("\n--- cleaning actions (clean_prices) ---")
for t in tickers:
    try:
        raw = load_prices(t, cfg, clean=False)
        if len(raw) == 0: continue
        _, log = clean_prices(raw)
    except FileNotFoundError:
        continue
    if log:
        print(t, log)

print("\n--- QUARANTINED (big move left after cleaning -> excluded from tests) ---")
nq = 0
for t in tickers:
    try:
        d = load_prices(t, cfg)
    except FileNotFoundError:
        continue
    if len(d) and (q := quarantine_reason(d)):
        nq += 1; print(t, q)
print(f"{nq} tickers quarantined")
