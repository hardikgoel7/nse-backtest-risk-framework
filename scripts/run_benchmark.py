"""Strategy 1: buy-and-hold Nifty 50. Reports price index and ETF (total-return proxy) on DEV window only."""
import sys
sys.path.insert(0, "src")
import pandas as pd
from bt.data import load_config, load_prices
from bt.metrics import summary

cfg = load_config()
rf, ppy = cfg["metrics"]["risk_free_annual"], cfg["metrics"]["periods_per_year"]

rows = {}
for label, tk in [("Nifty50 price idx", cfg["data"]["index_ticker"]),
                  ("NIFTYBEES (TR proxy)", cfg["data"]["benchmark_tr_proxy"])]:
    r = load_prices(tk, cfg)["Close"].pct_change().dropna()
    rows[label] = summary(r, rf, ppy)

out = pd.DataFrame(rows).T
print(out.round(4).to_string())
out.to_csv("reports/benchmark_dev.csv")
