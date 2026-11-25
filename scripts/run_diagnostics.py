"""Diagnostics on the DEV window. These EXPLAIN results; they never change the rule.
1. Excess return by signal year   2. Share of episodes in the 3 busiest months
3. Excess with 2020 removed (is one event doing all the work?)"""
import sys
sys.path.insert(0, "src")
import pandas as pd
from bt.data import load_config, load_prices, read_universe
from bt.signals import signal_a, signal_a2, signal_b
from bt.event_study import collect, cluster_bootstrap_ci

cfg = load_config(); s = cfg["strategy"]
closes = {}
for tk in read_universe(cfg):
    try:
        closes[tk] = load_prices(tk, cfg)["Close"].dropna()
    except FileNotFoundError:
        pass

for name, fn in [("A", signal_a), ("A2", signal_a2), ("B", signal_b)]:
    for h in s["primary_horizons"]:
        eps, _ = collect(closes, fn, s, h)
        ex = eps.ret - eps.base_mean
        print(f"\n=== Strategy {name}, horizon {h}d: {len(ex)} episodes ===")
        by = ex.groupby(ex.index.year).agg(episodes="count", mean_excess="mean").round(4)
        print(by.T.to_string())
        top3 = ex.groupby(ex.index.to_period("M")).size().nlargest(3)
        print("3 busiest months:", {str(k): int(v) for k, v in top3.items()},
              f"= {top3.sum() / len(ex):.0%} of episodes")
        ex2 = ex[ex.index.year != 2020]
        lo, hi = cluster_bootstrap_ci(ex2, s["bootstrap_draws"], 0)
        print(f"excess excluding 2020: mean {ex2.mean():.4f}  CI [{lo:.4f}, {hi:.4f}]  n={len(ex2)}")
