"""Runs strategies A, A2, B on the DEV window (2010-2023) only. Prints and saves results.
All 3 strategies x all horizons are reported. 'adj_ci' is a stricter interval that accounts for
having made many comparisons (Bonferroni)."""
import sys
sys.path.insert(0, "src")
import pandas as pd
from bt.data import load_config, load_prices, read_universe
from bt.costs import CostModel
from bt.signals import features, signal_a, signal_a2, signal_b, episode_starts
from bt.event_study import summarize

cfg = load_config()
s = cfg["strategy"]
rt = CostModel.from_config(cfg).round_trip_bps(1e5)
closes = {}
for tk in read_universe(cfg):
    try:
        closes[tk] = load_prices(tk, cfg)["Close"].dropna()
    except FileNotFoundError:
        print("MISSING", tk)

strategies = [("A", signal_a), ("A2", signal_a2), ("B", signal_b)]
n_comp = len(strategies) * len(s["horizons"])
print(f"{len(closes)} stocks | round-trip cost {rt:.1f} bps | {n_comp} comparisons: "
      f"expect ~{0.05 * n_comp:.1f} 'significant' results by pure luck at 95%")

rows, logs = [], []
for name, fn in strategies:
    for h in s["horizons"]:
        r = summarize(closes, fn, s, h, rt, n_comparisons=n_comp)
        rows.append({"strategy": name, "primary": h in s["primary_horizons"], **r})
    for tk, c in closes.items():
        f = features(c, s["rsi_period"])
        for d in episode_starts(fn(f, s) & f.rsi_m.notna(), s["cluster_gap_days"]):
            logs.append({"strategy": name, "ticker": tk, "signal_date": d})

res = pd.DataFrame(rows)
res.to_csv("reports/event_study_dev.csv", index=False)
pd.DataFrame(logs).to_csv("reports/signal_log_dev.csv", index=False)

show = res[["strategy", "horizon", "primary", "episodes", "baseline_mean", "mean_gross", "hit_rate",
            "baseline_hit", "excess_mean", "excess_ci_lo", "excess_ci_hi", "survives_adjustment"]]
pd.set_option("display.width", 250)
for name, g in show.groupby("strategy", sort=False):
    print(f"\n--- Strategy {name} ---")
    print(g.drop(columns="strategy").round(4).to_string(index=False))
print("\nsaved reports/event_study_dev.csv (full columns) and reports/signal_log_dev.csv")
