"""REPORT-ONLY robustness check on SAVED trade logs (no data download, nothing re-simulated, nothing tuned).
Question: is 'signal vs random' still ~0 when outliers are removed?
For each strategy and its control: n, mean, median, mean after trimming the top and bottom 1% of trades,
share of total profit made by the top 1% of trades, and the signal-minus-control gap for mean, median, trimmed mean
(month-clustered bootstrap 95% CI for the trimmed-mean gap).
Usage: python scripts/run_robustness.py   (reads reports/trade_log_*.csv that exist)"""
import os
import numpy as np, pandas as pd

LOGS = {  # file tag -> list of (strategy, control)
    "n500_t30": [("A2", "RANDOM"), ("A2", "BOUNCE")],
    "n500_t30_HOLDOUT": [("A2", "RANDOM"), ("A2", "BOUNCE")],
    "s_dev": [("S1", "RANDOM_15d"), ("S1", "BOUNCE_S1"), ("S2", "RANDOM_15d"), ("S2", "BOUNCE_S2"), ("S3", "RANDOM_63d_monthstop")],
    "s_HOLD": [("S1", "RANDOM_15d"), ("S1", "BOUNCE_S1"), ("S2", "RANDOM_15d"), ("S2", "BOUNCE_S2"), ("S3", "RANDOM_63d_monthstop")],
}
TRIM, DRAWS = 0.01, 2000
rng = np.random.default_rng(0)


def tmean(x, p=TRIM):
    x = np.sort(np.asarray(x)); k = int(len(x) * p)
    return x[k:len(x) - k].mean() if len(x) > 2 * k + 1 else x.mean()


def top_share(x, p=TRIM):
    x = np.sort(np.asarray(x))[::-1]; k = max(1, int(len(x) * p)); tot = x.sum()
    return np.nan if tot <= 0 else x[:k].sum() / tot


def boot_gap(sig_df, ctrl_net):
    """95% CI of (trimmed mean of signal - trimmed mean of control), resampling signal trades by calendar month."""
    groups = [g.net.to_numpy() for _, g in sig_df.groupby(sig_df.signal_date.dt.to_period("M"))]
    c = tmean(ctrl_net); out = []
    for _ in range(DRAWS):
        pick = rng.integers(0, len(groups), len(groups))
        out.append(tmean(np.concatenate([groups[i] for i in pick])) - c)
    return np.percentile(out, [2.5, 97.5])


for tag, pairs in LOGS.items():
    path = f"reports/trade_log_{tag}.csv"
    if not os.path.exists(path):
        print(f"\n(skip {tag}: {path} not found)"); continue
    tr = pd.read_csv(path, parse_dates=["signal_date"])
    print(f"\n=================== {tag} ===================")
    rows = []
    for a, b in pairs:
        if a not in set(tr.strategy) or b not in set(tr.strategy):
            continue
        ga, gb = tr[tr.strategy == a], tr[tr.strategy == b]
        lo, hi = boot_gap(ga, gb.net.to_numpy())
        rows.append({"strategy": a, "vs": b, "n": len(ga),
                     "mean%": ga.net.mean() * 100, "median%": ga.net.median() * 100, "trim1%_mean%": tmean(ga.net) * 100,
                     "top1%_profit_share": top_share(ga.net),
                     "ctrl_mean%": gb.net.mean() * 100, "ctrl_median%": gb.net.median() * 100, "ctrl_trim_mean%": tmean(gb.net) * 100,
                     "gap_mean%": (ga.net.mean() - gb.net.mean()) * 100, "gap_median%": (ga.net.median() - gb.net.median()) * 100,
                     "gap_trim%": (tmean(ga.net) - tmean(gb.net)) * 100, "gap_trim_lo%": lo * 100, "gap_trim_hi%": hi * 100})
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
print("\nReading guide: if gap_trim% stays near 0 and its CI spans 0, outliers were not hiding or creating an edge.")
