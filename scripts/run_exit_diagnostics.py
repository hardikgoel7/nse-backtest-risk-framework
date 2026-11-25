"""Reads reports/trade_log_dev.csv. Explains WHERE the average return comes from.
1. By year   2. Share of total profit from the 10 best trades; mean without them   3. Top 5 stocks."""
import sys
import pandas as pd

TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "dev"

tr = pd.read_csv(f"reports/trade_log_{TAG}.csv", parse_dates=["signal_date"])
for name in ["A2", "BOUNCE", "RANDOM"]:
    g = tr[tr.strategy == name].copy()
    print(f"\n=== {name}: {len(g)} trades, mean net {g.net.mean() * 100:.2f}% ===")
    by = g.groupby(g.signal_date.dt.year).agg(trades=("net", "size"),
                                              target_pct=("reason", lambda x: (x == "target").mean() * 100),
                                              mean_net_pct=("net", lambda x: x.mean() * 100)).round(2)
    print(by.T.to_string())
    top10 = g.net.nlargest(10)
    print(f"10 best trades = {top10.sum() / g.net.sum():.0%} of total net profit; "
          f"mean net WITHOUT them: {g.net.drop(top10.index).mean() * 100:.2f}%")
    print("top 5 stocks by total net:", {k: round(v * 100, 1) for k, v in g.groupby('ticker').net.sum().nlargest(5).items()})
    print(f"profitable years: {(by.mean_net_pct > 0).sum()} of {len(by)}")
