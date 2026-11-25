"""Re-reports the all-day baseline comparison from SAVED files (seconds, no re-simulation):
  reports/baseline_all_<tag>.parquet  (written by run_baseline_all.py)  +  the matching signal trade log.
Fixes the multiplicity count: the family is 3 strategies x 3 baselines = 9 comparisons, so the adjusted CI
uses alpha/9 (run_baseline_all.py printed alpha/3, which was too lenient).
Usage: python scripts/run_baseline_report.py --tag base_dev [--tag base_HOLD]"""
import sys, os
sys.path.insert(0, "src")
import pandas as pd
from bt.data import load_config
from bt.event_study import cluster_bootstrap_means, ci_from_means

cfg = load_config(); s = cfg["strategy"]
tags = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--tag"] or ["base_dev"]
NCOMP = 9
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

for tag in tags:
    bpath = f"reports/baseline_all_{tag}.parquet"
    spath = "reports/trade_log_s_HOLD.csv" if "HOLD" in tag.upper() else "reports/trade_log_s_dev.csv"
    if not os.path.exists(bpath):
        print(f"(skip {tag}: {bpath} not found)"); continue
    bl = pd.read_parquet(bpath); sig = pd.read_csv(spath, parse_dates=["signal_date"])
    print(f"\n=================== {tag}  (family = {NCOMP} comparisons) ===================")
    out = []
    for name, kind in (("S1", "D"), ("S2", "D"), ("S3", "M")):
        g = sig[sig.strategy == name].copy(); b = bl[bl.kind == kind]
        cols = {"pooled": g.net - b.net.mean(),
                "stock-matched*": g.net - g.ticker.map(b.groupby("ticker").net.mean()),
                "month-matched": g.net - g.signal_date.dt.to_period("M").map(b.groupby(b.date.dt.to_period("M")).net.mean())}
        for lab, v in cols.items():
            x = pd.Series(v.values, index=g.signal_date.values).dropna()
            m = cluster_bootstrap_means(x, s["bootstrap_draws"], 0)
            lo, hi = ci_from_means(m, 0.05); lo2, hi2 = ci_from_means(m, 0.05 / NCOMP)
            out.append({"strategy": name, "baseline": lab, "n": len(x), "signal_mean%": g.net.mean() * 100,
                        "excess%": x.mean() * 100, "ci_lo%": lo * 100, "ci_hi%": hi * 100,
                        "adj_lo%": lo2 * 100, "adj_hi%": hi2 * 100, "survives": bool(lo2 > 0 or hi2 < 0)})
    print(pd.DataFrame(out).round(2).to_string(index=False))
print("\n* stock-matched = pre-declared primary. All 9 reported; none is picked.")
