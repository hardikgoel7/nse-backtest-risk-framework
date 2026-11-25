"""UPGRADED BASELINE (2026-11-22): instead of 400 random days, use EVERY eligible day
(daily structure S1/S2: every day; monthly structure S3: the first trading day of every month) with the IDENTICAL
trade mechanics (breakout entry, stop = candle low, 3R, window, hold cap, costs). Strategy trades are READ from the saved
logs (reports/trade_log_s_dev.csv or s_HOLD.csv): rules and trades are unchanged; only the yardstick is upgraded.
Three baselines, all reported, none picked:
  pooled        = mean of ALL-day trades (all stocks, whole period)
  stock-matched = each signal trade minus the all-day mean of the SAME stock   [declared primary]
  month-matched = each signal trade minus the all-day mean of ALL stocks in the SAME calendar month
Also prints the all-day baseline by year = how much of the result is just an up- or down-trending market.
Usage: python scripts/run_baseline_all.py --universe data/nifty500_constituents.csv --tag base_dev
       python scripts/run_baseline_all.py --universe ... --holdout --tag base_HOLD   (holdout data re-used ONLY for the yardstick)"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from bt.data import load_config, load_prices, read_universe, quarantine_reason
from bt.costs import CostModel
from bt.signals import features
from bt.trade_sim import simulate_breakout_trade
from bt.event_study import cluster_bootstrap_means, ci_from_means


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


cfg = load_config(); s = cfg["strategy"]; P = cfg["trade"]; SP = cfg["strategies"]
if arg("--universe"): cfg["data"]["universe_file"] = arg("--universe")
TAG = arg("--tag", "base_dev"); HOLD = "--holdout" in sys.argv
SIGLOG = "reports/trade_log_s_HOLD.csv" if HOLD else "reports/trade_log_s_dev.csv"
DEV_END = pd.Timestamp(cfg["data"]["dev_end"]); MIN_ADV = float(P.get("min_adv_rupees", 0))
cost = CostModel.from_config(cfg)
base = dict(P, target_r=SP["common"]["target_r"], exit_rsi=None)
D = dict(base, max_hold_days=SP["S1"]["max_hold_days"]); WD = SP["S1"]["entry_window"]       # S1 and S2 share this structure
M = dict(base, max_hold_days=SP["S3"]["max_hold_days"]); WM = SP["S3"]["entry_window"]
rows = []; nq = ns = 0
tickers = read_universe(cfg)
print(f"{len(tickers)} tickers | {'HOLDOUT' if HOLD else 'DEV'} | all-day baseline (this takes a few minutes)")
for k, tk in enumerate(tickers):
    try:
        df = load_prices(tk, cfg, allow_holdout=HOLD)
    except FileNotFoundError:
        continue
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) and quarantine_reason(df): nq += 1; continue
    if len(df) < 400: ns += 1; continue
    O, H, L, C = (df[c].to_numpy(float) for c in ["Open", "High", "Low", "Close"])
    months = df.index.to_period("M")
    mlow = df["Low"].groupby(months).min().shift(1).reindex(months).to_numpy()
    mhigh = df["High"].groupby(months).max().shift(1).reindex(months).to_numpy()
    adv = (df["Close"] * df["Volume"]).rolling(20).mean().to_numpy() if "Volume" in df else None
    ok = features(df["Close"], s["rsi_period"]).rsi_m.notna().to_numpy()
    after = (df.index > DEV_END) if HOLD else np.ones(len(df), bool)
    first_of_month = pd.Series(np.arange(len(df)), index=df.index).groupby(months).first().to_numpy()
    first_set = set(first_of_month.tolist())
    for t in np.flatnonzero(ok & after):
        if adv is not None and not adv[t] >= MIN_ADV: continue
        if t >= 1:
            r = simulate_breakout_trade(O, H, L, C, t, H[t - 1], L[t - 1], WD, D, cost, P["notional_rupees"])
            if r["status"] == "ok": rows.append((tk, df.index[t], "D", r["net"]))
        if t in first_set:
            r = simulate_breakout_trade(O, H, L, C, t, mhigh[t], mlow[t], WM, M, cost, P["notional_rupees"])
            if r["status"] == "ok": rows.append((tk, df.index[t], "M", r["net"]))
    if (k + 1) % 100 == 0: print(f"  {k + 1}/{len(tickers)} stocks done", flush=True)

bl = pd.DataFrame(rows, columns=["ticker", "date", "kind", "net"])
bl.to_parquet(f"reports/baseline_all_{TAG}.parquet")
sig = pd.read_csv(SIGLOG, parse_dates=["signal_date"])
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(f"\nquarantined {nq}, too short {ns} | baseline trades: daily structure {int((bl.kind == 'D').sum())}, monthly structure {int((bl.kind == 'M').sum())}")

print("\n=== ALL-DAY BASELINE (what this trade structure earns on ANY day = market trend + structure; no RSI) ===")
for kind, lab in (("D", "daily structure (15d)"), ("M", "monthly structure (63d)")):
    g = bl[bl.kind == kind]
    print(f"{lab}: mean {g.net.mean()*100:+.2f}%  median {g.net.median()*100:+.2f}%  n={len(g)}")
    by = g.groupby(g.date.dt.year).net.agg(["mean", "size"]); by["mean"] *= 100
    print("  by year (mean net %):", "  ".join(f"{y}:{m:+.2f}" for y, m in by["mean"].items()))

print("\n=== SIGNAL vs three baselines (mean net per trade, %; 95% month-clustered CI; 3 strategies = 3 primary comparisons, adj = Bonferroni/3) ===")
out = []
for name, kind in (("S1", "D"), ("S2", "D"), ("S3", "M")):
    g = sig[sig.strategy == name].copy()
    b = bl[bl.kind == kind]
    pooled = b.net.mean()
    smean = b.groupby("ticker").net.mean()
    mmean = b.groupby(b.date.dt.to_period("M")).net.mean()
    g["ex_pooled"] = g.net - pooled
    g["ex_stock"] = g.net - g.ticker.map(smean)
    g["ex_month"] = g.net - g.signal_date.dt.to_period("M").map(mmean)
    for col, lab in (("ex_pooled", "pooled"), ("ex_stock", "stock-matched*"), ("ex_month", "month-matched")):
        x = g.dropna(subset=[col]).set_index("signal_date")[col]
        m = cluster_bootstrap_means(x, s["bootstrap_draws"], 0)
        lo, hi = ci_from_means(m, 0.05); lo2, hi2 = ci_from_means(m, 0.05 / 3)
        out.append({"strategy": name, "baseline": lab, "n": len(x), "signal_mean%": g.net.mean() * 100,
                    "excess%": x.mean() * 100, "ci_lo%": lo * 100, "ci_hi%": hi * 100,
                    "adj_lo%": lo2 * 100, "adj_hi%": hi2 * 100, "survives": bool(lo2 > 0 or hi2 < 0)})
print(pd.DataFrame(out).round(2).to_string(index=False))
print("* stock-matched = declared primary. Old 400-random-day control result stays in the log; this replaces it as the yardstick.")
