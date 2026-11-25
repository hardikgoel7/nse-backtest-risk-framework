"""ENTRY = buy-stop at the signal candle HIGH, STOP = its LOW, TARGET = fill + 3 x risk (see trade_sim.simulate_breakout_trade).
Runs the pre-declared strategies S1, S2, S3 (config.yaml `strategies:`) with exit = 3x risk.
Controls under the IDENTICAL stop/target/hold: RANDOM (random days), BOUNCE (daily bounce, no monthly/weekly filter, S1/S2 only).
Usage:  python scripts/run_strategies.py --universe data/nifty500_constituents.csv --tag s_dev
        python scripts/run_strategies.py --universe ... --holdout --tag s_HOLD     (ONCE; lock file)"""
import sys, os, datetime
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from bt.data import load_config, load_prices, read_universe, quarantine_reason
from bt.costs import CostModel
from bt.signals import features, signal_a2, signal_b, signal_bounce_only, episode_starts
from bt.trade_sim import simulate_breakout_trade
from bt.event_study import cluster_bootstrap_means, ci_from_means


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


cfg = load_config(); s = cfg["strategy"]; P = cfg["trade"]; SP = cfg["strategies"]
if arg("--universe"): cfg["data"]["universe_file"] = arg("--universe")
TAG = arg("--tag", "s_dev"); HOLD = "--holdout" in sys.argv
LOCK = "reports/HOLDOUT_DONE_strategies.txt"
if HOLD and os.path.exists(LOCK) and "--force" not in sys.argv:
    sys.exit("REFUSED: strategy holdout already used.\n" + open(LOCK).read())
DEV_END = pd.Timestamp(cfg["data"]["dev_end"]); MIN_ADV = float(P.get("min_adv_rupees", 0))
cost = CostModel.from_config(cfg); rng = np.random.default_rng(0)
NCOMP = SP["bonferroni_comparisons"]
base = dict(P, target_r=SP["common"]["target_r"], exit_rsi=None)
hold = {k: SP[k]["max_hold_days"] for k in ("S1", "S2", "S3")}
win = {k: SP[k]["entry_window"] for k in ("S1", "S2", "S3")}
s1 = dict(s, trend_min=SP["S1"]["trend_min"], support_lo=SP["S1"]["band"][0], support_hi=SP["S1"]["band"][1])
s2 = dict(s, trend_min=SP["S2"]["trend_min"], support_lo=SP["S2"]["band"][0], support_hi=SP["S2"]["band"][1])
s3 = dict(s, support_lo=SP["S3"]["band"][0], support_hi=SP["S3"]["band"][1])
rows, counts = [], {}
tickers = read_universe(cfg)
print(f"{len(tickers)} tickers | exit {base['target_r']:.0f}R | holds {hold} | {'HOLDOUT' if HOLD else 'DEV'}")


def trade(name, tk, df, arrs, t, hmax, month_candle, wnd, adv):
    O, H, L, C, mlow, mhigh = arrs
    if adv is not None and not adv[t] >= MIN_ADV:
        counts[(name, "illiquid")] = counts.get((name, "illiquid"), 0) + 1; return
    p = dict(base, max_hold_days=hmax)
    trig, stp = (mhigh[t], mlow[t]) if month_candle else ((H[t - 1], L[t - 1]) if t >= 1 else (np.nan, np.nan))
    r = simulate_breakout_trade(O, H, L, C, t, trig, stp, wnd, p, cost, P["notional_rupees"])
    counts[(name, r["status"])] = counts.get((name, r["status"]), 0) + 1
    if r["status"] == "ok":
        rows.append({"strategy": name, "ticker": tk, "signal_date": df.index[t], **r})


for tk in tickers:
    try:
        df = load_prices(tk, cfg, allow_holdout=HOLD)
    except FileNotFoundError:
        continue
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) and quarantine_reason(df):
        counts[("ALL", "quarantined")] = counts.get(("ALL", "quarantined"), 0) + 1; continue
    if len(df) < 400:
        counts[("ALL", "short_history")] = counts.get(("ALL", "short_history"), 0) + 1; continue
    O, H, L, C = (df[c].to_numpy(float) for c in ["Open", "High", "Low", "Close"])
    months = df.index.to_period("M")
    mlow = df["Low"].groupby(months).min().shift(1).reindex(months).to_numpy()   # low of last COMPLETED month
    mhigh = df["High"].groupby(months).max().shift(1).reindex(months).to_numpy()  # high of last COMPLETED month
    arrs = (O, H, L, C, mlow, mhigh)
    adv = (df["Close"] * df["Volume"]).rolling(20).mean().to_numpy() if "Volume" in df else None
    f = features(df["Close"], s["rsi_period"]); ok = f.rsi_m.notna()
    after = (df.index > DEV_END) if HOLD else np.ones(len(df), bool)
    plans = [("S1", signal_a2(f, s1), "S1", False), ("S2", signal_a2(f, s2), "S2", False),
             ("S3", signal_b(f, s3), "S3", True),
             ("BOUNCE_S1", signal_bounce_only(f, s1), "S1", False),
             ("BOUNCE_S2", signal_bounce_only(f, s2), "S2", False)]
    for name, sig, key, mcandle in plans:
        sig = sig & ok & after
        for t in df.index.get_indexer(episode_starts(sig, s["cluster_gap_days"])):
            trade(name, tk, df, arrs, t, hold[key], mcandle, win[key], adv)
    valid = np.flatnonzero((ok & after).to_numpy())
    pick = rng.choice(valid, size=min(P["baseline_samples_per_stock"], len(valid)), replace=False)
    for t in pick:
        trade("RANDOM_15d", tk, df, arrs, int(t), hold["S1"], False, win["S1"], adv)
        trade("RANDOM_63d_monthstop", tk, df, arrs, int(t), hold["S3"], True, win["S3"], adv)

tr = pd.DataFrame(rows); tr.to_csv(f"reports/trade_log_{TAG}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
out = []
for name, g in tr.groupby("strategy", sort=False):
    risk = g.risk_pct.mean()
    lo, hi = ci_from_means(cluster_bootstrap_means(g.set_index("signal_date").net, s["bootstrap_draws"], 0))
    out.append({"strategy": name, "trades": len(g), "skipped": counts.get((name, "skipped"), 0),
                "unfilled": counts.get((name, "unfilled"), 0), "invalid": counts.get((name, "invalidated"), 0),
                "censored": counts.get((name, "censored"), 0), "target%": (g.reason == "target").mean() * 100,
                "stop%": (g.reason == "stop").mean() * 100, "timeout%": (g.reason == "timeout").mean() * 100,
                "win%": (g.net > 0).mean() * 100, "avg_risk%": risk * 100, "breakeven_hit%": 25.0,
                "mean_net%": g.net.mean() * 100, "ci_lo%": lo * 100, "ci_hi%": hi * 100,
                "median_net%": g.net.median() * 100, "avg_days": g.days.mean()})
print("\n=== Outcomes (net of costs and slippage; exit = 3R from the actual fill) ===")
print(pd.DataFrame(out).round(2).to_string(index=False))
pairs = [("S1", "RANDOM_15d"), ("S1", "BOUNCE_S1"), ("S2", "RANDOM_15d"), ("S2", "BOUNCE_S2"), ("S3", "RANDOM_63d_monthstop")]
print(f"\n=== Excess net return per trade vs controls ({NCOMP} comparisons, adjusted CI = Bonferroni) ===")
cr = []
for a, b in pairs:
    if a not in set(tr.strategy) or b not in set(tr.strategy): continue
    ga = tr[tr.strategy == a].set_index("signal_date").net
    ex = ga - tr[tr.strategy == b].net.mean()
    m = cluster_bootstrap_means(ex, s["bootstrap_draws"], 0)
    lo, hi = ci_from_means(m, 0.05); lo2, hi2 = ci_from_means(m, 0.05 / NCOMP)
    cr.append({"strategy": a, "vs": b, "excess_mean%": ex.mean() * 100, "ci_lo%": lo * 100, "ci_hi%": hi * 100,
               "adj_lo%": lo2 * 100, "adj_hi%": hi2 * 100, "survives_adjustment": bool(lo2 > 0 or hi2 < 0)})
print(pd.DataFrame(cr).round(2).to_string(index=False))
print(f"\nsaved reports/trade_log_{TAG}.csv | quarantined {counts.get(('ALL','quarantined'),0)}, too short {counts.get(('ALL','short_history'),0)}"
      f" | illiquid dropped { {k[0]: v for k, v in counts.items() if k[1] == 'illiquid'} }")
if HOLD:
    open(LOCK, "w").write(f"strategy holdout run {datetime.datetime.now():%Y-%m-%d %H:%M} | args: {' '.join(sys.argv[1:])}\n")
    print(f"HOLDOUT lock written to {LOCK}. Do not run again.")
