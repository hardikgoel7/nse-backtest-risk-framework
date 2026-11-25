"""Tests YOUR exact trade rules on the DEV window (2010-2023):
entry next close after signal; stop = low of the RSI-40 candle; target +30%; max 120 days.
Compares A2 and A against two controls that use the IDENTICAL stop/target/time rules:
  BOUNCE = daily bounce only (no monthly/weekly filter)   RANDOM = random days.
Writes the full trade log to reports/trade_log_dev.csv."""
import sys, os, datetime
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from bt.data import load_config, load_prices, read_universe, quarantine_reason
from bt.costs import CostModel
from bt.signals import features, signal_a, signal_a2, signal_bounce_only, episode_starts
from bt.trade_sim import simulate_trade
from bt.event_study import cluster_bootstrap_means, ci_from_means

def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

cfg = load_config(); s = cfg["strategy"]; p = dict(cfg["trade"])
if arg("--universe"): cfg["data"]["universe_file"] = arg("--universe")
if arg("--target"): p["target_pct"] = float(arg("--target"))
if arg("--exit-rsi"): p["exit_rsi"] = float(arg("--exit-rsi"))   # no price target: hold until daily RSI >= this, exit next close
if arg("--rmult"): p["target_r"] = float(arg("--rmult"))   # target = entry + rmult x (entry - stop)
TAG = arg("--tag", "dev")
HOLD = "--holdout" in sys.argv        # OUT-OF-SAMPLE run: signals only after dev_end. Allowed ONCE (lock file).
LOCK = "reports/HOLDOUT_DONE.txt"
if HOLD and os.path.exists(LOCK) and "--force" not in sys.argv:
    sys.exit(f"REFUSED: holdout already used ({LOCK}). Re-running it = peeking. Details:\n" + open(LOCK).read())
DEV_END = pd.Timestamp(cfg["data"]["dev_end"])
MIN_ADV = float(p.get("min_adv_rupees", 0))   # liquidity floor, pre-declared in config
cost = CostModel.from_config(cfg)
rng = np.random.default_rng(0)
rows, counts = [], {}
tickers = read_universe(cfg)
if p.get("exit_rsi") is not None: tgt = f"RSI>={p['exit_rsi']:g} exit"
elif p.get("target_r"): tgt = f"{p['target_r']:.1f}R"
else: tgt = f"{p['target_pct']:.0%}"
print(f"{len(tickers)} tickers | min ADV Rs{MIN_ADV/1e7:.1f} cr | target {tgt} | max hold {p['max_hold_days']}d")

for tk in tickers:
    try:
        df = load_prices(tk, cfg, allow_holdout=HOLD)
    except FileNotFoundError:
        continue
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) and quarantine_reason(df):
        counts[("ALL", "quarantined")] = counts.get(("ALL", "quarantined"), 0) + 1; continue
    if len(df) < 400:      # too short for monthly RSI + any trade (e.g. recent listings)
        counts[("ALL", "short_history")] = counts.get(("ALL", "short_history"), 0) + 1; continue
    if "Volume" not in df and tk == tickers[0]: print("WARNING: no Volume column, liquidity floor NOT applied")
    adv = (df["Close"] * df["Volume"]).rolling(20).mean().to_numpy() if "Volume" in df else None
    O, H, L, C = (df[c].to_numpy(float) for c in ["Open", "High", "Low", "Close"])
    f = features(df["Close"], s["rsi_period"]); rsi_d = f.rsi_d.to_numpy()
    ok = f.rsi_m.notna()
    plans = {"A2": (signal_a2, 1), "A": (signal_a, 0), "BOUNCE": (signal_bounce_only, 1)}
    for name, (fn, lag) in plans.items():
        sig = fn(f, s) & ok
        if HOLD: sig = sig & (sig.index > DEV_END)      # only signals AFTER the dev window; history before is warm-up
        starts = episode_starts(sig, s["cluster_gap_days"])
        for t in df.index.get_indexer(starts):
            if adv is not None and not adv[t] >= MIN_ADV:
                counts[(name, "illiquid")] = counts.get((name, "illiquid"), 0) + 1; continue
            r = simulate_trade(O, H, L, C, t, lag, p, cost, p["notional_rupees"], rsi=rsi_d)
            counts[(name, r["status"])] = counts.get((name, r["status"]), 0) + 1
            if r["status"] == "ok":
                rows.append({"strategy": name, "ticker": tk, "signal_date": df.index[t], **r})
    valid = np.flatnonzero((ok & (df.index > DEV_END if HOLD else True)).to_numpy())
    pick = rng.choice(valid, size=min(p["baseline_samples_per_stock"], len(valid)), replace=False)
    for t in pick:
        if adv is not None and not adv[int(t)] >= MIN_ADV:
            counts[("RANDOM", "illiquid")] = counts.get(("RANDOM", "illiquid"), 0) + 1; continue
        r = simulate_trade(O, H, L, C, int(t), 1, p, cost, p["notional_rupees"], rsi=rsi_d)
        counts[("RANDOM", r["status"])] = counts.get(("RANDOM", r["status"]), 0) + 1
        if r["status"] == "ok":
            rows.append({"strategy": "RANDOM", "ticker": tk, "signal_date": df.index[t], **r})

tr = pd.DataFrame(rows)
tr.to_csv(f"reports/trade_log_{TAG}.csv", index=False)

out = []
for name, g in tr.groupby("strategy", sort=False):
    risk = g.risk_pct.mean()
    lo, hi = ci_from_means(cluster_bootstrap_means(g.set_index("signal_date").net, s["bootstrap_draws"], 0))
    out.append({"strategy": name, "trades": len(g),
                "skipped": counts.get((name, "skipped"), 0), "censored": counts.get((name, "censored"), 0),
                "target%": (g.reason == "target").mean() * 100, "stop%": (g.reason == "stop").mean() * 100,
                "timeout%": (g.reason == "timeout").mean() * 100, "rsi_exit%": (g.reason == "rsi").mean() * 100,
                "win%": (g.net > 0).mean() * 100, "trades>=3R%": (g.R >= 3).mean() * 100, "trades>=1R%": (g.R >= 1).mean() * 100,
                "avg_risk%": risk * 100, "breakeven_target%": (np.nan if p.get("exit_rsi") is not None else (1 / (1 + p["target_r"]) if p.get("target_r") else risk / (p["target_pct"] + risk)) * 100),
                "mean_net%": g.net.mean() * 100, "ci_lo%": lo * 100, "ci_hi%": hi * 100,
                "median_net%": g.net.median() * 100, "mean_R": g.R.mean(), "avg_days": g.days.mean()})
res = pd.DataFrame(out)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("\n=== Outcomes per strategy (net of costs and slippage) ===")
print(res.round(2).to_string(index=False))

print("\n=== Excess net return per trade vs controls (4 comparisons per run, adjusted CI = Bonferroni) ===")
cmp_rows = []
for a in ["A2", "A"]:
    for b in ["BOUNCE", "RANDOM"]:
        ga = tr[tr.strategy == a].set_index("signal_date").net
        ex = ga - tr[tr.strategy == b].net.mean()
        m = cluster_bootstrap_means(ex, s["bootstrap_draws"], 0)
        lo, hi = ci_from_means(m, 0.05); lo2, hi2 = ci_from_means(m, 0.05 / 4)
        cmp_rows.append({"strategy": a, "vs": b, "excess_mean%": ex.mean() * 100, "ci_lo%": lo * 100,
                         "ci_hi%": hi * 100, "adj_lo%": lo2 * 100, "adj_hi%": hi2 * 100,
                         "survives_adjustment": bool(lo2 > 0 or hi2 < 0)})
print(pd.DataFrame(cmp_rows).round(2).to_string(index=False))
print(f"\nsaved reports/trade_log_{TAG}.csv | illiquid trades dropped: " + str({k[0]: v for k, v in counts.items() if k[1] == "illiquid"})
      + f" | tickers quarantined: {counts.get(('ALL', 'quarantined'), 0)}, too short: {counts.get(('ALL', 'short_history'), 0)}")

if HOLD:
    with open(LOCK, "w") as fh:
        fh.write(f"holdout run on {datetime.datetime.now():%Y-%m-%d %H:%M} | args: {' '.join(sys.argv[1:])}\n")
    print(f"HOLDOUT lock written to {LOCK}. Do not run again.")
