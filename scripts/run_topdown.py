"""TOP-DOWN SECTOR LAYER: does filtering by sector strength add anything?

Rule (pre-declared, config `topdown:`): take a stock signal ONLY when its sector index has daily AND
weekly AND monthly RSI above 60 on the signal day. Weekly and monthly use the last COMPLETED bar, so the
filter is knowable at the close of the signal day.

Sector index: equal-weighted from the universe's own members of that sector (see config for why).
Sector labels come from data/sectors.csv (run scripts/fetch_sectors.py first).

WHAT IS COMPARED, and this is the whole point: splitting trades into sector-ON and sector-OFF is not
enough, because sector-ON days may simply be good days for every stock. So each group is measured
against the ALL-DAY baseline restricted to the SAME sector state. If the filter only picks good market
days, the signal's excess over that baseline will not move.

S1, S2, S3 trades are read from reports/trade_log_s_dev.csv and only TAGGED - no rule is changed and
nothing is re-simulated. S4 (the cup-and-handle proxy) is new and is simulated here.

Usage: python scripts/run_topdown.py --universe data/nifty500_constituents.csv"""
import sys, os
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from bt.data import load_config, load_prices, read_universe, quarantine_reason, _p
from bt.costs import CostModel
from bt.indicators import wilder_rsi, htf_rsi
from bt.trade_sim import simulate_breakout_trade
from bt.event_study import cluster_bootstrap_means, ci_from_means


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


cfg = load_config(); s = cfg["strategy"]; P = cfg["trade"]; SP = cfg["strategies"]; TD = cfg["topdown"]
if arg("--universe"): cfg["data"]["universe_file"] = arg("--universe")
MIN_ADV = float(P.get("min_adv_rupees", 0)); NCOMP = TD["bonferroni_comparisons"]
cost = CostModel.from_config(cfg); S4 = SP["S4"]
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)

sec_path = _p(cfg, "data/sectors.csv")
if not os.path.exists(sec_path): sys.exit("run scripts/fetch_sectors.py first")
sectors = dict(pd.read_csv(sec_path).dropna().values)

# ---------------------------------------------------------- load prices, build sector indices
closes, data = {}, {}
for tk in read_universe(cfg):
    try:
        df = load_prices(tk, cfg)
    except FileNotFoundError:
        continue
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) < 400 or quarantine_reason(df): continue
    closes[tk] = df["Close"]; data[tk] = df
print(f"{len(data)} tradable tickers | sector labels for {sum(t in sectors for t in data)}")

members = {}
for tk in data:
    members.setdefault(sectors.get(tk, "Unknown"), []).append(tk)
members = {k: v for k, v in members.items() if k not in ("Unknown", "") and len(v) >= TD["min_members"]}
print("sectors used: " + ", ".join(f"{k} ({len(v)})" for k, v in sorted(members.items())))

sector_on = {}
for sec, tks in members.items():
    rets = pd.concat([closes[t].pct_change() for t in tks], axis=1).mean(axis=1, skipna=True)
    idx = (1 + rets.fillna(0)).cumprod()
    on = ((wilder_rsi(idx, s["rsi_period"]) > TD["sector_min"])
          & (htf_rsi(idx, "W-FRI", s["rsi_period"]) > TD["sector_min"])
          & (htf_rsi(idx, "M", s["rsi_period"]) > TD["sector_min"]))
    sector_on[sec] = on
    print(f"  {sec:24s} sector-ON on {on.mean()*100:5.1f}% of days")

def tag(ticker, date):
    sec = sectors.get(ticker)
    if sec not in sector_on: return np.nan
    v = sector_on[sec].get(pd.Timestamp(date), np.nan)
    return bool(v) if v == v else np.nan

# ---------------------------------------------------------- S4: cup-and-handle proxy (new, simulated here)
p4 = dict(P, target_r=SP["common"]["target_r"], exit_rsi=None, max_hold_days=S4["max_hold_days"])
rows4 = []
for tk, df in data.items():
    O, H, L, C = (df[c].to_numpy(float) for c in ["Open", "High", "Low", "Close"])
    m = df.index.to_period("M")
    mc = df["Close"].groupby(m).last(); mh = df["High"].groupby(m).max(); ml = df["Low"].groupby(m).min()
    prior_max = mc.shift(1).rolling(S4["lookback_months"]).max()
    brk = mc >= (1 + S4["breakout_pct"]) * prior_max          # breakout month, known at that month's close
    adv = (df["Close"] * df["Volume"]).rolling(20).mean().to_numpy() if "Volume" in df else None
    first = pd.Series(np.arange(len(df)), index=df.index).groupby(m).first()
    prev = False
    for per in mc.index[S4["lookback_months"]:]:
        if not bool(brk.get(per, False)) or prev:             # only the FIRST month of a breakout run
            prev = bool(brk.get(per, False)); continue
        prev = True
        nxt = per + 1
        if nxt not in first.index: continue
        t = int(first[nxt])                                    # first trading day after the breakout month
        if adv is not None and not adv[t] >= MIN_ADV: continue
        r = simulate_breakout_trade(O, H, L, C, t - 1, float(mh[per]), float(ml[per]), S4["entry_window"], p4,
                                    cost, P["notional_rupees"])
        if r["status"] == "ok":
            rows4.append({"strategy": "S4", "ticker": tk, "signal_date": df.index[t], **r})
s4 = pd.DataFrame(rows4)
print(f"\nS4 cup-and-handle proxy: {len(s4)} trades "
      f"({S4['lookback_months']}-month base, breakout {S4['breakout_pct']:.0%} above it)")

# ---------------------------------------------------------- combine trades, tag, and score
tr = pd.read_csv("reports/trade_log_s_dev.csv", parse_dates=["signal_date"])
tr = tr[tr.strategy.isin(["S1", "S2", "S3"])]
tr = pd.concat([tr, s4], ignore_index=True)
tr["sector_on"] = [tag(t, d) for t, d in zip(tr.ticker, tr.signal_date)]
tr["net_per_day"] = tr.net / tr.days.clip(lower=1)
tr.to_csv("reports/trade_log_topdown_dev.csv", index=False)

bl = pd.read_parquet("reports/baseline_all_base_dev.parquet")
bl["sector_on"] = [tag(t, d) for t, d in zip(bl.ticker, bl.date)]
kind_of = {"S1": "D", "S2": "D", "S3": "M", "S4": "M"}

out, cmp_rows = [], []
for name in ["S1", "S2", "S3", "S4"]:
    g_all = tr[(tr.strategy == name) & tr.sector_on.notna()]
    b_all = bl[(bl.kind == kind_of[name]) & bl.sector_on.notna()]
    for state in (True, False):
        g = g_all[g_all.sector_on == state]; b = b_all[b_all.sector_on == state]
        if len(g) < 10: continue
        out.append({"strategy": name, "sector": "ON" if state else "OFF", "trades": len(g),
                    "mean_net%": g.net.mean() * 100, "median_net%": g.net.median() * 100,
                    "net_per_day%": g.net_per_day.mean() * 100, "avg_days": g.days.mean(),
                    "target%": (g.reason == "target").mean() * 100, "stop%": (g.reason == "stop").mean() * 100,
                    "win%": (g.net > 0).mean() * 100, "avg_risk%": g.risk_pct.mean() * 100,
                    "baseline_mean%": b.net.mean() * 100})
        x = pd.Series((g.net - b.net.mean()).values, index=g.signal_date.values)
        m = cluster_bootstrap_means(x, s["bootstrap_draws"], 0)
        lo, hi = ci_from_means(m, 0.05); lo2, hi2 = ci_from_means(m, 0.05 / NCOMP)
        cmp_rows.append({"strategy": name, "sector": "ON" if state else "OFF", "n": len(g),
                         "excess_vs_same_state_baseline%": x.mean() * 100, "ci_lo%": lo * 100, "ci_hi%": hi * 100,
                         "adj_lo%": lo2 * 100, "adj_hi%": hi2 * 100, "survives": bool(lo2 > 0 or hi2 < 0)})

print("\n=== Outcomes by sector state (DEV 2010-2023, net of costs; exit 3R) ===")
print(pd.DataFrame(out).round(2).to_string(index=False))
print(f"\n=== Excess vs the ALL-DAY baseline in the SAME sector state ({NCOMP} comparisons, adj = Bonferroni) ===")
print(pd.DataFrame(cmp_rows).round(2).to_string(index=False))
print("\nReading guide: if the sector filter adds skill, 'excess' is larger for sector ON than OFF.")
print("If ON and OFF excess are the same while raw mean_net is higher for ON, the filter is only picking")
print("better market days - the baseline rises with it and the signal has added nothing.")
print("\nwrote reports/trade_log_topdown_dev.csv")
