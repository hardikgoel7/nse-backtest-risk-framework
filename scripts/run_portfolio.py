"""EVENT-DRIVEN PORTFOLIO ENGINE (the project's engine deliverable).

Walks the market one trading day at a time with a fixed pot of money and real constraints:
position sizing by risk, a cap on concurrent positions, a cap on position size, no leverage,
costs and slippage on every fill. Nothing in the loop may look at a future bar.

Order of events on each day d:
  1. open positions   -> check gap-through-stop (exit at open), stop, target, time cap
  2. pending orders   -> stop broken first = cancelled; gap above trigger = fill at open; else fill at trigger
  3. signals of day d -> place a buy-stop order for the next `entry_window` days (never filled on day d)
  4. mark to market   -> equity = cash + sum(qty x close)
This is an ENGINE DEMONSTRATION. The research phase found no edge, so the equity curve is NOT a
performance claim; it shows what the rule would have done with real constraints.

Usage: python scripts/run_portfolio.py --universe data/nifty500_constituents.csv [--strategy S1]
Writes: reports/portfolio_equity.csv, portfolio_trades.csv, portfolio_monthly.csv (Power BI inputs)."""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from bt.data import load_config, load_prices, read_universe, quarantine_reason
from bt.costs import CostModel
from bt.signals import features, signal_a2, signal_b, episode_starts
from bt import metrics as M


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


cfg = load_config(); s = cfg["strategy"]; P = cfg["trade"]; SP = cfg["strategies"]; PF = cfg["portfolio"]
if arg("--universe"): cfg["data"]["universe_file"] = arg("--universe")
NAME = arg("--strategy", PF["strategy"])
SPEC = SP[NAME]; WIN = SPEC["entry_window"]; HOLD = SPEC["max_hold_days"]; TR = SP["common"]["target_r"]
MIN_ADV = float(P.get("min_adv_rupees", 0)); DEV_END = pd.Timestamp(cfg["data"]["dev_end"])
cost = CostModel.from_config(cfg)
sig_cfg = dict(s, support_lo=SPEC["band"][0], support_hi=SPEC["band"][1])
if "trend_min" in SPEC: sig_cfg["trend_min"] = SPEC["trend_min"]
monthly = NAME == "S3"

# ---------------------------------------------------------------- load
data, signals = {}, []
nq = ns = 0
for tk in read_universe(cfg):
    try:
        df = load_prices(tk, cfg, allow_holdout=True)     # full history: the engine never decides anything, it only replays
    except FileNotFoundError:
        continue
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) and quarantine_reason(df): nq += 1; continue
    if len(df) < 400: ns += 1; continue
    f = features(df["Close"], s["rsi_period"]); ok = f.rsi_m.notna()
    sig = (signal_b(f, sig_cfg) if monthly else signal_a2(f, sig_cfg)) & ok
    months = df.index.to_period("M")
    d = {"idx": df.index, "pos": {t: i for i, t in enumerate(df.index)},
         "O": df["Open"].to_numpy(float), "H": df["High"].to_numpy(float),
         "L": df["Low"].to_numpy(float), "C": df["Close"].to_numpy(float),
         "adv": (df["Close"] * df["Volume"]).rolling(20).mean().to_numpy() if "Volume" in df else None,
         "mlow": df["Low"].groupby(months).min().shift(1).reindex(months).to_numpy(),
         "mhigh": df["High"].groupby(months).max().shift(1).reindex(months).to_numpy()}
    data[tk] = d
    for t in df.index.get_indexer(episode_starts(sig, s["cluster_gap_days"])):
        if d["adv"] is not None and not d["adv"][t] >= MIN_ADV: continue
        trig, stp = (d["mhigh"][t], d["mlow"][t]) if monthly else ((d["H"][t - 1], d["L"][t - 1]) if t >= 1 else (np.nan, np.nan))
        if np.isfinite(trig) and np.isfinite(stp) and stp < trig:
            signals.append((df.index[t], tk, float(trig), float(stp)))
print(f"{len(data)} tradable tickers (quarantined {nq}, too short {ns}) | {NAME} | {len(signals)} signals "
      f"| capital Rs{PF['start_capital']:,} | risk {PF['risk_per_trade']:.0%}/trade | max {PF['max_positions']} positions")

cal = pd.DatetimeIndex(sorted(set().union(*[set(d["idx"]) for d in data.values()])))
by_day = {}
for dt, tk, trig, stp in signals: by_day.setdefault(dt, []).append((tk, trig, stp))

# ---------------------------------------------------------------- event loop
RF0 = cfg["metrics"]["risk_free_annual"]
CASH_RF = PF.get("cash_earns_rf", True) and "--no-cash-interest" not in sys.argv
DAILY_RF = RF0 / cfg["metrics"]["periods_per_year"] if CASH_RF else 0.0
cash = float(PF["start_capital"]); equity = cash
open_pos, pending, trades, curve, skipped = {}, [], [], [], {"no_slot": 0, "no_cash": 0, "dup": 0, "cancelled": 0, "unfilled": 0}

def close_position(p, px, reason, dt):
    global cash
    fill = cost.fill_price(px, "sell"); proceeds = p["qty"] * fill
    cash += proceeds - cost.fees("sell", proceeds)
    gross = fill / p["entry"] - 1
    trades.append({"strategy": NAME, "ticker": p["tk"], "signal_date": p["sig_dt"], "entry_date": p["entry_dt"],
                   "exit_date": dt, "days_held": (dt - p["entry_dt"]).days, "qty": p["qty"], "entry": p["entry"],
                   "stop": p["stop"], "target": p["target"], "exit": fill, "reason": reason,
                   "risk_pct": (p["entry"] - p["stop"]) / p["entry"], "gross_pct": gross,
                   "pnl": proceeds - cost.fees("sell", proceeds) - p["cost_basis"],
                   "R": gross / ((p["entry"] - p["stop"]) / p["entry"])})

for dt in cal:
    cash *= (1 + DAILY_RF)          # idle cash accrues the risk-free rate (see config.yaml portfolio.cash_earns_rf)
    # 1. exits
    for tk in list(open_pos):
        p = open_pos[tk]; d = data[tk]; i = d["pos"].get(dt)
        if i is None: continue
        if d["O"][i] <= p["stop"]: close_position(p, d["O"][i], "stop", dt); del open_pos[tk]
        elif d["L"][i] <= p["stop"]: close_position(p, p["stop"], "stop", dt); del open_pos[tk]
        elif d["O"][i] >= p["target"]: close_position(p, d["O"][i], "target", dt); del open_pos[tk]
        elif d["H"][i] >= p["target"]: close_position(p, p["target"], "target", dt); del open_pos[tk]
        elif i - p["entry_i"] >= HOLD: close_position(p, d["C"][i], "timeout", dt); del open_pos[tk]
    # 2. pending orders
    still = []
    for o in pending:
        tk = o["tk"]; d = data[tk]; i = d["pos"].get(dt)
        if i is None or dt <= o["sig_dt"]:
            if dt <= o["expiry"]: still.append(o)
            continue
        if d["L"][i] <= o["stop"]: skipped["cancelled"] += 1; continue          # stop broken before any fill
        if d["O"][i] >= o["trigger"] or d["H"][i] >= o["trigger"]:
            px = d["O"][i] if d["O"][i] >= o["trigger"] else o["trigger"]
            entry = cost.fill_price(px, "buy")
            if tk in open_pos and PF["one_position_per_ticker"]: skipped["dup"] += 1; continue
            if len(open_pos) >= PF["max_positions"]: skipped["no_slot"] += 1; continue
            risk_ps = entry - o["stop"]
            if risk_ps <= 0: continue
            qty = int(min(equity * PF["risk_per_trade"] / risk_ps, equity * PF["max_position_pct"] / entry))
            notional = qty * entry
            if qty <= 0 or notional + cost.fees("buy", notional) > cash: skipped["no_cash"] += 1; continue
            cash -= notional + cost.fees("buy", notional)
            open_pos[tk] = {"tk": tk, "qty": qty, "entry": entry, "stop": o["stop"], "target": entry + TR * risk_ps,
                            "entry_dt": dt, "entry_i": i, "sig_dt": o["sig_dt"], "cost_basis": notional + cost.fees("buy", notional)}
            continue
        if dt < o["expiry"]: still.append(o)
        else: skipped["unfilled"] += 1
    pending = still
    # 3. new signals (known at today's close, order live from tomorrow)
    for tk, trig, stp in by_day.get(dt, []):
        i = data[tk]["pos"][dt]
        exp_i = min(i + WIN, len(data[tk]["idx"]) - 1)
        pending.append({"tk": tk, "trigger": trig, "stop": stp, "sig_dt": dt, "expiry": data[tk]["idx"][exp_i]})
    # 4. mark to market
    mtm = sum(p["qty"] * data[p["tk"]]["C"][data[p["tk"]]["pos"].get(dt, p["entry_i"])] for p in open_pos.values())
    equity = cash + mtm
    curve.append((dt, equity, cash, len(open_pos), mtm / equity if equity > 0 else 0.0))

eq = pd.DataFrame(curve, columns=["date", "equity", "cash", "open_positions", "deployed_frac"]).set_index("date")
tr = pd.DataFrame(trades)
eq["drawdown"] = eq.equity / eq.equity.cummax() - 1
ret = eq.equity.pct_change().dropna()

bench = load_prices(cfg["data"]["index_ticker"], cfg, allow_holdout=True)["Close"].reindex(eq.index).ffill()
bret = bench.pct_change().dropna()
eq["nifty"] = bench / bench.iloc[0] * PF["start_capital"]
eq.to_csv("reports/portfolio_equity.csv")
tr.to_csv("reports/portfolio_trades.csv", index=False)
mon = (1 + ret).resample("ME").prod() - 1
mon.to_frame("monthly_return").to_csv("reports/portfolio_monthly.csv")

RF = cfg["metrics"]["risk_free_annual"]
def block(r, b, label):
    r = r.dropna(); b = b.reindex(r.index).dropna(); r = r.reindex(b.index)
    ab = M.alpha_beta(r, b, RF)
    print(f"\n--- {label} ({r.index[0].date()} to {r.index[-1].date()}) ---")
    print(f"  strategy : CAGR {M.cagr(r)*100:6.2f}%  vol {M.ann_vol(r)*100:5.2f}%  MaxDD {M.max_drawdown(r)*100:7.2f}%  "
          f"Calmar {M.calmar(r):5.2f}  Sharpe {M.sharpe(r, RF):5.2f}  Sortino {M.sortino(r, RF):5.2f}")
    print(f"  Nifty 50 : CAGR {M.cagr(b)*100:6.2f}%  vol {M.ann_vol(b)*100:5.2f}%  MaxDD {M.max_drawdown(b)*100:7.2f}%  "
          f"Calmar {M.calmar(b):5.2f}  Sharpe {M.sharpe(b, RF):5.2f}  Sortino {M.sortino(b, RF):5.2f}")
    print(f"  alpha {ab['alpha_ann']*100:+.2f}%/yr (t={ab['alpha_tstat']:.2f})  beta {ab['beta']:.2f}")
    for d in (-0.01, 0.01):
        print(f"    Sharpe at rf={RF+d:.1%}: {M.sharpe(r, RF+d):.2f} (Nifty {M.sharpe(b, RF+d):.2f})", end="")
    print()

print(f"\n=== EVENT-DRIVEN PORTFOLIO, {NAME} | risk-free {RF:.1%} p.a. (see config.yaml for source) ===")
block(ret, bret, "FULL PERIOD")
block(ret[ret.index <= DEV_END], bret, "DEVELOPMENT 2010-2023")
block(ret[ret.index > DEV_END], bret, "HOLDOUT 2024-2026")
print(f"\ntrades {len(tr)} | target {(tr.reason=='target').mean()*100:.1f}%  stop {(tr.reason=='stop').mean()*100:.1f}%  "
      f"timeout {(tr.reason=='timeout').mean()*100:.1f}% | win {(tr.pnl>0).mean()*100:.1f}% | mean R {tr.R.mean():.2f}")
print(f"capital deployment: average {eq.deployed_frac.mean()*100:.1f}% of equity in the market; "
      f"{(eq.open_positions == 0).mean()*100:.1f}% of days fully in cash"
      + (f" (idle cash earns {RF0:.1%} p.a.)" if CASH_RF else " (idle cash earns 0%)"))
print(f"orders not taken: {skipped} | avg open positions {eq.open_positions.mean():.1f} | "
      f"final equity Rs{eq.equity.iloc[-1]:,.0f} vs Nifty Rs{eq.nifty.iloc[-1]:,.0f}")
print("\nwrote reports/portfolio_equity.csv, portfolio_trades.csv, portfolio_monthly.csv")
