"""EGARCH(1,1) VOLATILITY REGIMES + STOP-DISTANCE ANALYSIS.

Why EGARCH and not GARCH: the asymmetry term lets a fall raise tomorrow's volatility more than an
equal rise does (the leverage effect). These rules buy dips, so entries land exactly where that term
matters, which is a testable reason for the high stop-out rate.

NO LOOK-AHEAD: parameters are re-estimated once a year on data up to 31 December of the previous year
(`last_obs`), and the volatility path inside each year is filtered forward from realised returns only.
A 2015 trade never sees a parameter fitted on 2016 data. The calm/turbulent cut is the EXPANDING median
of past forecasts, so it too uses only history.

DESCRIPTIVE, NOT A NEW TEST: both holdouts are already spent. This explains results that exist; it does
not re-test the rules, and no threshold here is tuned.

Part 1  index regimes  : EGARCH on the Nifty 50 -> daily conditional volatility -> calm / turbulent days.
                         Strategy and control trades are split by the regime of their signal day.
Part 2  stop distance  : each trade's stop distance measured in units of its stock's 20-day realised
                         daily volatility, then outcomes by bucket. Realised vol is used at stock level
                         because refitting EGARCH for 454 stocks x 17 years buys little for the cost.

Usage: python scripts/run_volatility.py --universe data/nifty500_constituents.csv
Writes reports/vol_regime.csv (date, sigma_annual_pct, regime) for Power BI."""
import sys, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from bt.data import load_config, load_prices, read_universe, quarantine_reason

try:
    from arch import arch_model
except ImportError:
    sys.exit("pip install arch")


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


cfg = load_config()
if arg("--universe"): cfg["data"]["universe_file"] = arg("--universe")
DEV_END = pd.Timestamp(cfg["data"]["dev_end"])
FIRST_FIT_YEAR = 2012          # first 2 years are the initial estimation sample
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

# ------------------------------------------------------------------ Part 1: index EGARCH
px = load_prices(cfg["data"]["index_ticker"], cfg, allow_holdout=True)["Close"]
r = (100 * np.log(px / px.shift(1))).dropna()          # percent log returns: arch prefers this scale
print(f"Nifty 50: {len(r)} daily returns, {r.index[0].date()} to {r.index[-1].date()}")

pieces, params = [], []
for y in range(FIRST_FIT_YEAR, r.index[-1].year + 1):
    cut = pd.Timestamp(f"{y}-01-01")
    past = r[r.index < cut]
    if len(past) < 400: continue
    res = arch_model(r, vol="EGARCH", p=1, o=1, q=1, dist="t").fit(last_obs=cut, disp="off")
    f = res.forecast(start=cut, horizon=1, reindex=False).variance.iloc[:, 0]
    f = f[(f.index >= cut) & (f.index < pd.Timestamp(f"{y + 1}-01-01"))]
    pieces.append(np.sqrt(f))
    p = res.params
    params.append({"fit_through": str(past.index[-1].date()), "n_obs": len(past),
                   "omega": p.get("omega"), "alpha": p.get("alpha[1]"), "gamma": p.get("gamma[1]"),
                   "beta": p.get("beta[1]"), "nu": p.get("nu")})

sig = pd.concat(pieces).sort_index()                    # one-step-ahead daily sigma, in percent
sig = sig[~sig.index.duplicated()]
print("\n=== EGARCH(1,1) with asymmetry, refit every January on past data only ===")
pr = pd.DataFrame(params).round(4)
print(pr.head(3).to_string(index=False)); print("  ...")
print(pr.tail(2).to_string(index=False))
g = pr.gamma.mean()
verdict = ("a fall raises tomorrow's volatility MORE than an equal rise: leverage effect present" if g < -0.01
           else "a rise raises tomorrow's volatility more than an equal fall: unusual, check the fits" if g > 0.01
           else "no meaningful asymmetry; EGARCH adds little over plain GARCH here")
print(f"\nmean asymmetry term gamma = {g:+.4f}  ->  {verdict}")
print(f"mean persistence beta = {pr.beta.mean():.3f} "
      f"({'high: volatility shocks decay slowly, so regimes last' if pr.beta.mean() > 0.9 else 'low: little volatility clustering in this sample'})")

med = sig.expanding(min_periods=250).median()           # past-only cut, never a full-sample quantile
regime = pd.Series(np.where(sig > med, "turbulent", "calm"), index=sig.index).where(med.notna())
out = pd.DataFrame({"sigma_daily_pct": sig, "sigma_annual_pct": sig * np.sqrt(252), "regime": regime}).dropna()
out.to_csv("reports/vol_regime.csv")
print(f"\nregime days: calm {(out.regime == 'calm').sum()}, turbulent {(out.regime == 'turbulent').sum()}"
      f" | annualised sigma: calm median {out[out.regime=='calm'].sigma_annual_pct.median():.1f}%, "
      f"turbulent median {out[out.regime=='turbulent'].sigma_annual_pct.median():.1f}%")

# ------------------------------------------------------------------ trades by regime
reg = out.regime
rows = []
for tag, label in (("s_dev", "DEV 2010-2023"), ("s_HOLD", "HOLDOUT 2024-2026")):
    try:
        tr = pd.read_csv(f"reports/trade_log_{tag}.csv", parse_dates=["signal_date"])
    except FileNotFoundError:
        continue
    tr["regime"] = tr.signal_date.map(reg)
    for name in ["S1", "S2", "S3", "RANDOM_15d", "RANDOM_63d_monthstop"]:
        g2 = tr[(tr.strategy == name) & tr.regime.notna()]
        for rg, gg in g2.groupby("regime"):
            if len(gg) < 10: continue
            rows.append({"window": label, "strategy": name, "regime": rg, "trades": len(gg),
                         "mean_net%": gg.net.mean() * 100, "median_net%": gg.net.median() * 100,
                         "target%": (gg.reason == "target").mean() * 100, "stop%": (gg.reason == "stop").mean() * 100,
                         "win%": (gg.net > 0).mean() * 100})
print("\n=== Trade outcomes by index volatility regime (descriptive; both holdouts already spent) ===")
reg_tbl = pd.DataFrame(rows)
print(reg_tbl.round(2).to_string(index=False))
for w in reg_tbl.window.unique():
    sub = reg_tbl[reg_tbl.window == w]
    for sname, cname in (("S1", "RANDOM_15d"), ("S2", "RANDOM_15d"), ("S3", "RANDOM_63d_monthstop")):
        for rg in ("calm", "turbulent"):
            a = sub[(sub.strategy == sname) & (sub.regime == rg)]
            b = sub[(sub.strategy == cname) & (sub.regime == rg)]
            if len(a) and len(b):
                print(f"  {w:18s} {sname} minus {cname:21s} in {rg:9s}: "
                      f"{a['mean_net%'].iloc[0] - b['mean_net%'].iloc[0]:+.2f}%")

# ------------------------------------------------------------------ Part 2: stop distance vs stock volatility
print("\n=== Stop distance in units of the stock's own 20-day realised daily volatility ===")
vol_map = {}
for tk in read_universe(cfg):
    try:
        df = load_prices(tk, cfg, allow_holdout=True)
    except FileNotFoundError:
        continue
    df = df.dropna(subset=["Close"])
    if len(df) < 400 or quarantine_reason(df): continue
    v = np.log(df["Close"] / df["Close"].shift(1)).rolling(20).std()
    vol_map[tk] = v
print(f"realised volatility computed for {len(vol_map)} stocks")

for tag, label in (("s_dev", "DEV 2010-2023"), ("s_HOLD", "HOLDOUT 2024-2026")):
    try:
        tr = pd.read_csv(f"reports/trade_log_{tag}.csv", parse_dates=["signal_date"])
    except FileNotFoundError:
        continue
    tr = tr[tr.strategy.isin(["S1", "S2"])].copy()
    tr["sigma"] = [vol_map[t].get(d, np.nan) if t in vol_map else np.nan
                   for t, d in zip(tr.ticker, tr.signal_date)]
    tr = tr.dropna(subset=["sigma"])
    tr["stop_sigmas"] = tr.risk_pct / tr.sigma
    tr["bucket"] = pd.cut(tr.stop_sigmas, [0, 1.5, 2.5, 3.5, 100],
                          labels=["<1.5 sigma", "1.5-2.5", "2.5-3.5", ">3.5 sigma"])
    t2 = tr.groupby(["strategy", "bucket"], observed=True).agg(
        trades=("net", "size"), stop_pct=("reason", lambda x: (x == "stop").mean() * 100),
        target_pct=("reason", lambda x: (x == "target").mean() * 100),
        mean_net_pct=("net", lambda x: x.mean() * 100)).round(2)
    print(f"\n--- {label} ---")
    print(t2.to_string())
    print(f"  median stop distance: {tr.stop_sigmas.median():.2f} sigma "
          f"(a stop this close is inside normal daily noise if it is under ~2 sigma)")
print("\nwrote reports/vol_regime.csv")
