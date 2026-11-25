"""VaR / ES FORECASTS AND VaR BACKTEST, for (1) the Nifty 50 and (2) the S1 portfolio built by run_portfolio.py.

Question answered: "if a risk manager had used each method every day to say 'tomorrow's loss will not exceed X',
how often was that promise broken, and when it was broken, were the breaks bunched together?"

Every forecast for day d uses only data up to day d-1:
  historical, normal : rolling 250-day window, shifted one day
  egarch             : EGARCH(1,1)-t refit each January on data through the previous December
Levels: 99% and 97.5% (97.5% ES is the Basel FRTB market-risk standard; 99% VaR the older one).
Output: reports/var_backtest.csv (daily returns, forecasts, breach flags) for the dashboard.
Usage: python scripts/run_var.py   (needs reports/portfolio_equity.csv from run_portfolio.py)"""
import sys, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from bt.data import load_config, load_prices
from bt.risk import hist_var_es, normal_var_es, t_var_es, egarch_forecasts, kupiec, christoffersen

cfg = load_config()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
LEVELS = [0.01, 0.025]
nifty = load_prices(cfg["data"]["index_ticker"], cfg, allow_holdout=True)["Close"]
series = {"NIFTY 50": nifty.pct_change().dropna()}
try:
    eq = pd.read_csv("reports/portfolio_equity.csv", index_col=0, parse_dates=True)["equity"]
    series["S1 PORTFOLIO"] = eq.pct_change().dropna()
except FileNotFoundError:
    print("(reports/portfolio_equity.csv not found: run scripts/run_portfolio.py first; Nifty only)")

rows, daily = [], []
for name, r in series.items():
    g = egarch_forecasts(100 * r, first_year=2012)          # percent units for a stable fit
    g = g.reindex(r.index)
    ratio = (g["sigma"] / 100).median() / r.std()
    garch_ok = bool((g["nu"].dropna() > 2.05).all() and 0.2 < ratio < 5 and (g["sigma"].dropna() < 50 * 100 * r.std()).all())
    if not garch_ok:
        print(f"NOTE {name}: EGARCH fit is degenerate (median sigma / sample sd = {ratio:.2f}); "
              f"reported for historical and normal only. Typical cause: many days of near-constant returns "
              f"(the portfolio sits in cash), which a GARCH model cannot describe.")
    for a in LEVELS:
        f = {"historical": hist_var_es(r, a), "normal": normal_var_es(r, a)}
        if garch_ok:
            f["egarch-t"] = t_var_es(g["sigma"] / 100, g["nu"], a, g["mu"] / 100)
        common = r.index
        for k in f: common = common.intersection(f[k].dropna().index)
        common = common[common >= pd.Timestamp("2013-01-01")]
        for method, fc in f.items():
            x = fc.loc[common]; rr = r.loc[common]; loss = -rr
            br = (loss > x["var"]).to_numpy()
            kp = kupiec(br, a); ch = christoffersen(br)
            rows.append({"series": name, "level": f"{1-a:.1%}", "method": method, "days": kp["T"],
                         "expected_breaches": round(kp["T"] * a, 1), "breaches": kp["breaches"],
                         "breach_rate%": kp["rate"] * 100, "kupiec_p": kp["p"],
                         "cluster_p": ch["p"], "P(breach | breach yesterday)%": ch["p_after_breach"] * 100,
                         "avg_VaR%": x["var"].mean() * 100, "avg_ES%": x["es"].mean() * 100,
                         "actual_avg_loss_on_breach%": loss[br].mean() * 100 if br.any() else np.nan})
            daily.append(pd.DataFrame({"series": name, "level": f"{1-a:.1%}", "method": method,
                                       "ret": rr, "var": x["var"], "es": x["es"], "breach": br}, index=common))

res = pd.DataFrame(rows)
print(f"=== VaR BACKTEST, 1-day horizon, {res.days.iloc[0]} trading days from 2013 ===")
print("kupiec_p < 0.05  -> wrong NUMBER of breaches | cluster_p < 0.05 -> breaches BUNCHED together (slow model)\n")
for (s, lv), g in res.groupby(["series", "level"], sort=False):
    print(f"--- {s} | {lv} ---")
    print(g.drop(columns=["series", "level"]).round(3).to_string(index=False)); print()
print("ES check: on breach days the actual average loss should be close to avg_ES% (ES is the promised average of bad days).")
pd.concat(daily).to_csv("reports/var_backtest.csv")
print("wrote reports/var_backtest.csv")
