"""Value at Risk (VaR) and Expected Shortfall (ES), plus the standard tests of whether a VaR model is honest.

Plain-language definitions (losses are reported as POSITIVE percentages):
  VaR at 99%  : a loss level that should be exceeded on only 1% of days.
  ES  at 99%  : the AVERAGE loss on those worst 1% of days (VaR's blind spot: how bad the bad days are).
Three ways to forecast tomorrow's VaR/ES, each using ONLY data up to today:
  historical : read the 1% worst day off the last 250 days.
  normal     : assume returns are bell-shaped with the last 250 days' volatility.
  garch      : use the EGARCH volatility forecast and fat-tailed Student-t errors, so the estimate
               rises as soon as markets turn turbulent.
Tests of a VaR model (a "VaR backtest"):
  Kupiec     : were there the RIGHT NUMBER of breaches? (99% VaR should be breached on ~1% of days)
  Christoffersen independence: were breaches SPREAD OUT, or did they arrive in clusters?
               Clustered breaches mean the model was slow to react, the classic failure of historical VaR.
"""
import numpy as np
import pandas as pd
from scipy import stats


def hist_var_es(r: pd.Series, alpha: float, window: int = 250) -> pd.DataFrame:
    """Rolling historical VaR/ES from the PREVIOUS `window` days (shifted: today's return is never used)."""
    q = r.rolling(window).quantile(alpha).shift(1)
    es = r.rolling(window).apply(lambda x: x[x <= np.quantile(x, alpha)].mean(), raw=True).shift(1)
    return pd.DataFrame({"var": -q, "es": -es})


def normal_var_es(r: pd.Series, alpha: float, window: int = 250) -> pd.DataFrame:
    """Rolling normal VaR/ES with the previous `window` days' mean and volatility."""
    mu = r.rolling(window).mean().shift(1); sd = r.rolling(window).std().shift(1)
    z = stats.norm.ppf(alpha)
    return pd.DataFrame({"var": -(mu + z * sd), "es": -(mu - sd * stats.norm.pdf(z) / alpha)})


def t_var_es(sigma: pd.Series, nu: pd.Series, alpha: float, mu: float | pd.Series = 0.0) -> pd.DataFrame:
    """VaR/ES for a STANDARDISED Student-t (unit variance), scaled by the forecast volatility `sigma`."""
    nu = nu.astype(float)
    q = stats.t.ppf(alpha, nu)
    scale = np.sqrt((nu - 2) / nu)                       # standardised t has variance 1
    var = -(mu + sigma * q * scale)
    es_std = scale * (nu + q ** 2) / (nu - 1) * stats.t.pdf(q, nu) / alpha
    return pd.DataFrame({"var": var, "es": -mu + sigma * es_std}, index=sigma.index)


def egarch_forecasts(r_pct: pd.Series, first_year: int, min_obs: int = 400) -> pd.DataFrame:
    """One-step-ahead EGARCH(1,1)-t volatility, refit every January on data through the previous December.
    Returns sigma (same units as r_pct) and the degrees of freedom nu in force that year. No look-ahead."""
    from arch import arch_model
    out = []
    for y in range(first_year, r_pct.index[-1].year + 1):
        cut = pd.Timestamp(f"{y}-01-01")
        if (r_pct.index < cut).sum() < min_obs: continue
        res = arch_model(r_pct, vol="EGARCH", p=1, o=1, q=1, dist="t").fit(last_obs=cut, disp="off")
        f = res.forecast(start=cut, horizon=1, reindex=False)
        v = f.variance.iloc[:, 0]; m = f.mean.iloc[:, 0]
        keep = (v.index >= cut) & (v.index < pd.Timestamp(f"{y + 1}-01-01"))
        out.append(pd.DataFrame({"sigma": np.sqrt(v[keep]), "mu": m[keep], "nu": res.params["nu"]}))
    df = pd.concat(out).sort_index()
    return df[~df.index.duplicated()]


def kupiec(breaches: np.ndarray, alpha: float) -> dict:
    """Proportion-of-failures test. H0: breach rate = alpha. Small p-value = wrong number of breaches."""
    b = np.asarray(breaches, bool); T = len(b); x = int(b.sum())
    if T == 0: return {"T": 0, "breaches": 0, "rate": np.nan, "p": np.nan}
    phat = x / T
    ll0 = (T - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll1 = (T - x) * np.log(1 - phat) + x * np.log(phat) if 0 < x < T else 0.0
    lr = -2 * (ll0 - ll1)
    return {"T": T, "breaches": x, "rate": phat, "LR": lr, "p": float(1 - stats.chi2.cdf(lr, 1))}


def christoffersen(breaches: np.ndarray) -> dict:
    """Independence test. H0: a breach today does not make a breach tomorrow more likely.
    Small p-value = breaches cluster (the model reacts too slowly)."""
    b = np.asarray(breaches, int)
    prev, cur = b[:-1], b[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum()); n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum()); n11 = int(((prev == 1) & (cur == 1)).sum())
    p01 = n01 / max(n00 + n01, 1); p11 = n11 / max(n10 + n11, 1); p = (n01 + n11) / max(len(cur), 1)

    def ll(k, pr):
        return k * np.log(pr) if k > 0 and pr > 0 else 0.0
    l0 = ll(n00 + n10, 1 - p) + ll(n01 + n11, p)
    l1 = ll(n00, 1 - p01) + ll(n01, p01) + ll(n10, 1 - p11) + ll(n11, p11)
    lr = -2 * (l0 - l1)
    return {"n11": n11, "p_after_breach": p11, "p_after_calm": p01, "LR": lr,
            "p": float(1 - stats.chi2.cdf(lr, 1))}
