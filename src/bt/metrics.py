"""Performance metrics on a series of simple daily returns (pandas Series)."""
import numpy as np
import pandas as pd


def equity_curve(r: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1 + r).cumprod()


def cagr(r: pd.Series, ppy: int = 252) -> float:
    n = len(r)
    if n == 0:
        return np.nan
    return float((1 + r).prod() ** (ppy / n) - 1)


def ann_vol(r: pd.Series, ppy: int = 252) -> float:
    return float(r.std(ddof=1) * np.sqrt(ppy))


def sharpe(r: pd.Series, rf_annual: float = 0.0, ppy: int = 252) -> float:
    ex = r - rf_annual / ppy
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(ppy)) if sd > 0 else np.nan


def sortino(r: pd.Series, rf_annual: float = 0.0, ppy: int = 252) -> float:
    """Downside deviation vs the daily risk-free (target), over ALL observations."""
    ex = r - rf_annual / ppy
    dd = np.sqrt(np.mean(np.minimum(ex, 0.0) ** 2))
    return float(ex.mean() / dd * np.sqrt(ppy)) if dd > 0 else np.nan


def drawdown(r: pd.Series) -> pd.Series:
    eq = equity_curve(r)
    return eq / eq.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    return float(drawdown(r).min())


def calmar(r: pd.Series, ppy: int = 252) -> float:
    mdd = abs(max_drawdown(r))
    return cagr(r, ppy) / mdd if mdd > 0 else np.nan


def alpha_beta(r: pd.Series, bench: pd.Series, rf_annual: float = 0.0, ppy: int = 252):
    """OLS of daily excess returns. Alpha annualised (x ppy), plus its t-stat."""
    df = pd.concat([r, bench], axis=1, join="inner").dropna()
    y = df.iloc[:, 0].to_numpy() - rf_annual / ppy
    x = df.iloc[:, 1].to_numpy() - rf_annual / ppy
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape
    s2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {"alpha_ann": float(coef[0] * ppy), "beta": float(coef[1]),
            "alpha_tstat": float(coef[0] / se[0])}


def summary(r: pd.Series, rf_annual: float = 0.0, ppy: int = 252) -> dict:
    return {
        "CAGR": cagr(r, ppy), "Vol": ann_vol(r, ppy),
        "Sharpe": sharpe(r, rf_annual, ppy), "Sortino": sortino(r, rf_annual, ppy),
        "MaxDD": max_drawdown(r), "Calmar": calmar(r, ppy), "N_days": len(r),
    }
