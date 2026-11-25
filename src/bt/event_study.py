"""Event study: what happens h trading days after each signal, versus ANY day in the same stock?

Forward return for a signal on day T: close[T+1+h] / close[T+1] - 1 (enter next close, hold h days).
Baseline = the average forward return over all days of the same stock where the RSI features exist.
Excess = signal return - that stock's baseline. Bootstrap resamples calendar MONTHS, because
signals from different stocks cluster in time (a market-wide rebound hits many stocks at once).
"""
import numpy as np
import pandas as pd
from .signals import features, episode_starts


def fwd_return(close: pd.Series, h: int) -> pd.Series:
    return close.shift(-(1 + h)) / close.shift(-1) - 1


def cluster_bootstrap_means(values: pd.Series, n_boot: int = 10000, seed: int = 0) -> np.ndarray:
    groups = [g.to_numpy() for _, g in values.groupby(values.index.to_period("M"))]
    k = len(groups)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, k, k)
        means[b] = np.concatenate([groups[i] for i in pick]).mean()
    return means


def ci_from_means(means: np.ndarray, alpha: float = 0.05):
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cluster_bootstrap_ci(values: pd.Series, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    return ci_from_means(cluster_bootstrap_means(values, n_boot, seed), alpha)


def collect(closes: dict, signal_fn, s: dict, h: int):
    """Returns (episodes DataFrame, baseline DataFrame) across all tickers for horizon h."""
    ep_rows, base_rows = [], []
    for tk, close in closes.items():
        f = features(close, s["rsi_period"])
        valid = f.rsi_m.notna()
        fwd = fwd_return(close, h)
        base = fwd[valid].dropna()
        if base.empty:
            continue
        starts = episode_starts(signal_fn(f, s) & valid, s["cluster_gap_days"])
        r = fwd.reindex(starts).dropna()
        base_rows.append(pd.DataFrame({"ticker": tk, "ret": base}))
        if len(r):
            ep_rows.append(pd.DataFrame({"ticker": tk, "ret": r, "base_mean": base.mean()}))
    eps = pd.concat(ep_rows) if ep_rows else pd.DataFrame(columns=["ticker", "ret", "base_mean"])
    return eps, pd.concat(base_rows)


def summarize(closes, signal_fn, s, h, round_trip_bps, seed=0, n_comparisons=1) -> dict:
    eps, base = collect(closes, signal_fn, s, h)
    out = {"horizon": h, "episodes": len(eps), "stocks": eps.ticker.nunique() if len(eps) else 0,
           "baseline_mean": base.ret.mean(), "baseline_hit": (base.ret > 0).mean()}
    if len(eps) == 0:
        return out
    ex = eps.ret - eps.base_mean
    means = cluster_bootstrap_means(ex, s["bootstrap_draws"], seed)
    lo, hi = ci_from_means(means, 0.05)
    lo_a, hi_a = ci_from_means(means, 0.05 / n_comparisons)   # Bonferroni: stricter, for many comparisons
    out.update({"mean_gross": eps.ret.mean(), "mean_net": eps.ret.mean() - round_trip_bps / 1e4,
                "median_gross": eps.ret.median(), "hit_rate": (eps.ret > 0).mean(),
                "excess_mean": ex.mean(), "excess_ci_lo": lo, "excess_ci_hi": hi,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "adj_ci_lo": lo_a, "adj_ci_hi": hi_a,
                "survives_adjustment": bool(lo_a > 0 or hi_a < 0)})
    return out
