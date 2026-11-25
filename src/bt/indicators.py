"""Indicators. Rule: nothing here may use information from the future.

wilder_rsi : classic Wilder RSI (seeded with a simple average, then recursive smoothing).
htf_rsi    : weekly/monthly RSI mapped onto daily dates using ONLY completed bars.
"""
import numpy as np
import pandas as pd


def wilder_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    c = close.to_numpy(dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return pd.Series(out, index=close.index)
    d = np.diff(c)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean()          # seed: simple average of first n changes
    out[n] = _rsi(au, ad)
    for i in range(n, len(d)):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
        out[i + 1] = _rsi(au, ad)
    return pd.Series(out, index=close.index)


def _rsi(au: float, ad: float) -> float:
    if ad == 0:
        return 100.0 if au > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + au / ad)


def htf_rsi(close: pd.Series, freq: str, n: int = 14) -> pd.Series:
    """RSI on higher-timeframe bars, aligned to daily dates with NO look-ahead.

    freq: 'M' (calendar month) or 'W-FRI' (week ending Friday).
    A daily date inside period P receives the RSI of the last COMPLETED period before P.
    The current, partial period never contributes. (Conservative: lags by up to one bar.)
    """
    periods = close.index.to_period(freq)
    bars = close.groupby(periods).last()           # last close of each period
    rsi_bar = wilder_rsi(bars, n).shift(1)         # known only after the period ends
    return pd.Series(rsi_bar.reindex(periods).to_numpy(), index=close.index)
