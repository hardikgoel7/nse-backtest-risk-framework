"""Your rule, written as code. Thresholds come from config.yaml (frozen).

A: monthly RSI > trend_min AND weekly RSI > trend_min AND daily RSI in [support_lo, support_hi]
A2 (bounce): monthly and weekly RSI > trend_min AND yesterday's daily RSI in [support_lo, support_hi]
   AND today's daily RSI is higher than yesterday's. Defined on 2026-11-21, before it was run.
B: monthly RSI in [support_lo, support_hi]   ("value buy")
A signal on row T is known at the CLOSE of day T. The trade happens at the close of T+1.
"""
import numpy as np
import pandas as pd
from .indicators import wilder_rsi, htf_rsi


def features(close: pd.Series, n: int = 14) -> pd.DataFrame:
    return pd.DataFrame({"rsi_d": wilder_rsi(close, n),
                         "rsi_w": htf_rsi(close, "W-FRI", n),
                         "rsi_m": htf_rsi(close, "M", n)})


def signal_a(f: pd.DataFrame, s: dict) -> pd.Series:
    return ((f.rsi_m > s["trend_min"]) & (f.rsi_w > s["trend_min"])
            & f.rsi_d.between(s["support_lo"], s["support_hi"]))


def signal_a2(f: pd.DataFrame, s: dict) -> pd.Series:
    prev = f.rsi_d.shift(1)
    return ((f.rsi_m > s["trend_min"]) & (f.rsi_w > s["trend_min"])
            & prev.between(s["support_lo"], s["support_hi"]) & (f.rsi_d > prev))


def signal_bounce_only(f: pd.DataFrame, s: dict) -> pd.Series:
    """Control for A2: the same daily bounce, but NO monthly/weekly filter."""
    prev = f.rsi_d.shift(1)
    return prev.between(s["support_lo"], s["support_hi"]) & (f.rsi_d > prev)


def signal_b(f: pd.DataFrame, s: dict) -> pd.Series:
    return f.rsi_m.between(s["support_lo"], s["support_hi"])


def episode_starts(sig: pd.Series, gap: int) -> pd.DatetimeIndex:
    """First day of each episode. Signal days within `gap` trading days of the previous
    signal day belong to the same episode. Stops one idea being counted 15 times."""
    pos = np.flatnonzero(sig.to_numpy())
    starts, last = [], None
    for p in pos:
        if last is None or p - last > gap:
            starts.append(p)
        last = p
    return sig.index[starts]
