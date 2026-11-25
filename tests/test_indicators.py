import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, pytest
from bt.indicators import wilder_rsi, htf_rsi

# StockCharts ChartSchool RSI worked example (14-period). Reference RSI values are
# published there; the implementation must reproduce them.
SC = [44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245, 45.8433,
      46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028, 46.0328, 46.4116,
      46.2222, 45.6439, 46.2122, 46.2521, 45.7137, 46.4515, 45.7835, 45.3548, 44.0288,
      44.1783, 44.2181, 44.5672, 43.4205, 42.6628, 43.1314]


def test_rsi_matches_stockcharts_example():
    r = wilder_rsi(pd.Series(SC), 14)
    assert r.iloc[14] == pytest.approx(70.53, abs=0.05)
    assert r.iloc[15] == pytest.approx(66.32, abs=0.05)


def test_rsi_monotone_up_is_100_down_is_0():
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert wilder_rsi(up).dropna().eq(100).all()
    assert wilder_rsi(up[::-1].reset_index(drop=True)).dropna().eq(0).all()


def test_rsi_nan_until_enough_history():
    r = wilder_rsi(pd.Series(SC), 14)
    assert r.iloc[:14].isna().all() and r.iloc[14:].notna().all()


def _daily(n=900, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


@pytest.mark.parametrize("freq", ["M", "W-FRI"])
def test_htf_rsi_ignores_current_partial_period(freq):
    c = _daily()
    base = htf_rsi(c, freq)
    last_period = c.index.to_period(freq)[-1]
    mask = c.index.to_period(freq) == last_period
    c2 = c.copy()
    c2[mask] = c2[mask] * 1.5                      # wreck every close in the last period
    pert = htf_rsi(c2, freq)
    pd.testing.assert_series_equal(base[mask], pert[mask])   # unchanged: no leakage


@pytest.mark.parametrize("freq", ["M", "W-FRI"])
def test_htf_rsi_changes_only_after_period_ends(freq):
    c = _daily()
    r = htf_rsi(c, freq).dropna()
    per = r.index.to_period(freq)
    # constant within a period
    assert (r.groupby(per).nunique() == 1).all()
