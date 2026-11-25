import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, pytest
from bt import metrics as m


def S(x): return pd.Series(x, dtype=float)


def test_cagr_doubling_in_one_year():
    r = S([2 ** (1 / 252) - 1] * 252)
    assert m.cagr(r) == pytest.approx(1.0, rel=1e-9)


def test_max_drawdown_known():
    # +100% then -50% => equity 2 -> 1 => DD -50%
    assert m.max_drawdown(S([1.0, -0.5])) == pytest.approx(-0.5)


def test_constant_returns_zero_vol_nan_sharpe():
    r = S([0.001] * 50)
    assert m.ann_vol(r) == 0 and np.isnan(m.sharpe(r))


def test_sharpe_scales_sqrt252():
    rng = np.random.default_rng(0)
    r = S(rng.normal(0.0005, 0.01, 5000))
    assert m.sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))


def test_alpha_beta_recovers_truth():
    rng = np.random.default_rng(1)
    b = S(rng.normal(0.0004, 0.01, 20000))
    r = 1.5 * b + 0.0002 + S(rng.normal(0, 0.002, 20000))
    ab = m.alpha_beta(r, b)
    assert ab["beta"] == pytest.approx(1.5, abs=0.02)
    assert ab["alpha_ann"] == pytest.approx(0.0002 * 252, abs=0.01)


def test_sortino_ge_sharpe_for_positive_skew():
    r = S([0.02, -0.005, 0.03, -0.004, 0.025, -0.006] * 50)
    assert m.sortino(r) > m.sharpe(r)
