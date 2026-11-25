import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, pytest
from scipy import stats
from bt.risk import kupiec, christoffersen, normal_var_es, t_var_es, hist_var_es


def test_kupiec_exact_rate_passes_and_double_rate_fails():
    b = np.zeros(1000, bool); b[::100] = True                  # exactly 1%
    assert kupiec(b, 0.01)["p"] > 0.9
    b2 = np.zeros(1000, bool); b2[::33] = True                 # ~3%
    assert kupiec(b2, 0.01)["p"] < 0.01


def test_christoffersen_detects_clustering():
    spread = np.zeros(1000, int); spread[::50] = 1
    clustered = np.zeros(1000, int); clustered[500:520] = 1
    assert christoffersen(spread)["p"] > 0.05
    assert christoffersen(clustered)["p"] < 0.01


def test_normal_var_matches_formula_and_es_exceeds_var():
    rng = np.random.default_rng(0); r = pd.Series(rng.normal(0, 0.01, 600))
    v = normal_var_es(r, 0.01).dropna()
    sd = r.iloc[-251:-1].std(); mu = r.iloc[-251:-1].mean()
    assert v["var"].iloc[-1] == pytest.approx(-(mu + stats.norm.ppf(0.01) * sd))
    assert (v["es"] > v["var"]).all()


def test_t_es_exceeds_var_and_fat_tails_exceed_normal():
    s = pd.Series([1.0, 1.0]); nu = pd.Series([5.0, 5.0])
    t = t_var_es(s, nu, 0.01)
    assert (t["es"] > t["var"]).all()
    assert t["var"].iloc[0] > -stats.norm.ppf(0.01) * 0.99 * 0 + 2.0   # fat tails: 99% VaR of unit-variance t5 > 2.33*0.9
    assert t["es"].iloc[0] > -stats.norm.ppf(0.01)                     # ES beyond the normal 2.33


def test_hist_var_uses_only_past_days():
    r = pd.Series(np.r_[np.full(300, 0.001), -0.5])            # a crash on the last day
    v = hist_var_es(r, 0.01)
    assert v["var"].iloc[-1] < 0.01                            # today's crash is NOT in today's forecast
