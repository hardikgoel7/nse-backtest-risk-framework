import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd, pytest
from bt.signals import features, signal_a, signal_a2, signal_b, episode_starts
from bt.event_study import fwd_return, cluster_bootstrap_ci, summarize

S = {"rsi_period": 14, "trend_min": 60, "support_lo": 38, "support_hi": 42,
     "cluster_gap_days": 20, "horizons": [5], "primary_horizons": [5], "bootstrap_draws": 200}


def walk(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))), index=idx)


def test_signal_a_logic():
    f = pd.DataFrame({"rsi_m": [65, 65, 55, 65, np.nan], "rsi_w": [65, 65, 65, 55, 65],
                      "rsi_d": [40, 45, 40, 40, 40]})
    assert signal_a(f, S).tolist() == [True, False, False, False, False]


def test_signal_b_logic():
    f = pd.DataFrame({"rsi_m": [37.9, 38, 40, 42, 42.1], "rsi_w": 0, "rsi_d": 0})
    assert signal_b(f, S).tolist() == [False, True, True, True, False]


def test_episode_grouping():
    def sig(pos, n=100):
        a = np.zeros(n, bool); a[pos] = True
        return pd.Series(a, index=pd.bdate_range("2020-01-01", periods=n))
    assert len(episode_starts(sig([3, 4, 5, 40, 41]), 20)) == 2
    assert len(episode_starts(sig([3, 20]), 20)) == 1      # gap 17 <= 20: same episode
    assert len(episode_starts(sig([3, 30]), 20)) == 2      # gap 27 > 20: new episode


def test_forward_return_entry_next_close():
    c = pd.Series(np.arange(100.0, 120.0), index=pd.bdate_range("2020-01-01", periods=20))
    r = fwd_return(c, 5)
    # signal on index 2 -> enter close[3]=103, exit close[3+5]=108
    assert r.iloc[2] == pytest.approx(108 / 103 - 1)
    assert r.iloc[-6:].isna().all()                          # no full horizon, no value


def test_signals_are_causal_no_lookahead():
    c = walk()
    full = features(c)
    for t in [400, 777, 1200]:
        part = features(c.iloc[: t + 1])
        pd.testing.assert_series_equal(full.iloc[t], part.iloc[t], check_names=False)


def test_bootstrap_ci_sane_and_deterministic():
    v = pd.Series(np.random.default_rng(1).normal(0.01, 0.05, 300),
                  index=pd.bdate_range("2015-01-01", periods=300))
    lo, hi = cluster_bootstrap_ci(v, 300, seed=3)
    assert lo < v.mean() < hi
    assert (lo, hi) == cluster_bootstrap_ci(v, 300, seed=3)


def test_event_study_runs_end_to_end():
    closes = {f"T{i}": walk(seed=i) for i in range(6)}
    out = summarize(closes, signal_b, S, 5, 33.0)
    assert out["episodes"] > 0 and "excess_ci_lo" in out
    assert out["mean_net"] == pytest.approx(out["mean_gross"] - 0.0033)


def test_signal_a2_bounce_logic():
    f = pd.DataFrame({"rsi_d": [40, 41, 45, 39, 41, 41], "rsi_m": 65.0, "rsi_w": 65.0})
    # t=1: prev 40 in band, 41>40 ; t=2: prev 41, 45>41 ; t=3: prev 45 out ; t=4: prev 39, 41>39 ; t=5: 41 not > 41
    assert signal_a2(f, S).tolist() == [False, True, True, False, True, False]


def test_signal_a2_needs_trend():
    f = pd.DataFrame({"rsi_d": [40, 41], "rsi_m": 65.0, "rsi_w": [65.0, 55.0]})
    assert signal_a2(f, S).tolist() == [False, False]


def test_adjusted_ci_is_wider():
    closes = {f"T{i}": walk(seed=i) for i in range(6)}
    out = summarize(closes, signal_b, S, 5, 33.0, n_comparisons=15)
    assert out["adj_ci_lo"] <= out["excess_ci_lo"] and out["adj_ci_hi"] >= out["excess_ci_hi"]
