import sys; sys.path.insert(0, "src")
import numpy as np, pytest
from bt.costs import CostModel
from bt.trade_sim import simulate_trade

NOCOST = CostModel(brokerage_pct=0, stt_buy=0, stt_sell=0, stamp_buy=0, exch_txn=0, sebi=0,
                   gst=0, dp_per_sell_scrip=0, slippage_bps=0)
P = {"target_pct": 0.30, "max_hold_days": 10}


def mk(n=30, price=100.0):
    a = np.full(n, price)
    return a.copy(), a.copy(), a.copy(), a.copy()     # O, H, L, C


def test_target_hit_fills_at_target():
    O, H, L, C = mk(); L[4] = 99; t = 5               # stop = L[t-1] = 99, entry C[6]=100
    H[8] = 135                                          # high touches 130 on day 8
    r = simulate_trade(O, H, L, C, t, 1, P, NOCOST)
    assert r["reason"] == "target" and r["gross"] == pytest.approx(0.30) and r["days"] == 2


def test_stop_hit_fills_at_stop():
    O, H, L, C = mk(); L[4] = 99; t = 5
    L[8] = 90
    r = simulate_trade(O, H, L, C, t, 1, P, NOCOST)
    assert r["reason"] == "stop" and r["gross"] == pytest.approx(99 / 100 - 1)
    assert r["R"] == pytest.approx(-1.0)


def test_gap_down_exits_at_open_below_stop():
    O, H, L, C = mk(); L[4] = 99; t = 5
    O[8] = 80; L[8] = 78
    r = simulate_trade(O, H, L, C, t, 1, P, NOCOST)
    assert r["reason"] == "stop" and r["gross"] == pytest.approx(-0.20)


def test_same_day_stop_and_target_is_stop():
    O, H, L, C = mk(); L[4] = 99; t = 5
    H[8] = 140; L[8] = 90
    assert simulate_trade(O, H, L, C, t, 1, P, NOCOST)["reason"] == "stop"


def test_timeout_exits_at_close():
    O, H, L, C = mk(); L[4] = 99; t = 5
    C[16] = 105                                         # day e+10 = 16
    r = simulate_trade(O, H, L, C, t, 1, P, NOCOST)
    assert r["reason"] == "timeout" and r["gross"] == pytest.approx(0.05)


def test_skipped_when_stop_not_below_entry():
    O, H, L, C = mk(); L[4] = 101; t = 5     # stop = 101 > entry 100
    assert simulate_trade(O, H, L, C, t, 1, P, NOCOST)["status"] == "skipped"


def test_censored_near_end_of_data():
    O, H, L, C = mk(n=12); t = 5
    assert simulate_trade(O, H, L, C, t, 1, P, NOCOST)["status"] == "censored"


def test_costs_reduce_return():
    O, H, L, C = mk(); L[4] = 99; t = 5; H[8] = 135
    r0 = simulate_trade(O, H, L, C, t, 1, P, NOCOST)
    r1 = simulate_trade(O, H, L, C, t, 1, P, CostModel())
    assert r1["net"] < r0["net"]


def test_r_multiple_target_is_three_times_risk():
    O, H, L, C = mk(); L[4] = 98; t = 5               # stop 98, entry 100 -> risk 2, target 106
    H[8] = 107
    r = simulate_trade(O, H, L, C, t, 1, {"target_r": 3.0, "target_pct": 0.30, "max_hold_days": 10}, NOCOST)
    assert r["reason"] == "target" and r["gross"] == pytest.approx(0.06) and r["R"] == pytest.approx(3.0)


def test_rsi_exit_sells_next_close_and_has_no_target():
    O, H, L, C = mk(); L[4] = 99; t = 5               # entry 100 at day 6
    rsi = np.full(30, 45.0); rsi[10] = 61.0           # RSI crosses 60 at close of day 10
    C[11] = 125; H[8] = 200                            # a big spike must NOT trigger any target
    p = {"exit_rsi": 60, "max_hold_days": 10}
    r = simulate_trade(O, H, L, C, t, 1, p, NOCOST, rsi=rsi)
    assert r["reason"] == "rsi" and r["days"] == 5 and r["gross"] == pytest.approx(0.25)


def test_rsi_exit_stop_still_active_on_exit_day():
    O, H, L, C = mk(); L[4] = 99; t = 5
    rsi = np.full(30, 45.0); rsi[10] = 61.0
    L[11] = 90                                         # stop hit on the day the RSI exit would execute
    r = simulate_trade(O, H, L, C, t, 1, {"exit_rsi": 60, "max_hold_days": 10}, NOCOST, rsi=rsi)
    assert r["reason"] == "stop"


def test_stop_override_used():
    O, H, L, C = mk(); t = 5                          # entry 100; explicit stop 95, target 3R = 115
    H[8] = 116
    r = simulate_trade(O, H, L, C, t, 0, {"target_r": 3.0, "max_hold_days": 10}, NOCOST, stop_override=95.0)
    assert r["reason"] == "target" and r["gross"] == pytest.approx(0.15) and r["risk_pct"] == pytest.approx(0.05)


from bt.trade_sim import simulate_breakout_trade
PB = {"target_r": 3.0, "max_hold_days": 10}


def test_breakout_intraday_fill_at_trigger_and_3R_target():
    O, H, L, C = mk(); t = 5                          # signal candle: high 102 (trigger), low 98 (stop)
    H[6] = 103                                          # touches 102 on day 6 -> fill at 102, risk 4, target 114
    H[8] = 115
    r = simulate_breakout_trade(O, H, L, C, t, 102.0, 98.0, 5, PB, NOCOST)
    assert r["status"] == "ok" and r["entry"] == pytest.approx(102) and r["reason"] == "target"
    assert r["gross"] == pytest.approx(12 / 102) and r["R"] == pytest.approx(3.0) and r["days"] == 2


def test_breakout_gap_up_fills_at_open_and_risk_uses_fill():
    O, H, L, C = mk(); t = 5
    O[6] = 105; H[6] = 106; L[6] = 105; C[6] = 105      # gap above trigger: fill at 105, risk 7, target 126
    H[9] = 130
    r = simulate_breakout_trade(O, H, L, C, t, 102.0, 98.0, 5, PB, NOCOST)
    assert r["entry"] == pytest.approx(105) and r["risk_pct"] == pytest.approx(7 / 105) and r["reason"] == "target"


def test_breakout_invalidated_if_stop_hit_before_fill():
    O, H, L, C = mk(); t = 5
    L[6] = 97                                           # breaks the stop before any trigger
    H[7] = 110
    assert simulate_breakout_trade(O, H, L, C, t, 102.0, 98.0, 5, PB, NOCOST)["status"] == "invalidated"


def test_breakout_same_day_trigger_and_stop_is_invalidated():
    O, H, L, C = mk(); t = 5
    H[6] = 103; L[6] = 97                               # order of events unknown -> conservative: no trade
    assert simulate_breakout_trade(O, H, L, C, t, 102.0, 98.0, 5, PB, NOCOST)["status"] == "invalidated"


def test_breakout_unfilled_when_window_ends():
    O, H, L, C = mk(); t = 5
    assert simulate_breakout_trade(O, H, L, C, t, 102.0, 98.0, 5, PB, NOCOST)["status"] == "unfilled"


def test_breakout_gap_up_then_through_stop_same_day_exits_at_stop():
    O, H, L, C = mk(); t = 5
    O[6] = 105; H[6] = 106; L[6] = 90; C[6] = 92
    r = simulate_breakout_trade(O, H, L, C, t, 102.0, 98.0, 5, PB, NOCOST)
    assert r["reason"] == "stop" and r["days"] == 0 and r["gross"] == pytest.approx(98 / 105 - 1)
