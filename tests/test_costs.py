import sys; sys.path.insert(0, "src")
import pytest
from bt.costs import CostModel

C = CostModel()


def test_buy_fees_100k():
    # STT 100 + stamp 15 + exch 3.0699 + sebi 0.10 + gst 0.18*(3.0699+0.10)
    exp = 100 + 15 + 3.0699 + 0.10 + 0.18 * (3.0699 + 0.10)
    assert C.fees("buy", 1e5) == pytest.approx(exp, rel=1e-9)


def test_sell_fees_100k_includes_dp_no_stamp():
    exp = 100 + 3.0699 + 0.10 + 0.18 * (3.0699 + 0.10) + 15.34
    assert C.fees("sell", 1e5) == pytest.approx(exp, rel=1e-9)


def test_zero_notional_is_free():
    assert C.fees("buy", 0) == 0 and C.fees("sell", 0) == 0


def test_bad_side_raises():
    with pytest.raises(ValueError):
        C.fees("hold", 100)


def test_slippage_is_adverse():
    assert C.fill_price(100, "buy") > 100 > C.fill_price(100, "sell")
    assert C.fill_price(100, "buy") == pytest.approx(100.05)


def test_round_trip_in_expected_range():
    # ~0.2% STT + 0.015% stamp + small fees + 10 bps slippage => roughly 31-33 bps at Rs 1 lakh
    assert 30 < C.round_trip_bps(1e5) < 34
