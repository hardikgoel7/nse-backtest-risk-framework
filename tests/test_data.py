import sys; sys.path.insert(0, "src")
import pandas as pd
from bt.data import clean_prices


def mk(vals):
    return pd.DataFrame({"Close": vals}, index=pd.bdate_range("2020-01-01", periods=len(vals)))


def test_reciprocal_spike_removed():
    df = mk([100, 101, 10, 10.2, 101.5, 102])
    out, log = clean_prices(df)
    assert list(out["Close"]) == [100, 101, 101.5, 102]
    assert log[0][0] == "spike_dropped"


def test_real_crash_untouched():
    df = mk([100, 101, 70, 69, 71, 72])      # -30% and stays down
    out, log = clean_prices(df)
    assert len(out) == len(df) and log == []


def test_early_unadjusted_split_trimmed():
    df = mk([1000, 1010, 1005, 100, 101, 102, 103])
    out, log = clean_prices(df)
    assert list(out["Close"]) == [100, 101, 102, 103]
    assert log[0][0] == "early_split_trim"


def test_clean_series_unchanged():
    df = mk([100, 101, 102, 101, 103])
    out, log = clean_prices(df)
    assert out.equals(df) and log == []


def test_quality_report_handles_empty_frame():
    import pandas as pd
    from bt.data import quality_report
    q = quality_report(pd.DataFrame({"Close": []}, index=pd.DatetimeIndex([])))
    assert q["rows"] == 0 and q["jumps"] == []


def test_upward_spike_reversed_next_day_removed():
    out, log = clean_prices(mk([100, 101, 206, 102, 103]))
    assert list(out["Close"]) == [100, 101, 102, 103] and log[0][0] == "spike_dropped"


def test_slow_recovery_after_crash_is_NOT_a_bad_print():
    # Yes Bank-style: -56% real crash, recovers over ~6 days. Must be left alone.
    out, log = clean_prices(mk([100] * 25 + [101, 44, 48, 52, 70, 85, 99, 100]))
    assert len(out) == 33 and log == []


def test_quarantine_flags_unreversed_big_move():
    from bt.data import quarantine_reason
    assert quarantine_reason(mk([100, 101, 30, 31, 30]))[1] < -0.5
    assert quarantine_reason(mk([100, 101, 90, 95, 99])) is None
