from datetime import date

import pandas as pd
import pytest

from run_stock_screen import resolve_signal_date, score_and_observe_price_date


def test_resolve_signal_date_uses_majority_market_date():
    price_dates = [
        date(2026, 8, 31),
        pd.Timestamp("2026-08-31 15:00:00"),
        date(2026, 8, 31),
        date(2026, 8, 28),
    ]

    actual = resolve_signal_date(price_dates, total_tickers=4)

    assert actual == pd.Timestamp("2026-08-31")


def test_resolve_signal_date_ignores_missing_dates():
    price_dates = [
        date(2026, 8, 31),
        date(2026, 8, 31),
        None,
        pd.NaT,
    ]

    actual = resolve_signal_date(price_dates, total_tickers=4)

    assert actual == pd.Timestamp("2026-08-31")


def test_resolve_signal_date_rejects_no_market_dates():
    with pytest.raises(RuntimeError, match="no screened ticker price dates"):
        resolve_signal_date([None, pd.NaT], total_tickers=2)


def test_resolve_signal_date_rejects_insufficient_coverage():
    price_dates = [
        date(2026, 8, 31),
        date(2026, 8, 28),
        None,
        None,
        None,
        None,
    ]

    with pytest.raises(RuntimeError, match="sufficient coverage"):
        resolve_signal_date(price_dates, total_tickers=6)


def test_score_observer_records_exact_consumed_final_bar():
    observed = []
    df = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.to_datetime(["2026-08-28", "2026-08-31"]),
    )

    def scorer(frame):
        assert frame is df
        return {"score": 8}

    result = score_and_observe_price_date(scorer, observed, df)

    assert result == {"score": 8}
    assert observed == [pd.Timestamp("2026-08-31")]


def test_score_observer_does_not_record_failed_score():
    observed = []
    df = pd.DataFrame(
        {"Close": [100.0]},
        index=pd.to_datetime(["2026-08-31"]),
    )

    def scorer(_frame):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        score_and_observe_price_date(scorer, observed, df)

    assert observed == []
