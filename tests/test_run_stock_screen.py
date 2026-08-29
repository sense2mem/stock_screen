from datetime import date

import pandas as pd
import pytest

from run_stock_screen import resolve_signal_date


def test_resolve_signal_date_uses_majority_market_date():
    price_dates = [
        date(2026, 8, 28),
        pd.Timestamp("2026-08-28 15:00:00"),
        date(2026, 8, 28),
        date(2026, 8, 27),
    ]

    actual = resolve_signal_date(price_dates, total_tickers=4)

    assert actual == pd.Timestamp("2026-08-28")


def test_resolve_signal_date_ignores_missing_dates():
    price_dates = [
        date(2026, 8, 28),
        date(2026, 8, 28),
        None,
        pd.NaT,
    ]

    actual = resolve_signal_date(price_dates, total_tickers=4)

    assert actual == pd.Timestamp("2026-08-28")


def test_resolve_signal_date_rejects_no_market_dates():
    with pytest.raises(RuntimeError, match="no ticker price dates"):
        resolve_signal_date([None, pd.NaT], total_tickers=2)


def test_resolve_signal_date_rejects_insufficient_coverage():
    price_dates = [
        date(2026, 8, 28),
        date(2026, 8, 27),
        None,
        None,
        None,
        None,
    ]

    with pytest.raises(RuntimeError, match="sufficient coverage"):
        resolve_signal_date(price_dates, total_tickers=6)
