from __future__ import annotations

import numpy as np
import pandas as pd

from condition_backtest_v2 import (
    add_trigger_days,
    confirmed_prices,
    limited_first_dates,
    summarize,
    unique_entries,
)


def _prices(periods: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2026-07-20", periods=periods, freq="B")
    close = pd.Series(np.arange(periods, dtype=float) + 100.0, index=dates)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close},
        index=dates,
    )


def test_current_japanese_bar_is_excluded_before_close() -> None:
    prices = _prices(5)
    today = prices.index[-1]
    before = pd.Timestamp(today.date(), tz="Asia/Tokyo") + pd.Timedelta(hours=13)
    after = pd.Timestamp(today.date(), tz="Asia/Tokyo") + pd.Timedelta(hours=16)

    assert today not in confirmed_prices(prices, before).index
    assert today in confirmed_prices(prices, after).index


def test_trigger_search_is_limited_to_day_60_including_day_zero() -> None:
    dates = pd.date_range("2026-01-01", periods=70, freq="B")
    flags = pd.DataFrame(False, index=dates, columns=["A", "B", "C"])
    flags.iloc[60, flags.columns.get_loc("C")] = True
    flags.iloc[61, flags.columns.get_loc("A")] = True

    result = limited_first_dates(flags, dates[0], days=60)

    assert result["C"] == dates[60]
    assert "A" not in result


def test_unique_entries_remove_same_trade_from_multiple_score8_signals() -> None:
    detail = pd.DataFrame(
        [
            {"signal_date": "2026-02-03", "ticker": "7717.T", "condition": "A", "entry_date": "2026-04-01"},
            {"signal_date": "2026-02-06", "ticker": "7717.T", "condition": "A", "entry_date": "2026-04-01"},
            {"signal_date": "2026-02-06", "ticker": "7717.T", "condition": "B", "entry_date": "2026-04-01"},
        ]
    )

    result = unique_entries(detail)

    assert len(result) == 2
    assert set(result["condition"]) == {"A", "B"}


def test_trigger_day_uses_actual_price_index() -> None:
    prices = _prices(10)
    detail = pd.DataFrame(
        [
            {
                "signal_date": prices.index[2],
                "ticker": "1001.T",
                "condition": "C",
                "condition_first_date": prices.index[5],
                "entry_date": prices.index[6],
            }
        ]
    )

    result = add_trigger_days(detail, {"1001.T": prices})

    assert result.loc[0, "condition_trigger_day"] == 3


def test_summary_uses_deduplicated_entries() -> None:
    detail = pd.DataFrame(
        [
            {
                "signal_date": "2026-01-01", "ticker": "1001.T", "condition": "A",
                "entry_date": "2026-01-10", "status": "CLOSED", "return_5d_pct": 1.0,
                "return_10d_pct": 2.0, "return_20d_pct": 3.0, "return_40d_pct": 4.0,
                "return_50d_pct": 5.0, "return_60d_pct": 10.0,
                "max_return_60d_pct": 15.0, "max_drawdown_60d_pct": -5.0,
            },
            {
                "signal_date": "2026-01-02", "ticker": "1001.T", "condition": "A",
                "entry_date": "2026-01-10", "status": "CLOSED", "return_5d_pct": 1.0,
                "return_10d_pct": 2.0, "return_20d_pct": 3.0, "return_40d_pct": 4.0,
                "return_50d_pct": 5.0, "return_60d_pct": 10.0,
                "max_return_60d_pct": 15.0, "max_drawdown_60d_pct": -5.0,
            },
        ]
    )
    status = pd.DataFrame(
        [
            {"condition": condition, "status": "CLOSED" if condition == "A" else "NOT_TRIGGERED"}
            for condition in ("A", "B", "C")
        ]
    )

    result = summarize(detail, status).set_index("condition")

    assert result.loc["A", "triggered_signals"] == 2
    assert result.loc["A", "unique_entries"] == 1
    assert result.loc["A", "duplicate_events_removed"] == 1
    assert result.loc["A", "average_60d_return_pct"] == 10.0
