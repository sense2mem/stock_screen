from __future__ import annotations

import numpy as np
import pandas as pd

from condition_backtest import (
    build_average_condition_curves,
    build_condition_backtest,
    build_condition_flags,
    build_markdown_summary,
    calculate_daily_condition_metrics,
    evaluate_condition_event,
    find_first_condition_dates,
)


def test_flags_and_first_dates_start_at_signal_date() -> None:
    index = pd.date_range("2026-01-01", periods=6, freq="B")
    metrics = pd.DataFrame(
        {
            "pullback_from_60d_high_pct": [-13, -13, -5, -13, -13, -5],
            "ma20_distance_pct": [2, 2, 2, 0, 2, 2],
            "di_spread": [4, 4, 4, 2, 2, 4],
        },
        index=index,
    )
    flags = build_condition_flags(metrics)
    first = find_first_condition_dates(flags, index[2])

    assert first["C"] == index[2]
    assert first["A"] == index[3]
    assert first["B"] == index[4]
    assert flags.loc[index[4], ["A", "B", "C"]].tolist() == [True, True, False]


def test_daily_metrics_use_rolling_high_ma_and_directional_movement() -> None:
    index = pd.date_range("2025-09-01", periods=80, freq="B")
    close = pd.Series(np.r_[np.full(60, 100.0), np.linspace(86, 95, 20)], index=index)
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
        },
        index=index,
    )
    metrics = calculate_daily_condition_metrics(prices)

    assert np.isclose(metrics.loc[index[60], "pullback_from_60d_high_pct"], -14.0)
    assert metrics.loc[index[-1], "plus_di14"] > metrics.loc[index[-1], "minus_di14"]
    assert metrics.loc[index[-1], "ma20_distance_pct"] > 0


def test_event_uses_next_open_and_60th_close() -> None:
    dates = pd.date_range("2025-09-01", periods=140, freq="B")
    signal_date = dates[69]
    condition_date = dates[70]
    prices = pd.DataFrame(
        {
            "Open": np.arange(140, dtype=float) + 100,
            "High": np.arange(140, dtype=float) + 102,
            "Low": np.arange(140, dtype=float) + 98,
            "Close": np.arange(140, dtype=float) + 101,
        },
        index=dates,
    )
    metrics = pd.DataFrame(
        {
            "pullback_from_60d_high_pct": -13.0,
            "ma20_distance_pct": 2.0,
            "plus_di14": 30.0,
            "minus_di14": 20.0,
            "di_spread": 10.0,
        },
        index=dates,
    )
    flags = pd.DataFrame(False, index=dates, columns=["A", "B", "C"])
    flags.loc[condition_date, ["A", "B", "C"]] = True
    signal = pd.Series(
        {
            "signal_date": signal_date,
            "ticker": "1001.T",
            "name": "Example",
            "score": 8,
            "sell_score": 0,
        }
    )

    row, curve = evaluate_condition_event(
        signal, prices, metrics, flags, "B", condition_date
    )

    expected_entry = prices.loc[dates[71], "Open"]
    expected_exit = prices.loc[dates[130], "Close"]
    assert row["entry_date"] == dates[71]
    assert row["entry_price"] == expected_entry
    assert row["exit_60d_date"] == dates[130]
    assert np.isclose(row["return_60d_pct"], (expected_exit / expected_entry - 1) * 100)
    assert row["condition_labels_on_first_date"] == "A|B|C"
    assert row["condition_group"] == "A_B_C"
    assert row["status"] == "CLOSED"
    assert len(curve) == 60


def test_build_backtest_reports_triggered_open_and_not_triggered() -> None:
    dates = pd.date_range("2025-07-01", periods=150, freq="B")
    close = pd.Series(np.full(150, 100.0), index=dates)
    close.iloc[65:85] = np.linspace(86, 95, 20)
    prices = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
        },
        index=dates,
    )
    signals = pd.DataFrame(
        [
            {
                "signal_date": dates[60],
                "ticker": "1001.T",
                "name": "Example",
                "score": 8,
                "sell_score": 0,
            }
        ]
    )

    detail, summary, curves, status = build_condition_backtest(
        signals, {"1001.T": prices}
    )

    assert set(detail["condition"]) >= {"A"}
    assert len(status) == 3
    assert summary["source_signals"].tolist() == [1, 1, 1]
    assert list(curves.columns) == [
        "day",
        "all_condition_events",
        "all_unique_entries",
        "A",
        "B",
        "C",
        "A_only",
        "A_B",
        "A_B_C",
        "C_only",
    ]


def test_curve_deduplicates_same_entry_for_condition_group() -> None:
    curve = pd.Series([101.0, 102.0], index=[1, 2])
    records = [
        {
            "condition": "A",
            "condition_group": "A_B",
            "event_key": (pd.Timestamp("2026-01-01"), "1001.T", pd.Timestamp("2026-01-05")),
            "curve": curve,
        },
        {
            "condition": "B",
            "condition_group": "A_B",
            "event_key": (pd.Timestamp("2026-01-01"), "1001.T", pd.Timestamp("2026-01-05")),
            "curve": curve,
        },
    ]
    result = build_average_condition_curves(records, horizon=2)

    assert result.loc[0, "all_condition_events"] == 101.0
    assert result.loc[0, "all_unique_entries"] == 101.0
    assert result.loc[0, "A_B"] == 101.0


def test_markdown_contains_60_day_results() -> None:
    status = pd.DataFrame(
        [
            {
                "signal_date": "2026-01-01",
                "ticker": "1001.T",
                "name": "Example",
                "condition": condition,
                "condition_first_date": "2026-01-05",
                "entry_date": "2026-01-06",
                "observed_trading_days": 60,
                "status": "CLOSED",
                "message": "ok",
            }
            for condition in ("A", "B", "C")
        ]
    )
    detail = pd.DataFrame(
        [
            {
                "condition": condition,
                "condition_first_date": "2026-01-05",
                "ticker": "1001.T",
                "name": "Example",
                "condition_labels_on_first_date": "A|B|C",
                "status": "CLOSED",
                "return_20d_pct": 5.0,
                "return_60d_pct": 10.0,
                "max_return_60d_pct": 12.0,
                "max_drawdown_60d_pct": -3.0,
            }
            for condition in ("A", "B", "C")
        ]
    )
    from condition_backtest import summarize_condition_backtest

    summary = summarize_condition_backtest(detail, status)
    markdown = build_markdown_summary(summary, detail)

    assert "first-occurrence backtest" in markdown
    assert "Avg 60d %" in markdown
    assert "1001.T" in markdown
