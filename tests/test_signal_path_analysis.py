from __future__ import annotations

import numpy as np
import pandas as pd

from signal_path_analysis import (
    PathRules,
    build_average_curves,
    calculate_signal_features,
    classify_price_path,
    detect_pattern_stage,
    summarize_patterns,
)


def prices_from_closes(closes: list[float], start: str = "2026-01-02") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(closes))
    close = pd.Series(closes, index=index, dtype=float)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.arange(len(close)) + 100,
        },
        index=index,
    )


def signal(date: str) -> pd.Series:
    return pd.Series(
        {
            "signal_date": pd.Timestamp(date),
            "ticker": "1234.T",
            "name": "Test",
            "score": 8,
            "sell_score": 0,
            "rsi": 55,
            "adx14": 22,
            "plus_di14": 30,
            "minus_di14": 18,
        }
    )


def test_double_rise_success_classification() -> None:
    closes = [100] + list(np.linspace(100, 106, 10)) + list(np.linspace(105, 98, 15)) + list(np.linspace(99, 112, 35))
    frame = prices_from_closes(closes)
    row, curve = classify_price_path(signal(str(frame.index[0].date())), frame, PathRules())
    assert row["path_label"] == "DOUBLE_RISE_SUCCESS"
    assert row["status"] == "CLOSED"
    assert pd.notna(row["breakout_day"])
    assert len(curve) == 60


def test_no_first_rise_classification() -> None:
    closes = [100] + list(np.linspace(100, 102, 15)) + list(np.linspace(101, 96, 15)) + list(np.linspace(97, 101, 30))
    frame = prices_from_closes(closes)
    row, _ = classify_price_path(signal(str(frame.index[0].date())), frame, PathRules())
    assert row["path_label"] == "NO_FIRST_RISE"


def test_in_progress_when_less_than_horizon() -> None:
    frame = prices_from_closes([100] * 31)
    row, curve = classify_price_path(signal(str(frame.index[0].date())), frame, PathRules())
    assert row["status"] == "IN_PROGRESS"
    assert len(curve) == 30


def test_breakout_approach_pattern() -> None:
    closes = [110, 108, 105, 100, 104, 109, 112, 108, 100, 92, 96, 104, 111, 108, 103, 98, 101, 105, 108, 109]
    result = detect_pattern_stage(prices_from_closes(closes), pivot_span=1)
    assert result["pattern_stage"] == "BREAKOUT_APPROACH"
    assert result["l2_date"] < result["l3_date"]


def test_post_breakout_pattern() -> None:
    closes = [110, 108, 105, 100, 104, 109, 112, 108, 100, 92, 96, 104, 111, 108, 103, 98, 101, 108, 113, 114]
    result = detect_pattern_stage(prices_from_closes(closes), pivot_span=1)
    assert result["pattern_stage"] == "POST_BREAKOUT"


def test_features_do_not_use_future_prices() -> None:
    frame = prices_from_closes(list(np.linspace(80, 120, 140)))
    signal_date = frame.index[100]
    before = calculate_signal_features(signal(str(signal_date.date())), frame)
    changed = frame.copy()
    changed.loc[changed.index > signal_date, "Close"] = 9999
    after = calculate_signal_features(signal(str(signal_date.date())), changed)
    assert before["ma20_distance_pct"] == after["ma20_distance_pct"]
    assert before["distance_to_20d_high_pct"] == after["distance_to_20d_high_pct"]
    assert before["pattern_stage"] == after["pattern_stage"]


def test_summary_contains_return_statistics() -> None:
    paths = pd.DataFrame(
        [
            {"path_label": "DOUBLE_RISE_SUCCESS", "status": "CLOSED", "return_50d_pct": 5, "return_60d_pct": 10, "pullback_pct": -6, "breakout_day": 45},
            {"path_label": "SECOND_RISE_FAILED", "status": "CLOSED", "return_50d_pct": -2, "return_60d_pct": -3, "pullback_pct": -7, "breakout_day": np.nan},
        ]
    )
    summary = summarize_patterns(paths)
    success = summary.loc[summary["path_label"] == "DOUBLE_RISE_SUCCESS"].iloc[0]
    assert success["signals"] == 1
    assert success["average_60d_return_pct"] == 10


def test_average_curves_are_grouped() -> None:
    curves = [
        ("DOUBLE_RISE_SUCCESS", pd.Series([101.0, 102.0], index=[1, 2])),
        ("SECOND_RISE_FAILED", pd.Series([99.0, 98.0], index=[1, 2])),
    ]
    result = build_average_curves(curves, horizon=2)
    assert result.loc[0, "all"] == 100.0
    assert result.loc[1, "DOUBLE_RISE_SUCCESS"] == 102.0
