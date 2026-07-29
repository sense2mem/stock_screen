from __future__ import annotations

import pandas as pd

from condition_signal_output import build_condition_signals, build_markdown_summary


def features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": "2026-01-05",
                "ticker": "1001.T",
                "name": "A only",
                "score": 8,
                "sell_score": 0,
                "pullback_from_60d_high_pct": -12.0,
                "ma20_distance_pct": 0.5,
                "plus_di14": 22,
                "minus_di14": 20,
                "di_spread": 2.0,
                "rsi": 50,
                "adx14": 18,
                "pattern_stage": "NONE",
            },
            {
                "signal_date": "2026-01-06",
                "ticker": "1002.T",
                "name": "A B C",
                "score": 8,
                "sell_score": 0,
                "pullback_from_60d_high_pct": -15.0,
                "ma20_distance_pct": 1.0,
                "plus_di14": 30,
                "minus_di14": 27,
                "di_spread": 3.0,
                "rsi": 55,
                "adx14": 22,
                "pattern_stage": "BREAKOUT_APPROACH",
            },
            {
                "signal_date": "2026-01-07",
                "ticker": "1003.T",
                "name": "C only",
                "score": 8,
                "sell_score": 0,
                "pullback_from_60d_high_pct": -5.0,
                "ma20_distance_pct": 2.0,
                "plus_di14": 28,
                "minus_di14": 20,
                "di_spread": 8.0,
                "rsi": 60,
                "adx14": 25,
                "pattern_stage": "NONE",
            },
            {
                "signal_date": "2026-01-08",
                "ticker": "1004.T",
                "name": "No match",
                "score": 8,
                "sell_score": 0,
                "pullback_from_60d_high_pct": -11.99,
                "ma20_distance_pct": 0.99,
                "plus_di14": 24,
                "minus_di14": 22,
                "di_spread": 2.99,
                "rsi": 45,
                "adx14": 15,
                "pattern_stage": "NONE",
            },
        ]
    )


def test_condition_flags_and_boundary_values() -> None:
    result = build_condition_signals(features())
    assert result["ticker"].tolist() == ["1001.T", "1002.T", "1003.T"]
    labels = dict(zip(result["ticker"], result["condition_labels"]))
    assert labels["1001.T"] == "A"
    assert labels["1002.T"] == "A|B|C"
    assert labels["1003.T"] == "C"


def test_condition_b_is_subset_of_condition_a() -> None:
    result = build_condition_signals(features())
    assert not result.loc[result["condition_b"], "condition_a"].eq(False).any()


def test_missing_values_do_not_match() -> None:
    frame = features().iloc[:1].copy()
    frame.loc[:, "pullback_from_60d_high_pct"] = pd.NA
    frame.loc[:, "ma20_distance_pct"] = pd.NA
    frame.loc[:, "di_spread"] = pd.NA
    result = build_condition_signals(frame)
    assert result.empty


def test_markdown_summary_contains_counts_and_tickers() -> None:
    result = build_condition_signals(features())
    summary = build_markdown_summary(result)
    assert "| A | 2 |" in summary
    assert "| B | 1 |" in summary
    assert "| C | 2 |" in summary
    assert "1002.T" in summary
    assert "A\\|B\\|C" in summary
