from __future__ import annotations

import numpy as np
import pandas as pd

from condition_c_topix_analysis import (
    build_analysis,
    confirmed_prices,
    evaluate_trade,
    summarize,
)


def _prices(periods: int = 70, start: str = "2026-01-05") -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="B")
    close = pd.Series(np.linspace(100.0, 130.0, periods), index=dates)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close},
        index=dates,
    )


def test_partial_today_bar_is_excluded_before_japanese_close() -> None:
    prices = _prices(5)
    today = prices.index[-1]
    before = pd.Timestamp(today.date(), tz="Asia/Tokyo") + pd.Timedelta(hours=13)
    after = pd.Timestamp(today.date(), tz="Asia/Tokyo") + pd.Timedelta(hours=16)

    assert today not in confirmed_prices(prices, before).index
    assert today in confirmed_prices(prices, after).index


def test_stop_loss_uses_matching_topix_exit_date() -> None:
    stock = _prices(70)
    signal_date = stock.index[0]
    entry_date = stock.index[1]
    entry = float(stock.loc[entry_date, "Open"])
    stock.loc[stock.index[6], "Low"] = entry * 0.85
    benchmark = _prices(70)
    signal = pd.Series({"signal_date": signal_date, "ticker": "1001.T", "score": 8})

    row = evaluate_trade(signal, stock, benchmark, stop_loss_pct=14.0, holding_days=60)

    assert row["status"] == "CLOSED"
    assert row["exit_reason"] == "STOP_LOSS"
    assert row["strategy_return_pct"] == -14.0
    assert row["strategy_exit_date"] == stock.index[6]
    assert row["benchmark_exit_date"] == stock.index[6]
    assert np.isclose(
        row["excess_return_pct"],
        row["strategy_return_pct"] - row["benchmark_return_pct"],
    )


def test_non_stop_trade_exits_at_60th_close() -> None:
    stock = _prices(70)
    benchmark = _prices(70)
    signal = pd.Series({"signal_date": stock.index[0], "ticker": "1001.T", "score": 8})

    row = evaluate_trade(signal, stock, benchmark, stop_loss_pct=14.0, holding_days=60)

    assert row["status"] == "CLOSED"
    assert row["exit_reason"] == "HOLD_60D"
    assert row["holding_days_to_exit"] == 60
    assert row["strategy_exit_date"] == stock.index[60]
    assert np.isclose(row["strategy_return_pct"], row["no_stop_60d_return_pct"])


def test_summary_reports_strategy_benchmark_and_excess() -> None:
    signals = pd.DataFrame(
        [
            {"signal_date": "2026-01-05", "ticker": "1001.T", "score": 8},
            {"signal_date": "2026-01-05", "ticker": "1002.T", "score": 8},
        ]
    )
    first = _prices(70)
    second = _prices(70)
    entry = float(second.iloc[1]["Open"])
    second.loc[second.index[5], "Low"] = entry * 0.80
    benchmark = _prices(70)

    detail = build_analysis(
        signals,
        {"1001.T": first, "1002.T": second},
        benchmark,
        stop_loss_pct=14.0,
        holding_days=60,
    )
    result = summarize(detail).set_index("segment")

    assert result.loc["ALL", "closed_trades"] == 2
    assert result.loc["ALL", "stop_loss_trades"] == 1
    assert result.loc["ALL", "held_60d_trades"] == 1
    expected = detail["strategy_return_pct"].mean() - detail["benchmark_return_pct"].mean()
    assert np.isclose(result.loc["ALL", "average_excess_return_pct"], expected)
