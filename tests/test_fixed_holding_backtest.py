from pathlib import Path

import numpy as np
import pandas as pd

from fixed_holding_backtest import (
    download_prices,
    evaluate_signal,
    load_signals,
    select_signals,
    summarize_results,
)


OHLC = ["Open", "High", "Low", "Close"]


def prices(periods=45):
    index = pd.bdate_range("2026-01-05", periods=periods)
    return pd.DataFrame(
        {
            "Open": np.arange(periods) + 100.0,
            "High": np.arange(periods) + 103.0,
            "Low": np.arange(periods) + 98.0,
            "Close": np.arange(periods) + 102.0,
        },
        index=index,
    )


def test_next_trading_day_open_and_fifth_day_close_and_excursions():
    data = prices()
    signal = pd.Series({"signal_date": "2026-01-05", "ticker": "AAA", "score": 8})
    result = evaluate_signal(signal, data, [5])[0]
    assert result["entry_date"] == pd.Timestamp("2026-01-06")
    assert result["entry_price"] == 101
    assert result["exit_date"] == pd.Timestamp("2026-01-12")
    assert result["exit_price"] == 107
    assert result["mfe_pct"] == (108 / 101 - 1) * 100
    assert result["mae_pct"] == (99 / 101 - 1) * 100


def test_first_in_streak_uses_screening_order_and_allows_reentry():
    signals = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2026-01-09", "2026-01-12", "2026-01-09", "2026-01-13", "2026-01-14"]
            ),
            "ticker": ["AAA", "AAA", "BBB", "BBB", "AAA"],
            "score": [8] * 5,
        }
    )
    selected = select_signals(signals, "first_in_streak")
    aaa = selected[selected.ticker == "AAA"].sort_values("signal_date")
    assert aaa.is_used.tolist() == [True, False, True]


def test_insufficient_days_excluded_from_summary():
    signal = pd.Series({"signal_date": "2026-01-05", "ticker": "AAA", "score": 8})
    detail = pd.DataFrame(evaluate_signal(signal, prices(20), [40]))
    assert detail.iloc[0].status == "OPEN_INSUFFICIENT_DAYS"
    assert summarize_results(detail, [40]).iloc[0].trades == 0


def test_load_signals_normalizes_and_deduplicates(tmp_path: Path):
    pd.DataFrame({"ticker": [" aaa ", "AAA", "LOW"], "score": [8, 9, 7]}).to_csv(
        tmp_path / "screen_2026-01-05_buy.csv", index=False
    )
    signals, files, raw = load_signals(tmp_path, "screen_*_buy.csv", 8)
    assert (files, raw, len(signals)) == (1, 3, 1)
    assert signals.iloc[0].ticker == "AAA"


def test_screening_day_without_qualifying_ticker_breaks_streak(tmp_path: Path):
    for day, score in [("05", 8), ("06", 7), ("07", 8)]:
        pd.DataFrame({"ticker": ["AAA"], "score": [score]}).to_csv(
            tmp_path / f"screen_2026-01-{day}_buy.csv", index=False
        )
    signals, _, _ = load_signals(tmp_path, "screen_*_buy.csv", 8)
    assert select_signals(signals, "first_in_streak").is_used.tolist() == [True, True]


def _ticker_first_frame():
    data = prices(10)
    data.columns = pd.MultiIndex.from_product([["OK"], OHLC])
    return data


def _field_first_frame():
    data = prices(10)
    data.columns = pd.MultiIndex.from_product([OHLC, ["OK"]])
    return data


def test_download_prices_handles_ticker_first_and_missing_ticker(monkeypatch):
    monkeypatch.setattr(
        "fixed_holding_backtest.yf.download",
        lambda *args, **kwargs: _ticker_first_frame(),
    )
    found, failures = download_prices(
        ["OK", "MISSING"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")
    )
    assert set(found) == {"OK"}
    assert set(failures) == {"MISSING"}
    assert list(found["OK"].columns) == OHLC


def test_download_prices_handles_field_first_multiindex(monkeypatch):
    monkeypatch.setattr(
        "fixed_holding_backtest.yf.download",
        lambda *args, **kwargs: _field_first_frame(),
    )
    found, failures = download_prices(
        ["OK", "MISSING"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")
    )
    assert set(found) == {"OK"}
    assert set(failures) == {"MISSING"}
    assert list(found["OK"].columns) == OHLC


def test_download_exception_marks_chunk_failed_and_does_not_raise(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr("fixed_holding_backtest.yf.download", fail)
    found, failures = download_prices(
        ["AAA", "BBB"], pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")
    )
    assert found == {}
    assert set(failures) == {"AAA", "BBB"}
    assert all("network unavailable" in message for message in failures.values())
