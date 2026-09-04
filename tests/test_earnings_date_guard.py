import pandas as pd

from earnings_date_guard import (
    _date_to_unix_seconds,
    _unix_to_date,
    repair_earnings_dates,
)


def _quote_ts(day):
    return _date_to_unix_seconds(pd.Timestamp(day))


class _CalendarTicker:
    calendar = {
        "Earnings Date": [pd.Timestamp("2026-11-11")],
    }

    def get_earnings_dates(self, limit=12):
        return None


class _NoDateTicker:
    calendar = {}

    def get_earnings_dates(self, limit=12):
        return None


def test_stale_quote_date_is_repaired_for_relevant_score8_ticker():
    val_df = pd.DataFrame(
        [
            {
                "ticker": "1982.T",
                "earn_start_ts": _quote_ts("2025-08-06"),
                "earn_ts": _quote_ts("2025-08-06"),
                "earn_end_ts": _quote_ts("2025-08-06"),
            }
        ]
    )
    tech_df = pd.DataFrame(
        [{"ticker": "1982.T", "score": 8, "sell_score": 0}]
    )

    repaired = repair_earnings_dates(
        val_df,
        tech_df,
        as_of_date="2026-09-04",
        buy_score_min=6,
        sell_score_min=5,
        pause_sec=0,
        ticker_factory=lambda _ticker: _CalendarTicker(),
    )

    assert repaired.loc[0, "earn_date_raw_stale"]
    assert repaired.loc[0, "earn_date_source"] == "yfinance_calendar"
    assert _unix_to_date(repaired.loc[0, "earn_start_ts"]) == pd.Timestamp(
        "2026-11-11"
    )


def test_future_quote_date_is_kept_without_ticker_level_fallback():
    val_df = pd.DataFrame(
        [
            {
                "ticker": "1982.T",
                "earn_start_ts": _quote_ts("2026-11-11"),
                "earn_ts": _quote_ts("2026-11-11"),
                "earn_end_ts": _quote_ts("2026-11-12"),
            }
        ]
    )
    tech_df = pd.DataFrame(
        [{"ticker": "1982.T", "score": 8, "sell_score": 0}]
    )

    calls = []

    def factory(ticker):
        calls.append(ticker)
        return _NoDateTicker()

    repaired = repair_earnings_dates(
        val_df,
        tech_df,
        as_of_date="2026-09-04",
        buy_score_min=6,
        sell_score_min=5,
        pause_sec=0,
        ticker_factory=factory,
    )

    assert calls == []
    assert repaired.loc[0, "earn_date_source"] == "yahoo_quote_future"
    assert _unix_to_date(repaired.loc[0, "earn_start_ts"]) == pd.Timestamp(
        "2026-11-11"
    )


def test_stale_quote_date_becomes_unknown_for_irrelevant_ticker_without_fallback():
    val_df = pd.DataFrame(
        [
            {
                "ticker": "9999.T",
                "earn_start_ts": _quote_ts("2025-08-06"),
                "earn_ts": _quote_ts("2025-08-06"),
                "earn_end_ts": _quote_ts("2025-08-06"),
            }
        ]
    )
    tech_df = pd.DataFrame(
        [{"ticker": "9999.T", "score": 2, "sell_score": 0}]
    )

    calls = []

    def factory(ticker):
        calls.append(ticker)
        return _CalendarTicker()

    repaired = repair_earnings_dates(
        val_df,
        tech_df,
        as_of_date="2026-09-04",
        buy_score_min=6,
        sell_score_min=5,
        pause_sec=0,
        ticker_factory=factory,
    )

    assert calls == []
    assert repaired.loc[0, "earn_date_source"] == "unknown"
    assert pd.isna(repaired.loc[0, "earn_start_ts"])
    assert pd.isna(repaired.loc[0, "earn_ts"])
    assert pd.isna(repaired.loc[0, "earn_end_ts"])
