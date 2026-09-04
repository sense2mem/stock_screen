# -*- coding: utf-8 -*-
"""Validate and repair upcoming earnings dates used by the daily screen.

Yahoo's quote endpoint occasionally leaves Japanese equities with historical
``earningsTimestamp*`` values. The screen previously treated those historical
values as the *next* earnings date, producing misleading negative
``days_to_earnings`` values.

This module keeps the fast batch quote lookup as the primary source, but:
1. rejects quote timestamps that are earlier than the market-session date;
2. selects the earliest remaining future quote timestamp; and
3. for technically relevant rows whose quote date is stale/missing, performs a
   targeted yfinance ticker-level fallback instead of querying every listed
   stock one-by-one.
"""

from __future__ import annotations

from collections.abc import Iterable
import time
from typing import Any

import numpy as np
import pandas as pd


EARNINGS_TS_COLUMNS = (
    "earn_start_ts",
    "earn_ts",
    "earn_end_ts",
)


def _normalize_date(value: Any) -> pd.Timestamp | None:
    """Normalize an arbitrary date-like value to a timezone-naive date."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None

    if pd.isna(ts):
        return None

    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Tokyo").tz_localize(None)

    return ts.normalize()


def _unix_to_date(value: Any) -> pd.Timestamp | None:
    """Convert Yahoo seconds/milliseconds timestamps to a JST calendar date."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(raw):
        return None

    if abs(raw) > 1e11:
        raw /= 1000.0

    try:
        return (
            pd.to_datetime(raw, unit="s", utc=True)
            .tz_convert("Asia/Tokyo")
            .normalize()
            .tz_localize(None)
        )
    except Exception:
        return None


def _date_to_unix_seconds(value: Any) -> float:
    """Encode a calendar date as a Unix timestamp that round-trips in JST."""
    day = _normalize_date(value)
    if day is None:
        return np.nan

    return float(day.tz_localize("Asia/Tokyo").timestamp())


def _iter_date_values(value: Any) -> Iterable[Any]:
    """Yield date-like leaves from common yfinance calendar structures."""
    if value is None:
        return

    if isinstance(value, pd.DataFrame):
        for idx in value.index:
            yield idx
        for column in value.columns:
            for item in value[column].tolist():
                yield from _iter_date_values(item)
        return

    if isinstance(value, pd.Series):
        for idx in value.index:
            yield idx
        for item in value.tolist():
            yield from _iter_date_values(item)
        return

    if isinstance(value, pd.Index):
        for item in value.tolist():
            yield from _iter_date_values(item)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower().replace("_", " ")
            if "earn" in key_text and "date" in key_text:
                yield from _iter_date_values(item)
        return

    if isinstance(value, (list, tuple, set, np.ndarray)):
        for item in value:
            yield from _iter_date_values(item)
        return

    yield value


def _pick_future_date(values: Iterable[Any], as_of_date: Any) -> pd.Timestamp | None:
    """Pick the earliest date on or after ``as_of_date``."""
    as_of = _normalize_date(as_of_date)
    if as_of is None:
        raise ValueError("as_of_date is required")

    candidates: list[pd.Timestamp] = []

    for value in values:
        day = _normalize_date(value)
        if day is not None and day >= as_of:
            candidates.append(day)

    return min(candidates) if candidates else None


def _pick_future_quote_date(
    row: pd.Series,
    as_of_date: Any,
) -> tuple[pd.Timestamp | None, bool]:
    """Return earliest future Yahoo quote date and whether past values existed."""
    as_of = _normalize_date(as_of_date)
    if as_of is None:
        raise ValueError("as_of_date is required")

    future: list[pd.Timestamp] = []
    saw_past = False

    for column in EARNINGS_TS_COLUMNS:
        day = _unix_to_date(row.get(column))
        if day is None:
            continue
        if day >= as_of:
            future.append(day)
        else:
            saw_past = True

    return (min(future) if future else None, saw_past)


def _extract_calendar_future_date(
    calendar_value: Any,
    as_of_date: Any,
) -> pd.Timestamp | None:
    """Extract a future earnings date from ``Ticker.calendar`` output."""
    values = list(_iter_date_values(calendar_value))
    return _pick_future_date(values, as_of_date)


def _extract_earnings_dates_future_date(
    earnings_dates: Any,
    as_of_date: Any,
) -> pd.Timestamp | None:
    """Extract a future earnings date from ``Ticker.get_earnings_dates``."""
    if earnings_dates is None:
        return None

    if isinstance(earnings_dates, pd.DataFrame):
        values = list(earnings_dates.index)
    else:
        values = list(_iter_date_values(earnings_dates))

    return _pick_future_date(values, as_of_date)


def fetch_ticker_future_earnings_date(
    ticker: str,
    as_of_date: Any,
    ticker_factory=None,
) -> tuple[pd.Timestamp | None, str, str]:
    """Fetch one ticker's nearest future earnings date.

    Returns ``(date, source, error)``. Failures are intentionally non-fatal;
    callers can keep the earnings date unknown rather than reusing stale data.
    ``ticker_factory`` exists for deterministic tests.
    """
    if ticker_factory is None:
        import yfinance as yf

        ticker_factory = yf.Ticker

    errors: list[str] = []

    try:
        ticker_obj = ticker_factory(str(ticker))
    except Exception as exc:
        return None, "unknown", f"ticker_init:{type(exc).__name__}:{exc}"[:200]

    try:
        calendar_value = ticker_obj.calendar
        day = _extract_calendar_future_date(calendar_value, as_of_date)
        if day is not None:
            return day, "yfinance_calendar", ""
    except Exception as exc:
        errors.append(f"calendar:{type(exc).__name__}:{exc}")

    try:
        get_earnings_dates = getattr(ticker_obj, "get_earnings_dates", None)
        if callable(get_earnings_dates):
            earnings_dates = get_earnings_dates(limit=12)
            day = _extract_earnings_dates_future_date(
                earnings_dates,
                as_of_date,
            )
            if day is not None:
                return day, "yfinance_earnings_dates", ""
    except Exception as exc:
        errors.append(f"earnings_dates:{type(exc).__name__}:{exc}")

    return None, "unknown", ";".join(errors)[:200]


def repair_earnings_dates(
    val_df: pd.DataFrame,
    tech_df: pd.DataFrame,
    as_of_date: Any,
    buy_score_min: int,
    sell_score_min: int,
    pause_sec: float = 0.05,
    ticker_factory=None,
) -> pd.DataFrame:
    """Return valuation data with stale earnings timestamps removed/repaired.

    Batch quote data is sanitized for every ticker. Ticker-level network
    fallback is limited to rows that can materially affect the current BUY or
    technical SELL candidate set, preventing thousands of serial requests.
    """
    out = val_df.copy()

    if "ticker" not in out.columns:
        out["ticker"] = ""

    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()

    for column in EARNINGS_TS_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["earn_date_source"] = "unknown"
    out["earn_date_raw_stale"] = False
    out["earn_date_lookup_error"] = ""

    needs_fallback: set[str] = set()

    for idx, row in out.iterrows():
        future_day, saw_past = _pick_future_quote_date(row, as_of_date)
        out.at[idx, "earn_date_raw_stale"] = bool(saw_past)

        if future_day is None:
            for column in EARNINGS_TS_COLUMNS:
                out.at[idx, column] = np.nan
            needs_fallback.add(str(row.get("ticker", "")).upper())
            continue

        unix_ts = _date_to_unix_seconds(future_day)
        for column in EARNINGS_TS_COLUMNS:
            out.at[idx, column] = unix_ts
        out.at[idx, "earn_date_source"] = "yahoo_quote_future"

    tech = tech_df.copy()
    if "ticker" not in tech.columns:
        tech["ticker"] = ""
    tech["ticker"] = tech["ticker"].astype(str).str.strip().str.upper()

    for column in ("score", "sell_score"):
        if column not in tech.columns:
            tech[column] = 0
        tech[column] = pd.to_numeric(tech[column], errors="coerce").fillna(0)

    relevant = set(
        tech.loc[
            tech["score"].ge(buy_score_min)
            | tech["sell_score"].ge(sell_score_min),
            "ticker",
        ]
        .dropna()
        .astype(str)
        .str.upper()
    )

    fallback_tickers = sorted(
        ticker for ticker in (needs_fallback & relevant) if ticker
    )

    for ticker in fallback_tickers:
        day, source, error = fetch_ticker_future_earnings_date(
            ticker,
            as_of_date,
            ticker_factory=ticker_factory,
        )

        mask = out["ticker"].eq(ticker)

        if day is not None:
            unix_ts = _date_to_unix_seconds(day)
            for column in EARNINGS_TS_COLUMNS:
                out.loc[mask, column] = unix_ts
            out.loc[mask, "earn_date_source"] = source
        else:
            out.loc[mask, "earn_date_source"] = "unknown"

        if error:
            out.loc[mask, "earn_date_lookup_error"] = error

        if pause_sec > 0:
            time.sleep(pause_sec)

    return out
