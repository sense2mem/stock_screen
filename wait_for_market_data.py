# -*- coding: utf-8 -*-
"""Lightweight Yahoo market-data readiness probe for the daily stock screen.

The full stock screen is intentionally expensive.  This helper checks a small
set of liquid TSE names first and returns a special exit status when Yahoo has
not published the requested TSE session yet.

Exit codes:
    0   target session is available
    75  target session is not available yet (temporary/defer, not a hard error)
    1   unexpected/fatal error
"""

from __future__ import annotations

import argparse
from collections import Counter
import sys
import time

import pandas as pd
import yfinance as yf


PROBE_TICKERS = (
    "7203.T",  # Toyota
    "6758.T",  # Sony Group
    "8306.T",  # MUFG
    "9432.T",  # NTT
    "9984.T",  # SoftBank Group
)

DEFAULT_ATTEMPTS = 7
DEFAULT_INTERVAL_SECONDS = 300
MIN_READY_TICKERS = 3


def _extract_ticker_frame(df_all: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    if isinstance(df_all.columns, pd.MultiIndex):
        level0 = df_all.columns.get_level_values(0)
        level1 = df_all.columns.get_level_values(1)

        if ticker in level0:
            return df_all[ticker].copy()
        if ticker in level1:
            return df_all.xs(ticker, axis=1, level=1).copy()
        return pd.DataFrame()

    # yfinance returns a single-level frame when only one symbol survives.
    return df_all.copy()


def fetch_latest_market_dates(target_date: pd.Timestamp) -> dict[str, pd.Timestamp]:
    target = pd.Timestamp(target_date).normalize()
    start = target - pd.Timedelta(days=10)
    end = target + pd.Timedelta(days=2)

    df_all = yf.download(
        tickers=" ".join(PROBE_TICKERS),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="ticker",
    )

    latest: dict[str, pd.Timestamp] = {}

    for ticker in PROBE_TICKERS:
        frame = _extract_ticker_frame(df_all, ticker)
        if frame.empty:
            continue

        if "Close" not in frame.columns:
            continue

        valid = frame.dropna(subset=["Close"]).copy()

        if "Volume" in valid.columns:
            volume = pd.to_numeric(valid["Volume"], errors="coerce")
            valid = valid.loc[volume > 0]

        if valid.empty:
            continue

        idx = pd.DatetimeIndex(valid.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)

        latest[ticker] = pd.Timestamp(idx[-1]).normalize()

    return latest


def summarize_dates(latest: dict[str, pd.Timestamp]) -> str:
    if not latest:
        return "no probe bars"

    counts = Counter(latest.values())
    return ", ".join(
        f"{day.strftime('%Y-%m-%d')}={count}"
        for day, count in counts.most_common()
    )


def target_is_ready(
    latest: dict[str, pd.Timestamp],
    target_date: pd.Timestamp,
    required: int = MIN_READY_TICKERS,
) -> bool:
    target = pd.Timestamp(target_date).normalize()
    ready = sum(day == target for day in latest.values())
    return ready >= required


def wait_for_target(
    target_date: pd.Timestamp,
    attempts: int = DEFAULT_ATTEMPTS,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
) -> bool:
    target = pd.Timestamp(target_date).normalize()

    for attempt in range(1, attempts + 1):
        try:
            latest = fetch_latest_market_dates(target)
        except Exception as exc:
            latest = {}
            print(
                "market-data probe error:",
                repr(exc),
                flush=True,
            )

        future_dates = [
            day for day in latest.values()
            if day > target
        ]
        if future_dates:
            newest = max(future_dates)
            raise RuntimeError(
                "Yahoo probe returned a future session: "
                f"target={target.strftime('%Y-%m-%d')} "
                f"future={newest.strftime('%Y-%m-%d')}"
            )

        print(
            "market-data readiness probe:",
            f"attempt={attempt}/{attempts}",
            f"target={target.strftime('%Y-%m-%d')}",
            f"dates=[{summarize_dates(latest)}]",
            flush=True,
        )

        if target_is_ready(latest, target):
            print(
                "market-data readiness: READY",
                f"target={target.strftime('%Y-%m-%d')}",
                flush=True,
            )
            return True

        if attempt < attempts:
            print(
                "market-data readiness: NOT READY; "
                f"retrying after {interval_seconds} seconds",
                flush=True,
            )
            time.sleep(interval_seconds)

    print(
        "market-data readiness: DEFER",
        f"target={target.strftime('%Y-%m-%d')}",
        "Yahoo has not published enough target-session bars yet.",
        flush=True,
    )
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-date", required=True)
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=DEFAULT_INTERVAL_SECONDS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = pd.Timestamp(args.target_date).normalize()

    if args.attempts < 1:
        raise ValueError("--attempts must be >= 1")
    if args.interval_seconds < 0:
        raise ValueError("--interval-seconds must be >= 0")

    ready = wait_for_target(
        target,
        attempts=args.attempts,
        interval_seconds=args.interval_seconds,
    )
    return 0 if ready else 75


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("market-data readiness fatal error:", repr(exc), flush=True)
        sys.exit(1)
