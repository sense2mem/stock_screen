#!/usr/bin/env python3
"""Backtest score signals with exits after a fixed number of trading days."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger("fixed_holding_backtest")
OHLC_COLUMNS = {"Open", "High", "Low", "Close"}
OPTIONAL_COLUMNS = [
    "name", "sell_score", "close", "rsi", "adx14", "plus_di14", "minus_di14",
    "pe_ttm", "pbr", "adv20_m", "market_cap", "next_earn_date",
]
DETAIL_COLUMNS = [
    "signal_date", "ticker", "name", "score", "sell_score", "holding_days",
    "entry_date", "entry_price", "exit_date", "exit_price", "return_pct",
    "mfe_pct", "mae_pct", "status",
] + [c for c in OPTIONAL_COLUMNS if c not in {"name", "sell_score"}]
STATUS_COLUMNS = ["signal_date", "ticker", "holding_days", "status", "message"]
SUMMARY_COLUMNS = [
    "holding_days", "trades", "wins", "losses", "win_rate_pct",
    "average_return_pct", "median_return_pct", "average_win_pct",
    "average_loss_pct", "profit_factor", "best_return_pct", "worst_return_pct",
    "average_mfe_pct", "median_mfe_pct", "average_mae_pct", "median_mae_pct",
]


def _signal_date(path: Path) -> pd.Timestamp:
    match = re.search(r"screen_(\d{4}-\d{2}-\d{2})_buy\.csv$", path.name)
    if not match:
        raise ValueError(f"cannot extract signal date from {path.name}")
    return pd.Timestamp(match.group(1))


def load_signals(
    signals_dir: Path | str,
    pattern: str,
    score_min: float,
) -> tuple[pd.DataFrame, int, int]:
    """Load, normalize and de-duplicate candidate files."""
    paths = sorted(Path(signals_dir).glob(pattern))
    if not paths:
        raise ValueError(f"no input CSVs matching {Path(signals_dir) / pattern}")

    frames: list[pd.DataFrame] = []
    raw_count = 0
    screening_dates: list[pd.Timestamp] = []

    for path in paths:
        try:
            frame = pd.read_csv(path, dtype={"ticker": "string"})
            date = _signal_date(path)
        except Exception as exc:
            LOGGER.warning("Skipping invalid input %s: %s", path, exc)
            continue

        screening_dates.append(date)
        raw_count += len(frame)
        if not {"ticker", "score"}.issubset(frame.columns):
            LOGGER.warning("Skipping %s: ticker and/or score is missing", path)
            continue

        frame = frame.copy()
        frame["signal_date"] = date
        frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        frame = frame[
            frame["ticker"].notna()
            & frame["ticker"].ne("")
            & frame["score"].ge(score_min)
        ]
        frames.append(frame)

    if not frames:
        raise ValueError("no readable CSV contains the required ticker and score columns")

    signals = pd.concat(frames, ignore_index=True, sort=False)
    signals = (
        signals.sort_values(["signal_date", "ticker"])
        .drop_duplicates(["signal_date", "ticker"], keep="first")
    )
    for column in OPTIONAL_COLUMNS:
        if column not in signals:
            signals[column] = pd.NA
    signals = signals.reset_index(drop=True)

    # Keep dates on which no ticker passed score_min. Such a date breaks a
    # ticker's streak, while weekends/holidays (no screening file) do not.
    signals.attrs["screening_dates"] = sorted(set(screening_dates))
    return signals, len(paths), raw_count


def select_signals(signals: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Mark signals used; streaks are based on the ordered screening dates."""
    result = signals.copy()
    result["is_used"] = True
    result["excluded_reason"] = ""
    if mode == "all" or result.empty:
        return result

    dates = signals.attrs.get("screening_dates")
    if not dates:
        dates = pd.to_datetime(result["signal_date"]).dt.normalize().unique()
    dates = [pd.Timestamp(date).normalize() for date in sorted(dates)]
    positions = {date: i for i, date in enumerate(dates)}

    for _, group in result.groupby("ticker", sort=False):
        previous: int | None = None
        for idx in group.sort_values("signal_date").index:
            current = positions[pd.Timestamp(result.at[idx, "signal_date"]).normalize()]
            if previous is not None and current == previous + 1:
                result.at[idx, "is_used"] = False
                result.at[idx, "excluded_reason"] = "CONTINUATION_OF_STREAK"
            previous = current
    return result


def _normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        for level in range(frame.columns.nlevels):
            values = frame.columns.get_level_values(level)
            if OHLC_COLUMNS.issubset(set(values)):
                frame.columns = values
                break

    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    return frame.sort_index()


def _extract_ticker_frame(data: pd.DataFrame, ticker: str, chunk_size: int) -> pd.DataFrame:
    """Extract one ticker from either yfinance MultiIndex orientation."""
    if data is None or data.empty:
        raise ValueError("empty price response")

    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(data.columns.get_level_values(0))
        level1 = set(data.columns.get_level_values(1))
        if ticker in level0:
            return data[ticker].copy()
        if ticker in level1:
            return data.xs(ticker, axis=1, level=1).copy()
        raise KeyError(f"ticker {ticker} not present in price response")

    if chunk_size == 1 and OHLC_COLUMNS.issubset(set(data.columns)):
        return data.copy()

    raise KeyError(f"ticker {ticker} not present in flat multi-ticker response")


def evaluate_signal(
    signal: pd.Series,
    prices: pd.DataFrame,
    holdings: Iterable[int],
) -> list[dict]:
    """Evaluate one signal against injected OHLC prices (no network dependency)."""
    prices = _normalize_prices(prices)
    signal_date = pd.Timestamp(signal["signal_date"])
    if signal_date.tzinfo is not None:
        signal_date = signal_date.tz_localize(None)
    signal_date = signal_date.normalize()

    base = {
        c: signal.get(c, pd.NA)
        for c in ["signal_date", "ticker", "name", "score", "sell_score"]
        + OPTIONAL_COLUMNS[2:]
    }
    future = prices.loc[prices.index > signal_date]
    rows: list[dict] = []

    for holding in holdings:
        row = {
            **base,
            "holding_days": int(holding),
            "entry_date": pd.NaT,
            "entry_price": np.nan,
            "exit_date": pd.NaT,
            "exit_price": np.nan,
            "return_pct": np.nan,
            "mfe_pct": np.nan,
            "mae_pct": np.nan,
            "status": "INVALID_SIGNAL",
        }
        if future.empty:
            row["status"] = "NO_NEXT_TRADING_DAY"
        elif not OHLC_COLUMNS.issubset(set(prices.columns)):
            row["status"] = "NO_PRICE_DATA"
        else:
            entry = pd.to_numeric(
                pd.Series([future.iloc[0]["Open"]]), errors="coerce"
            ).iloc[0]
            if pd.isna(entry) or entry <= 0:
                row["status"] = "NO_PRICE_DATA"
            else:
                row["entry_date"] = future.index[0]
                row["entry_price"] = float(entry)
                if len(future) < holding:
                    row["status"] = "OPEN_INSUFFICIENT_DAYS"
                else:
                    window = future.iloc[:holding]
                    exit_price = pd.to_numeric(
                        pd.Series([window.iloc[-1]["Close"]]), errors="coerce"
                    ).iloc[0]
                    highs = pd.to_numeric(window["High"], errors="coerce")
                    lows = pd.to_numeric(window["Low"], errors="coerce")
                    if pd.isna(exit_price) or highs.dropna().empty or lows.dropna().empty:
                        row["status"] = "NO_PRICE_DATA"
                    else:
                        row.update(
                            {
                                "exit_date": window.index[-1],
                                "exit_price": float(exit_price),
                                "return_pct": (exit_price / entry - 1) * 100,
                                "mfe_pct": (highs.max() / entry - 1) * 100,
                                "mae_pct": (lows.min() / entry - 1) * 100,
                                "status": "CLOSED",
                            }
                        )
        rows.append(row)
    return rows


def summarize_results(detail: pd.DataFrame, holdings: Iterable[int]) -> pd.DataFrame:
    rows: list[dict] = []
    closed = detail[detail["status"].eq("CLOSED")] if not detail.empty else detail
    for holding in holdings:
        group = closed[closed["holding_days"].eq(holding)]
        returns = pd.to_numeric(
            group.get("return_pct", pd.Series(dtype=float)), errors="coerce"
        ).dropna()
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        loss_sum = -losses.sum()
        rows.append(
            {
                "holding_days": holding,
                "trades": len(returns),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate_pct": len(wins) / len(returns) * 100 if len(returns) else np.nan,
                "average_return_pct": returns.mean(),
                "median_return_pct": returns.median(),
                "average_win_pct": wins.mean(),
                "average_loss_pct": losses.mean(),
                "profit_factor": wins.sum() / loss_sum if loss_sum > 0 else np.nan,
                "best_return_pct": returns.max(),
                "worst_return_pct": returns.min(),
                "average_mfe_pct": group["mfe_pct"].mean(),
                "median_mfe_pct": group["mfe_pct"].median(),
                "average_mae_pct": group["mae_pct"].mean(),
                "median_mae_pct": group["mae_pct"].median(),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def download_prices(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_size: int = 20,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    prices: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    for offset in range(0, len(tickers), chunk_size):
        chunk = tickers[offset : offset + chunk_size]
        try:
            data = yf.download(
                chunk,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="ticker",
            )
        except Exception as exc:
            for ticker in chunk:
                failures[ticker] = str(exc)
            continue

        for ticker in chunk:
            try:
                frame = _extract_ticker_frame(data, ticker, len(chunk))
                frame = _normalize_prices(frame).dropna(how="all")
                if frame.empty or not OHLC_COLUMNS.issubset(set(frame.columns)):
                    raise ValueError("empty or incomplete price response")
                prices[ticker] = frame
            except Exception as exc:
                failures[ticker] = str(exc)

    return prices, failures


def _write_outputs(
    output: Path,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    status: pd.DataFrame,
    marked: pd.DataFrame,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "fixed_holding_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "fixed_holding_summary.csv", index=False, encoding="utf-8-sig")
    status.to_csv(output / "fixed_holding_status.csv", index=False, encoding="utf-8-sig")
    marked.to_csv(output / "signals_used.csv", index=False, encoding="utf-8-sig")


def run(args: argparse.Namespace) -> None:
    signals, file_count, raw_count = load_signals(
        args.signals_dir, args.signal_pattern, args.score_min
    )
    marked = select_signals(signals, args.signal_mode)
    used = marked[marked["is_used"]].copy()
    output = Path(args.output_dir)

    LOGGER.info(
        "CSV files=%d raw rows=%d score>=%.2f unique signals=%d",
        file_count,
        raw_count,
        args.score_min,
        len(signals),
    )
    LOGGER.info(
        "signals after %s=%d tickers=%d",
        args.signal_mode,
        len(used),
        used["ticker"].nunique() if not used.empty else 0,
    )

    if used.empty:
        LOGGER.warning("No qualifying signals; writing empty reports")
        detail = pd.DataFrame(columns=DETAIL_COLUMNS)
        summary = summarize_results(detail, args.holdings)
        status = pd.DataFrame(columns=STATUS_COLUMNS)
        _write_outputs(output, detail, summary, status, marked)
        LOGGER.info("output directory: %s", output.resolve())
        return

    tickers = sorted(used["ticker"].unique())
    start = used["signal_date"].min() - pd.Timedelta(days=7)
    end = pd.Timestamp.now().tz_localize(None).normalize() + pd.Timedelta(days=2)
    prices, failures = download_prices(tickers, start, end)
    LOGGER.info("price downloads successful=%d failed=%d", len(prices), len(failures))

    detail_rows: list[dict] = []
    status_rows: list[dict] = []
    for _, signal in used.iterrows():
        ticker = signal["ticker"]
        if ticker not in prices:
            message = failures.get(ticker, "no price data")
            for holding in args.holdings:
                row = {c: signal.get(c, pd.NA) for c in DETAIL_COLUMNS}
                row.update(
                    {
                        "holding_days": holding,
                        "status": "PRICE_DOWNLOAD_FAILED",
                    }
                )
                detail_rows.append(row)
                status_rows.append(
                    {
                        "signal_date": signal["signal_date"],
                        "ticker": ticker,
                        "holding_days": holding,
                        "status": "PRICE_DOWNLOAD_FAILED",
                        "message": message,
                    }
                )
            continue

        for row in evaluate_signal(signal, prices[ticker], args.holdings):
            detail_rows.append(row)
            if row["status"] != "CLOSED":
                status_rows.append(
                    {
                        "signal_date": row["signal_date"],
                        "ticker": ticker,
                        "holding_days": row["holding_days"],
                        "status": row["status"],
                        "message": "fixed holding period cannot yet be evaluated",
                    }
                )

    detail = pd.DataFrame(detail_rows).reindex(columns=DETAIL_COLUMNS)
    summary = summarize_results(detail, args.holdings)
    status = pd.DataFrame(status_rows, columns=STATUS_COLUMNS)
    _write_outputs(output, detail, summary, status, marked)

    for holding in args.holdings:
        LOGGER.info(
            "holding=%d CLOSED=%d OPEN=%d",
            holding,
            int(
                (
                    (detail.holding_days == holding)
                    & (detail.status == "CLOSED")
                ).sum()
            ),
            int(
                (
                    (detail.holding_days == holding)
                    & (detail.status == "OPEN_INSUFFICIENT_DAYS")
                ).sum()
            ),
        )
    LOGGER.info("output directory: %s", output.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-dir", default="fixed_holding_input")
    parser.add_argument("--signal-pattern", default="screen_*_buy.csv")
    parser.add_argument("--score-min", type=float, default=8)
    parser.add_argument(
        "--holdings", type=int, nargs="+", default=[5, 10, 20, 40, 50, 60]
    )
    parser.add_argument(
        "--signal-mode",
        choices=["all", "first_in_streak"],
        default="first_in_streak",
    )
    parser.add_argument("--output-dir", default="fixed_holding_report")
    args = parser.parse_args()
    if not args.holdings or any(value < 1 for value in args.holdings):
        parser.error("--holdings must contain positive integers")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
