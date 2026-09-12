#!/usr/bin/env python3
"""Compare conventional signal-day condition C trades with matched-period TOPIX returns."""
from __future__ import annotations

import argparse
import logging
import os
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("condition_c_topix_analysis")
TOKYO = ZoneInfo("Asia/Tokyo")
MARKET_CLOSE = time(15, 30)
DEFAULT_BENCHMARK_TICKER = "998405.T"
REQUIRED_STOCK_COLUMNS = {"Open", "High", "Low", "Close"}
REQUIRED_BENCHMARK_COLUMNS = {"Open", "Close"}
DETAIL_COLUMNS = [
    "signal_date", "ticker", "name", "score", "sell_score",
    "entry_date", "entry_price", "stop_loss_pct", "stop_price",
    "stop_hit", "stop_date", "holding_days_to_exit", "exit_reason",
    "strategy_exit_date", "strategy_exit_price", "strategy_return_pct",
    "no_stop_60d_return_pct", "benchmark_ticker", "benchmark_entry_date",
    "benchmark_entry_open", "benchmark_exit_date", "benchmark_exit_close",
    "benchmark_return_pct", "excess_return_pct", "outperformed_benchmark",
    "status", "message",
]
SUMMARY_COLUMNS = [
    "segment", "signals", "closed_trades", "stop_loss_trades", "held_60d_trades",
    "average_strategy_return_pct", "median_strategy_return_pct", "strategy_win_rate_pct",
    "average_benchmark_return_pct", "median_benchmark_return_pct",
    "average_excess_return_pct", "median_excess_return_pct", "outperform_rate_pct",
    "average_no_stop_60d_return_pct", "stop_rule_return_change_pct",
    "best_strategy_return_pct", "worst_strategy_return_pct",
]


def normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    return result.sort_index()


def confirmed_prices(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop today's partial Japanese daily bar before the official close."""
    result = normalize_prices(frame)
    current = pd.Timestamp.now(tz=TOKYO) if now is None else pd.Timestamp(now)
    current = current.tz_localize(TOKYO) if current.tzinfo is None else current.tz_convert(TOKYO)
    today = current.tz_localize(None).normalize()
    if current.time() < MARKET_CLOSE and today in result.index:
        result = result.loc[result.index < today]
    return result


def _pct(exit_price: object, entry_price: object) -> float:
    exit_value = pd.to_numeric(pd.Series([exit_price]), errors="coerce").iloc[0]
    entry_value = pd.to_numeric(pd.Series([entry_price]), errors="coerce").iloc[0]
    if pd.isna(exit_value) or pd.isna(entry_value) or float(entry_value) <= 0:
        return np.nan
    return (float(exit_value) / float(entry_value) - 1.0) * 100.0


def _empty_row(signal: pd.Series, benchmark_ticker: str, stop_loss_pct: float) -> dict:
    return {
        "signal_date": pd.Timestamp(signal.get("signal_date")).normalize(),
        "ticker": signal.get("ticker"), "name": signal.get("name"),
        "score": signal.get("score"), "sell_score": signal.get("sell_score"),
        "entry_date": pd.NaT, "entry_price": np.nan,
        "stop_loss_pct": -abs(float(stop_loss_pct)), "stop_price": np.nan,
        "stop_hit": False, "stop_date": pd.NaT, "holding_days_to_exit": pd.NA,
        "exit_reason": pd.NA, "strategy_exit_date": pd.NaT,
        "strategy_exit_price": np.nan, "strategy_return_pct": np.nan,
        "no_stop_60d_return_pct": np.nan, "benchmark_ticker": benchmark_ticker,
        "benchmark_entry_date": pd.NaT, "benchmark_entry_open": np.nan,
        "benchmark_exit_date": pd.NaT, "benchmark_exit_close": np.nan,
        "benchmark_return_pct": np.nan, "excess_return_pct": np.nan,
        "outperformed_benchmark": pd.NA, "status": "INVALID_SIGNAL", "message": "",
    }


def evaluate_trade(
    signal: pd.Series,
    stock_prices: pd.DataFrame,
    benchmark_prices: pd.DataFrame,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    stop_loss_pct: float = 14.0,
    holding_days: int = 60,
) -> dict:
    """Evaluate one matured trade and compare it with TOPIX over the same actual holding period."""
    stock = normalize_prices(stock_prices)
    benchmark = normalize_prices(benchmark_prices)
    row = _empty_row(signal, benchmark_ticker, stop_loss_pct)
    signal_date = row["signal_date"]
    future = stock.loc[stock.index > signal_date]

    if future.empty:
        row.update(status="NO_NEXT_TRADING_DAY", message="no trading day after signal date")
        return row
    if not REQUIRED_STOCK_COLUMNS.issubset(stock.columns):
        row.update(status="NO_PRICE_DATA", message="stock OHLC columns are incomplete")
        return row

    entry_date = pd.Timestamp(future.index[0]).normalize()
    entry_price = pd.to_numeric(pd.Series([future.iloc[0]["Open"]]), errors="coerce").iloc[0]
    if pd.isna(entry_price) or float(entry_price) <= 0:
        row.update(status="NO_PRICE_DATA", message="stock entry open is unavailable")
        return row

    entry_price = float(entry_price)
    stop_return = -abs(float(stop_loss_pct))
    stop_price = entry_price * (1.0 + stop_return / 100.0)
    row.update(entry_date=entry_date, entry_price=entry_price, stop_price=stop_price)

    # Require the complete 60-day cohort to avoid censoring recent open trades.
    if len(future) < holding_days:
        row.update(
            status="OPEN_INSUFFICIENT_DAYS",
            message=f"only {len(future)} of {holding_days} trading days are observable",
        )
        return row

    window = future.iloc[:holding_days]
    lows = pd.to_numeric(window["Low"], errors="coerce")
    close_60 = pd.to_numeric(pd.Series([window.iloc[-1]["Close"]]), errors="coerce").iloc[0]
    if lows.dropna().empty or pd.isna(close_60):
        row.update(status="NO_PRICE_DATA", message="stock holding-period prices are incomplete")
        return row

    no_stop_return = _pct(close_60, entry_price)
    stop_hits = lows[lows <= stop_price]
    if not stop_hits.empty:
        exit_date = pd.Timestamp(stop_hits.index[0]).normalize()
        holding_days_to_exit = int(window.index.get_loc(exit_date)) + 1
        exit_reason = "STOP_LOSS"
        strategy_exit_price = stop_price
        strategy_return = stop_return
        stop_hit = True
        stop_date = exit_date
    else:
        exit_date = pd.Timestamp(window.index[-1]).normalize()
        holding_days_to_exit = holding_days
        exit_reason = "HOLD_60D"
        strategy_exit_price = float(close_60)
        strategy_return = no_stop_return
        stop_hit = False
        stop_date = pd.NaT

    row.update(
        stop_hit=stop_hit, stop_date=stop_date, holding_days_to_exit=holding_days_to_exit,
        exit_reason=exit_reason, strategy_exit_date=exit_date,
        strategy_exit_price=strategy_exit_price, strategy_return_pct=strategy_return,
        no_stop_60d_return_pct=no_stop_return,
    )

    if not REQUIRED_BENCHMARK_COLUMNS.issubset(benchmark.columns):
        row.update(status="BENCHMARK_DATA_MISSING", message="benchmark Open/Close columns are incomplete")
        return row
    if entry_date not in benchmark.index or exit_date not in benchmark.index:
        row.update(
            status="BENCHMARK_DATE_MISSING",
            message=f"benchmark lacks entry or exit date: {entry_date.date()} / {exit_date.date()}",
        )
        return row

    benchmark_entry = pd.to_numeric(
        pd.Series([benchmark.at[entry_date, "Open"]]), errors="coerce"
    ).iloc[0]
    benchmark_exit = pd.to_numeric(
        pd.Series([benchmark.at[exit_date, "Close"]]), errors="coerce"
    ).iloc[0]
    if pd.isna(benchmark_entry) or pd.isna(benchmark_exit) or float(benchmark_entry) <= 0:
        row.update(status="BENCHMARK_DATA_MISSING", message="benchmark entry or exit price is unavailable")
        return row

    benchmark_return = _pct(benchmark_exit, benchmark_entry)
    excess_return = strategy_return - benchmark_return
    row.update(
        benchmark_entry_date=entry_date, benchmark_entry_open=float(benchmark_entry),
        benchmark_exit_date=exit_date, benchmark_exit_close=float(benchmark_exit),
        benchmark_return_pct=benchmark_return, excess_return_pct=excess_return,
        outperformed_benchmark=bool(excess_return > 0), status="CLOSED",
        message="matched-period comparison completed",
    )
    return row


def build_analysis(
    signals: pd.DataFrame,
    stock_prices: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    stop_loss_pct: float = 14.0,
    holding_days: int = 60,
    failures: dict[str, str] | None = None,
) -> pd.DataFrame:
    failures = failures or {}
    rows: list[dict] = []
    source = signals.copy()
    source["signal_date"] = pd.to_datetime(source["signal_date"], errors="coerce").dt.normalize()
    source["ticker"] = source["ticker"].astype("string")
    source = source.dropna(subset=["signal_date", "ticker"]).drop_duplicates(
        ["signal_date", "ticker"], keep="first"
    )
    for _, signal in source.iterrows():
        ticker = str(signal["ticker"])
        if ticker not in stock_prices:
            row = _empty_row(signal, benchmark_ticker, stop_loss_pct)
            row.update(
                status="PRICE_DOWNLOAD_FAILED",
                message=failures.get(ticker, "stock price download failed"),
            )
        else:
            row = evaluate_trade(
                signal, stock_prices[ticker], benchmark_prices, benchmark_ticker,
                stop_loss_pct, holding_days,
            )
        rows.append(row)
    detail = pd.DataFrame(rows).reindex(columns=DETAIL_COLUMNS)
    if not detail.empty:
        detail = detail.sort_values(["signal_date", "ticker"], kind="stable").reset_index(drop=True)
    return detail


def _segment_summary(segment: str, frame: pd.DataFrame, total_signals: int) -> dict:
    closed = frame[frame["status"].eq("CLOSED")]
    strategy = pd.to_numeric(closed.get("strategy_return_pct"), errors="coerce").dropna()
    benchmark = pd.to_numeric(closed.get("benchmark_return_pct"), errors="coerce").dropna()
    excess = pd.to_numeric(closed.get("excess_return_pct"), errors="coerce").dropna()
    no_stop = pd.to_numeric(closed.get("no_stop_60d_return_pct"), errors="coerce").dropna()
    return {
        "segment": segment,
        "signals": total_signals,
        "closed_trades": len(closed),
        "stop_loss_trades": int(closed["exit_reason"].eq("STOP_LOSS").sum()),
        "held_60d_trades": int(closed["exit_reason"].eq("HOLD_60D").sum()),
        "average_strategy_return_pct": strategy.mean(),
        "median_strategy_return_pct": strategy.median(),
        "strategy_win_rate_pct": strategy.gt(0).mean() * 100.0 if len(strategy) else np.nan,
        "average_benchmark_return_pct": benchmark.mean(),
        "median_benchmark_return_pct": benchmark.median(),
        "average_excess_return_pct": excess.mean(),
        "median_excess_return_pct": excess.median(),
        "outperform_rate_pct": excess.gt(0).mean() * 100.0 if len(excess) else np.nan,
        "average_no_stop_60d_return_pct": no_stop.mean(),
        "stop_rule_return_change_pct": strategy.mean() - no_stop.mean() if len(strategy) and len(no_stop) else np.nan,
        "best_strategy_return_pct": strategy.max(),
        "worst_strategy_return_pct": strategy.min(),
    }


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    rows = [_segment_summary("ALL", detail, len(detail))]
    closed = detail[detail["status"].eq("CLOSED")]
    for reason in ("STOP_LOSS", "HOLD_60D"):
        group = closed[closed["exit_reason"].eq(reason)]
        rows.append(_segment_summary(reason, group, len(group)))
    return pd.DataFrame(rows).reindex(columns=SUMMARY_COLUMNS)


def _display(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value).replace("|", "\\|")


def build_markdown(summary: pd.DataFrame, detail: pd.DataFrame, benchmark_ticker: str,
                   stop_loss_pct: float, holding_days: int) -> str:
    lines = [
        "## Conventional signal-day condition C vs TOPIX", "",
        f"- Strategy: {-abs(stop_loss_pct):.1f}% stop; otherwise exit at trading day {holding_days} close.",
        f"- Benchmark: `{benchmark_ticker}` entry-day open to the strategy's actual exit-day close.",
        "- Only cohorts with all 60 trading days observable are included, avoiding censoring bias.",
        "- Benchmark is the TOPIX price index; execution costs, gap-through-stop losses and taxes are excluded.", "",
        "| Segment | Signals | Closed | Stops | Hold 60d | Strategy avg % | TOPIX avg % | Excess avg % | Excess median % | Outperform % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        keys = ["segment", "signals", "closed_trades", "stop_loss_trades", "held_60d_trades",
                "average_strategy_return_pct", "average_benchmark_return_pct",
                "average_excess_return_pct", "median_excess_return_pct", "outperform_rate_pct"]
        lines.append("| " + " | ".join(_display(row.get(key)) for key in keys) + " |")
    lines += ["", "### Matched-period trades", "",
              "| Signal | Ticker | Exit reason | Exit | Strategy % | TOPIX % | Excess % | Beat TOPIX |",
              "|---|---|---|---|---:|---:|---:|---|"]
    closed = detail[detail["status"].eq("CLOSED")]
    for _, row in closed.head(200).iterrows():
        signal_date = pd.Timestamp(row.signal_date).strftime("%Y-%m-%d")
        exit_date = pd.Timestamp(row.strategy_exit_date).strftime("%Y-%m-%d")
        values = [signal_date, row.ticker, row.exit_reason, exit_date,
                  row.strategy_return_pct, row.benchmark_return_pct,
                  row.excess_return_pct, row.outperformed_benchmark]
        lines.append("| " + " | ".join(_display(value) for value in values) + " |")
    if closed.empty:
        lines.append("| - | - | - | - | - | - | - | No completed comparisons |")
    return "\n".join(lines) + "\n"


def write_outputs(output_dir: Path | str, detail: pd.DataFrame, summary: pd.DataFrame) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "condition_c_topix_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "condition_c_topix_summary.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", default="signal_path_report/condition_c_signal_day.csv")
    parser.add_argument("--output-dir", default="signal_path_report")
    parser.add_argument("--benchmark-ticker", default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument("--stop-loss-pct", type=float, default=14.0)
    parser.add_argument("--holding-days", type=int, default=60)
    parser.add_argument("--github-summary", default=os.environ.get("GITHUB_STEP_SUMMARY", ""))
    args = parser.parse_args()
    if args.stop_loss_pct <= 0 or args.stop_loss_pct >= 100:
        parser.error("--stop-loss-pct must be greater than 0 and less than 100")
    if args.holding_days < 1:
        parser.error("--holding-days must be positive")
    return args


def run(args: argparse.Namespace) -> None:
    from fixed_holding_backtest import download_prices

    path = Path(args.signals)
    signals = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": "string"})
    if signals.empty:
        detail = pd.DataFrame(columns=DETAIL_COLUMNS)
        result_summary = summarize(detail)
        write_outputs(args.output_dir, detail, result_summary)
        return

    signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce").dt.normalize()
    tickers = sorted(signals["ticker"].dropna().astype(str).unique())
    start = signals["signal_date"].min() - pd.Timedelta(days=7)
    end = pd.Timestamp.now().tz_localize(None).normalize() + pd.Timedelta(days=2)
    downloaded, failures = download_prices(tickers + [args.benchmark_ticker], start, end)
    benchmark = downloaded.pop(args.benchmark_ticker, pd.DataFrame())
    stock_prices = {ticker: confirmed_prices(frame) for ticker, frame in downloaded.items()}
    benchmark = confirmed_prices(benchmark) if not benchmark.empty else benchmark
    if benchmark.empty:
        LOGGER.warning("Benchmark download failed: %s", failures.get(args.benchmark_ticker, "no data"))

    detail = build_analysis(
        signals, stock_prices, benchmark, args.benchmark_ticker,
        args.stop_loss_pct, args.holding_days, failures,
    )
    result_summary = summarize(detail)
    write_outputs(args.output_dir, detail, result_summary)
    if args.github_summary:
        with open(args.github_summary, "a", encoding="utf-8") as handle:
            handle.write(build_markdown(
                result_summary, detail, args.benchmark_ticker,
                args.stop_loss_pct, args.holding_days,
            ))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
