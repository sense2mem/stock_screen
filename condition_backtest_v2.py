#!/usr/bin/env python3
"""Harden condition backtests: confirmed bars, 60-day triggers and deduped trades."""
from __future__ import annotations

import argparse
import logging
import os
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import condition_backtest as legacy
from fixed_holding_backtest import download_prices, load_signals, select_signals

LOGGER = logging.getLogger("condition_backtest_v2")
TOKYO = ZoneInfo("Asia/Tokyo")
MARKET_CLOSE = time(15, 30)
CONDITIONS = ("A", "B", "C")
HOLDINGS = (5, 10, 20, 40, 50, 60)


def confirmed_prices(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    result = legacy.normalize_prices(frame)
    current = pd.Timestamp.now(tz=TOKYO) if now is None else pd.Timestamp(now)
    current = current.tz_localize(TOKYO) if current.tzinfo is None else current.tz_convert(TOKYO)
    today = current.tz_localize(None).normalize()
    if current.time() < MARKET_CLOSE and today in result.index:
        result = result.loc[result.index < today]
    return result


def limited_first_dates(flags: pd.DataFrame, signal_date: object, days: int = 60) -> dict[str, pd.Timestamp]:
    date = pd.Timestamp(signal_date)
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    candidate = flags.loc[flags.index >= date.normalize()].iloc[: days + 1]
    result = {}
    for condition in CONDITIONS:
        hits = candidate.index[candidate[condition]]
        if len(hits):
            result[condition] = pd.Timestamp(hits[0]).normalize()
    return result


def trigger_day(prices: pd.DataFrame, signal_date: object, condition_date: object):
    dates = prices.index[prices.index >= pd.Timestamp(signal_date).normalize()]
    positions = np.flatnonzero(dates == pd.Timestamp(condition_date).normalize())
    return int(positions[0]) if len(positions) else pd.NA


def add_trigger_days(detail: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = detail.copy()
    result.insert(
        result.columns.get_loc("condition_first_date") + 1,
        "condition_trigger_day",
        [
            trigger_day(prices[str(row.ticker)], row.signal_date, row.condition_first_date)
            if str(row.ticker) in prices and pd.notna(row.condition_first_date) else pd.NA
            for _, row in result.iterrows()
        ],
    )
    return result


def unique_entries(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail.copy()
    return (
        detail.sort_values(["signal_date", "ticker", "condition", "entry_date"], kind="stable")
        .drop_duplicates(["ticker", "condition", "entry_date"], keep="first")
        .reset_index(drop=True)
    )


def summary_columns() -> list[str]:
    columns = [
        "condition", "source_signals", "triggered_signals", "unique_entries",
        "duplicate_events_removed", "not_triggered_signals", "price_download_failed",
        "closed_60d", "open_60d",
    ]
    for h in HOLDINGS:
        columns += [f"observed_{h}d", f"average_{h}d_return_pct", f"median_{h}d_return_pct", f"win_rate_{h}d_pct"]
    return columns + [
        "average_max_return_60d_pct", "median_max_return_60d_pct",
        "average_max_drawdown_60d_pct", "median_max_drawdown_60d_pct",
    ]


def summarize(detail: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    unique = unique_entries(detail)
    rows = []
    for condition in CONDITIONS:
        raw = detail[detail.condition.eq(condition)] if not detail.empty else detail
        group = unique[unique.condition.eq(condition)] if not unique.empty else unique
        s = status[status.condition.eq(condition)] if not status.empty else status
        row = {
            "condition": condition, "source_signals": len(s), "triggered_signals": len(raw),
            "unique_entries": len(group), "duplicate_events_removed": len(raw)-len(group),
            "not_triggered_signals": int(s.status.eq("NOT_TRIGGERED").sum()) if not s.empty else 0,
            "price_download_failed": int(s.status.eq("PRICE_DOWNLOAD_FAILED").sum()) if not s.empty else 0,
            "closed_60d": int(group.status.eq("CLOSED").sum()) if not group.empty else 0,
            "open_60d": int(group.status.eq("OPEN_INSUFFICIENT_DAYS").sum()) if not group.empty else 0,
        }
        for h in HOLDINGS:
            values = pd.to_numeric(group.get(f"return_{h}d_pct", pd.Series(dtype=float)), errors="coerce").dropna()
            row.update({f"observed_{h}d": len(values), f"average_{h}d_return_pct": values.mean(),
                        f"median_{h}d_return_pct": values.median(),
                        f"win_rate_{h}d_pct": values.gt(0).mean()*100 if len(values) else np.nan})
        for column in ("max_return_60d_pct", "max_drawdown_60d_pct"):
            values = pd.to_numeric(group.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
            row[f"average_{column}"] = values.mean()
            row[f"median_{column}"] = values.median()
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=summary_columns())


def curves(detail: pd.DataFrame, prices: dict[str, pd.DataFrame], horizon: int = 60) -> pd.DataFrame:
    unique = unique_entries(detail)
    records = []
    for _, row in unique.iterrows():
        ticker = str(row.ticker)
        if ticker not in prices or pd.isna(row.entry_date) or pd.isna(row.entry_price):
            continue
        future = prices[ticker].loc[prices[ticker].index >= pd.Timestamp(row.entry_date)].iloc[:horizon]
        curve = pd.to_numeric(future["Close"], errors="coerce") / float(row.entry_price) * 100
        curve.index = range(1, len(curve)+1)
        records.append({"condition": row.condition, "group": row.condition_group,
                        "ticker": ticker, "entry_date": row.entry_date, "curve": curve})
    def avg(items, day):
        values = [float(x["curve"].loc[day]) for x in items if day in x["curve"].index]
        return float(np.mean(values)) if values else np.nan
    output = []
    for day in range(1, horizon+1):
        row = {"day": day, "all_condition_events": avg(records, day)}
        by_entry = {}
        for item in records:
            by_entry.setdefault((item["ticker"], item["entry_date"]), item)
        row["all_unique_entries"] = avg(by_entry.values(), day)
        for condition in CONDITIONS:
            row[condition] = avg([item for item in records if item["condition"] == condition], day)
        for group in ("A_only", "A_B", "A_B_C", "C_only"):
            row[group] = avg([item for item in by_entry.values() if item["group"] == group], day)
        output.append(row)
    return pd.DataFrame(output, columns=legacy.CURVE_COLUMNS)


def markdown(summary: pd.DataFrame, detail: pd.DataFrame, max_trigger_days: int) -> str:
    unique = unique_entries(detail)
    lines = [
        "## Condition A-C confirmed first-occurrence backtest", "",
        f"- Trigger search: score8 day 0 through trading day {max_trigger_days}.",
        "- Before 15:30 JST, today's Japanese daily bar is excluded.",
        "- Entry: next trading-day open. Statistics: duplicate ticker/condition/entry-date trades removed.", "",
        "| Condition | Source | Triggered | Unique | Duplicates | Closed 60d | Avg 20d % | Avg 60d % | Median 60d % | Win 60d % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        values = [row.condition, int(row.source_signals), int(row.triggered_signals), int(row.unique_entries),
                  int(row.duplicate_events_removed), int(row.closed_60d), row.average_20d_return_pct,
                  row.average_60d_return_pct, row.median_60d_return_pct, row.win_rate_60d_pct]
        lines.append("| " + " | ".join("-" if pd.isna(value) else f"{value:.2f}" if isinstance(value, float) else str(value) for value in values) + " |")
    lines += ["", "### Unique entries", "", "| Trigger date | Day | Ticker | Name | Condition | Status | 20d % | 60d % |",
              "|---|---:|---|---|---|---|---:|---:|"]
    for _, row in unique.head(200).iterrows():
        date = pd.Timestamp(row.condition_first_date).strftime("%Y-%m-%d")
        values = [date, row.condition_trigger_day, row.ticker, row.get("name", ""), row.condition, row.status,
                  row.get("return_20d_pct"), row.get("return_60d_pct")]
        lines.append("| " + " | ".join("-" if pd.isna(value) else f"{value:.2f}" if isinstance(value, float) else str(value).replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-dir", default="fixed_holding_input")
    parser.add_argument("--signal-pattern", default="screen_*_buy.csv")
    parser.add_argument("--score-min", type=float, default=8)
    parser.add_argument("--signal-mode", choices=["all", "first_in_streak"], default="first_in_streak")
    parser.add_argument("--output-dir", default="signal_path_report")
    parser.add_argument("--path-detail", default="signal_path_report/signal_path_detail.csv")
    parser.add_argument("--lookback", type=int, default=120)
    parser.add_argument("--max-trigger-days", type=int, default=60)
    parser.add_argument("--github-summary", default=os.environ.get("GITHUB_STEP_SUMMARY", ""))
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    signals, files, raw = load_signals(args.signals_dir, args.signal_pattern, args.score_min)
    used = select_signals(signals, args.signal_mode)
    used = used[used.is_used].copy()
    LOGGER.info("CSV files=%d raw rows=%d used=%d", files, raw, len(used))
    path = Path(args.path_detail)
    path_detail = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": "string"}) if path.exists() else None
    prices, failures = {}, {}
    if not used.empty:
        start = pd.to_datetime(used.signal_date).min() - pd.Timedelta(days=args.lookback*2)
        end = pd.Timestamp.now().tz_localize(None).normalize() + pd.Timedelta(days=2)
        prices, failures = download_prices(sorted(used.ticker.astype(str).unique()), start, end)
    prices = {ticker: confirmed_prices(frame) for ticker, frame in prices.items()}
    original = legacy.find_first_condition_dates
    legacy.find_first_condition_dates = lambda flags, signal_date: limited_first_dates(flags, signal_date, args.max_trigger_days)
    try:
        detail, _, _, status = legacy.build_condition_backtest(used, prices, failures, path_detail, HOLDINGS)
    finally:
        legacy.find_first_condition_dates = original
    detail = add_trigger_days(detail, prices)
    result_summary = summarize(detail, status)
    result_curves = curves(detail, prices)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output/"condition_backtest_detail.csv", index=False, encoding="utf-8-sig")
    unique_entries(detail).to_csv(output/"condition_backtest_unique_entries.csv", index=False, encoding="utf-8-sig")
    result_summary.to_csv(output/"condition_backtest_summary.csv", index=False, encoding="utf-8-sig")
    result_curves.to_csv(output/"condition_average_price_curve.csv", index=False, encoding="utf-8-sig")
    status.to_csv(output/"condition_backtest_status.csv", index=False, encoding="utf-8-sig")
    if args.github_summary:
        with open(args.github_summary, "a", encoding="utf-8") as handle:
            handle.write(markdown(result_summary, detail, args.max_trigger_days))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
