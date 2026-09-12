#!/usr/bin/env python3
"""Backtest the first post-score8 occurrence of conditions A-C."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("condition_backtest")
DEFAULT_HOLDINGS = (5, 10, 20, 40, 50, 60)
CONDITIONS = ("A", "B", "C")
OHLC = {"Open", "High", "Low", "Close"}
BASE_COLUMNS = [
    "signal_date", "ticker", "name", "score", "sell_score", "condition",
    "condition_first_date", "condition_labels_on_first_date", "condition_group",
    "pullback_from_60d_high_pct", "ma20_distance_pct", "plus_di14",
    "minus_di14", "di_spread", "entry_date", "entry_price",
]
TAIL_COLUMNS = [
    "exit_60d_date", "exit_60d_price", "max_return_60d_pct",
    "max_drawdown_60d_pct", "observed_trading_days", "source_path_label",
    "source_path_status", "status",
]
STATUS_COLUMNS = [
    "signal_date", "ticker", "name", "condition", "condition_first_date",
    "entry_date", "observed_trading_days", "status", "message",
]
CURVE_COLUMNS = [
    "day", "all_condition_events", "all_unique_entries", "A", "B", "C",
    "A_only", "A_B", "A_B_C", "C_only",
]


def detail_columns(holdings: Iterable[int] = DEFAULT_HOLDINGS) -> list[str]:
    return BASE_COLUMNS + [f"return_{int(h)}d_pct" for h in holdings] + TAIL_COLUMNS


def summary_columns(holdings: Iterable[int] = DEFAULT_HOLDINGS) -> list[str]:
    columns = [
        "condition", "source_signals", "triggered_signals", "not_triggered_signals",
        "price_download_failed", "closed_60d", "open_60d",
    ]
    for h in holdings:
        columns += [
            f"observed_{int(h)}d", f"average_{int(h)}d_return_pct",
            f"median_{int(h)}d_return_pct", f"win_rate_{int(h)}d_pct",
        ]
    return columns + [
        "average_max_return_60d_pct", "median_max_return_60d_pct",
        "average_max_drawdown_60d_pct", "median_max_drawdown_60d_pct",
    ]


def normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    return result.sort_index()


def _rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def calculate_daily_condition_metrics(prices: pd.DataFrame, di_period: int = 14) -> pd.DataFrame:
    prices = normalize_prices(prices)
    columns = [
        "pullback_from_60d_high_pct", "ma20_distance_pct", "plus_di14",
        "minus_di14", "di_spread",
    ]
    if not {"High", "Low", "Close"}.issubset(prices.columns):
        return pd.DataFrame(index=prices.index, columns=columns, dtype=float)
    high = pd.to_numeric(prices["High"], errors="coerce")
    low = pd.to_numeric(prices["Low"], errors="coerce")
    close = pd.to_numeric(prices["Close"], errors="coerce")
    ma20 = close.rolling(20, min_periods=20).mean()
    high60 = close.rolling(60, min_periods=60).max()
    up_move, down_move = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=prices.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=prices.index)
    previous = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-previous).abs(), (low-previous).abs()], axis=1).max(axis=1)
    atr = _rma(tr, di_period)
    plus_di = 100.0 * _rma(plus_dm, di_period) / atr.replace(0, np.nan)
    minus_di = 100.0 * _rma(minus_dm, di_period) / atr.replace(0, np.nan)
    return pd.DataFrame({
        "pullback_from_60d_high_pct": (close / high60 - 1.0) * 100.0,
        "ma20_distance_pct": (close / ma20 - 1.0) * 100.0,
        "plus_di14": plus_di, "minus_di14": minus_di,
        "di_spread": plus_di - minus_di,
    }, index=prices.index)


def build_condition_flags(metrics: pd.DataFrame) -> pd.DataFrame:
    pullback = pd.to_numeric(metrics.get("pullback_from_60d_high_pct"), errors="coerce")
    ma20 = pd.to_numeric(metrics.get("ma20_distance_pct"), errors="coerce")
    spread = pd.to_numeric(metrics.get("di_spread"), errors="coerce")
    result = pd.DataFrame(index=metrics.index)
    result["A"] = pullback.le(-12.0).fillna(False)
    result["B"] = result["A"] & ma20.ge(1.0).fillna(False)
    result["C"] = ma20.ge(1.0).fillna(False) & spread.ge(3.0).fillna(False)
    return result.astype(bool)


def find_first_condition_dates(flags: pd.DataFrame, signal_date: object) -> dict[str, pd.Timestamp]:
    date = pd.Timestamp(signal_date)
    if date.tzinfo is not None:
        date = date.tz_localize(None)
    candidate = flags.loc[flags.index >= date.normalize()]
    result = {}
    for condition in CONDITIONS:
        hits = candidate.index[candidate[condition]]
        if len(hits):
            result[condition] = pd.Timestamp(hits[0]).normalize()
    return result


def _labels(flags: pd.DataFrame, date: pd.Timestamp) -> str:
    return "|".join(c for c in CONDITIONS if date in flags.index and bool(flags.at[date, c]))


def _group(labels: str) -> str:
    return {"A": "A_only", "C": "C_only"}.get(labels, labels.replace("|", "_"))


def _pct(value: object, entry: float) -> float:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return (float(value) / entry - 1.0) * 100.0 if pd.notna(value) and entry > 0 else np.nan


def evaluate_condition_event(
    signal: pd.Series,
    prices: pd.DataFrame,
    metrics: pd.DataFrame,
    flags: pd.DataFrame,
    condition: str,
    condition_date: pd.Timestamp,
    holdings: Iterable[int] = DEFAULT_HOLDINGS,
    source_path: dict[str, object] | None = None,
) -> tuple[dict, pd.Series | None]:
    holdings = tuple(sorted({int(h) for h in holdings}))
    horizon = max(holdings)
    prices = normalize_prices(prices)
    future = prices.loc[prices.index > condition_date]
    metric = metrics.loc[condition_date] if condition_date in metrics.index else pd.Series(dtype=float)
    source_path = source_path or {}
    labels = _labels(flags, condition_date)
    row = {
        "signal_date": pd.Timestamp(signal.get("signal_date")).normalize(),
        "ticker": signal.get("ticker"), "name": signal.get("name"),
        "score": signal.get("score"), "sell_score": signal.get("sell_score"),
        "condition": condition, "condition_first_date": condition_date,
        "condition_labels_on_first_date": labels, "condition_group": _group(labels),
        **{c: metric.get(c, np.nan) for c in [
            "pullback_from_60d_high_pct", "ma20_distance_pct", "plus_di14",
            "minus_di14", "di_spread",
        ]},
        "entry_date": pd.NaT, "entry_price": np.nan, "exit_60d_date": pd.NaT,
        "exit_60d_price": np.nan, "max_return_60d_pct": np.nan,
        "max_drawdown_60d_pct": np.nan, "observed_trading_days": 0,
        "source_path_label": source_path.get("path_label", pd.NA),
        "source_path_status": source_path.get("status", pd.NA),
        "status": "NO_NEXT_TRADING_DAY",
        **{f"return_{h}d_pct": np.nan for h in holdings},
    }
    if future.empty:
        return row, None
    if not OHLC.issubset(prices.columns):
        row["status"] = "NO_PRICE_DATA"
        return row, None
    entry = pd.to_numeric(pd.Series([future.iloc[0]["Open"]]), errors="coerce").iloc[0]
    if pd.isna(entry) or entry <= 0:
        row["status"] = "NO_PRICE_DATA"
        return row, None
    entry = float(entry)
    observed = future.iloc[:horizon]
    row.update(entry_date=future.index[0], entry_price=entry, observed_trading_days=len(observed))
    curve = pd.to_numeric(observed["Close"], errors="coerce") / entry * 100.0
    curve.index = range(1, len(curve) + 1)
    for h in holdings:
        if len(future) >= h:
            row[f"return_{h}d_pct"] = _pct(future.iloc[h-1]["Close"], entry)
    if len(future) < horizon:
        row["status"] = "OPEN_INSUFFICIENT_DAYS"
        return row, curve
    window = future.iloc[:horizon]
    highs = pd.to_numeric(window["High"], errors="coerce").dropna()
    lows = pd.to_numeric(window["Low"], errors="coerce").dropna()
    exit_price = pd.to_numeric(pd.Series([window.iloc[-1]["Close"]]), errors="coerce").iloc[0]
    if pd.isna(exit_price) or highs.empty or lows.empty:
        row["status"] = "NO_PRICE_DATA"
        return row, curve
    row.update(
        exit_60d_date=window.index[-1], exit_60d_price=float(exit_price),
        max_return_60d_pct=(highs.max()/entry-1.0)*100.0,
        max_drawdown_60d_pct=(lows.min()/entry-1.0)*100.0, status="CLOSED",
    )
    return row, curve


def _path_lookup(path_detail: pd.DataFrame | None) -> dict:
    if path_detail is None or path_detail.empty:
        return {}
    frame = path_detail.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype("string")
    return {
        (row.signal_date, str(row.ticker)): {"path_label": row.get("path_label"), "status": row.get("status")}
        for _, row in frame.iterrows() if pd.notna(row.signal_date) and pd.notna(row.ticker)
    }


def summarize_condition_backtest(detail: pd.DataFrame, status: pd.DataFrame,
                                 holdings: Iterable[int] = DEFAULT_HOLDINGS) -> pd.DataFrame:
    holdings = tuple(sorted({int(h) for h in holdings}))
    rows = []
    for condition in CONDITIONS:
        s = status[status.condition.eq(condition)] if not status.empty else status
        group = detail[detail.condition.eq(condition)] if not detail.empty else detail
        row = {
            "condition": condition, "source_signals": len(s), "triggered_signals": len(group),
            "not_triggered_signals": int(s.status.eq("NOT_TRIGGERED").sum()) if not s.empty else 0,
            "price_download_failed": int(s.status.eq("PRICE_DOWNLOAD_FAILED").sum()) if not s.empty else 0,
            "closed_60d": int(group.status.eq("CLOSED").sum()) if not group.empty else 0,
            "open_60d": int(group.status.eq("OPEN_INSUFFICIENT_DAYS").sum()) if not group.empty else 0,
        }
        for h in holdings:
            values = pd.to_numeric(group.get(f"return_{h}d_pct", pd.Series(dtype=float)), errors="coerce").dropna()
            row.update({
                f"observed_{h}d": len(values), f"average_{h}d_return_pct": values.mean(),
                f"median_{h}d_return_pct": values.median(),
                f"win_rate_{h}d_pct": values.gt(0).mean()*100.0 if len(values) else np.nan,
            })
        for column in ("max_return_60d_pct", "max_drawdown_60d_pct"):
            values = pd.to_numeric(group.get(column, pd.Series(dtype=float)), errors="coerce").dropna()
            row[f"average_{column}"] = values.mean()
            row[f"median_{column}"] = values.median()
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=summary_columns(holdings))


def build_average_condition_curves(records: list[dict], horizon: int = 60) -> pd.DataFrame:
    unique = {}
    for record in records:
        unique.setdefault(tuple(record["event_key"]), record)
    def avg(items: Iterable[dict], day: int) -> float:
        values = [float(r["curve"].loc[day]) for r in items if day in r["curve"].index]
        return float(np.mean(values)) if values else np.nan
    rows = []
    for day in range(1, horizon+1):
        row = {c: np.nan for c in CURVE_COLUMNS}
        row.update(day=day, all_condition_events=avg(records, day), all_unique_entries=avg(unique.values(), day))
        for condition in CONDITIONS:
            row[condition] = avg((r for r in records if r["condition"] == condition), day)
        for group in ("A_only", "A_B", "A_B_C", "C_only"):
            row[group] = avg((r for r in unique.values() if r["condition_group"] == group), day)
        rows.append(row)
    return pd.DataFrame(rows, columns=CURVE_COLUMNS)


def build_condition_backtest(
    signals: pd.DataFrame, prices: dict[str, pd.DataFrame], failures: dict[str, str] | None = None,
    path_detail: pd.DataFrame | None = None, holdings: Iterable[int] = DEFAULT_HOLDINGS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    holdings = tuple(sorted({int(h) for h in holdings}))
    failures, paths = failures or {}, _path_lookup(path_detail)
    details, statuses, records, metric_cache, flag_cache = [], [], [], {}, {}
    for _, signal in signals.iterrows():
        ticker, signal_date = str(signal.get("ticker")), pd.Timestamp(signal.get("signal_date")).normalize()
        base = {"signal_date": signal_date, "ticker": ticker, "name": signal.get("name")}
        if ticker not in prices:
            for condition in CONDITIONS:
                statuses.append({**base, "condition": condition, "condition_first_date": pd.NaT,
                    "entry_date": pd.NaT, "observed_trading_days": 0,
                    "status": "PRICE_DOWNLOAD_FAILED", "message": failures.get(ticker, "no price data")})
            continue
        if ticker not in metric_cache:
            metric_cache[ticker] = calculate_daily_condition_metrics(prices[ticker])
            flag_cache[ticker] = build_condition_flags(metric_cache[ticker])
        metrics, flags = metric_cache[ticker], flag_cache[ticker]
        first_dates = find_first_condition_dates(flags, signal_date)
        for condition in CONDITIONS:
            if condition not in first_dates:
                statuses.append({**base, "condition": condition, "condition_first_date": pd.NaT,
                    "entry_date": pd.NaT, "observed_trading_days": 0, "status": "NOT_TRIGGERED",
                    "message": "condition has not occurred on or after the score8 signal date"})
                continue
            date = first_dates[condition]
            row, curve = evaluate_condition_event(signal, prices[ticker], metrics, flags, condition, date,
                                                  holdings, paths.get((signal_date, ticker), {}))
            details.append(row)
            statuses.append({**base, "condition": condition, "condition_first_date": date,
                "entry_date": row["entry_date"], "observed_trading_days": row["observed_trading_days"],
                "status": row["status"], "message": "60 trading days observed" if row["status"] == "CLOSED"
                else "60 trading days are not yet fully observable"})
            if curve is not None:
                records.append({"condition": condition, "condition_group": row["condition_group"],
                    "event_key": (signal_date, ticker, date), "curve": curve})
    detail = pd.DataFrame(details).reindex(columns=detail_columns(holdings))
    status = pd.DataFrame(statuses, columns=STATUS_COLUMNS)
    if not detail.empty:
        detail = detail.sort_values(["condition_first_date", "ticker", "condition"], kind="stable").reset_index(drop=True)
    if not status.empty:
        status = status.sort_values(["signal_date", "ticker", "condition"], kind="stable").reset_index(drop=True)
    return detail, summarize_condition_backtest(detail, status, holdings), \
        build_average_condition_curves(records, max(holdings)), status


def write_condition_outputs(output_dir: Path | str, detail: pd.DataFrame, summary: pd.DataFrame,
                            curves: pd.DataFrame, status: pd.DataFrame) -> None:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    for filename, frame in (
        ("condition_backtest_detail.csv", detail), ("condition_backtest_summary.csv", summary),
        ("condition_average_price_curve.csv", curves), ("condition_backtest_status.csv", status),
    ):
        frame.to_csv(output/filename, index=False, encoding="utf-8-sig")


def _display(value: object, digits: int = 2) -> str:
    if pd.isna(value): return "-"
    if isinstance(value, (int, float, np.integer, np.floating)): return f"{float(value):.{digits}f}"
    return str(value).replace("|", "\\|")


def build_markdown_summary(summary: pd.DataFrame, detail: pd.DataFrame) -> str:
    lines = [
        "## Condition A-C first-occurrence backtest", "",
        "- Conditions are evaluated after every close from the score8 signal date onward.",
        "- Entry is the next trading day's open after the first occurrence.",
        "- The 60-day exit is the 60th trading day's close; MFE/MAE use High/Low.", "",
        "| Condition | Source | Triggered | Closed 60d | Open 60d | Avg 20d % | Win 20d % | Avg 60d % | Median 60d % | Win 60d % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        keys = ["condition", "source_signals", "triggered_signals", "closed_60d", "open_60d",
                "average_20d_return_pct", "win_rate_20d_pct", "average_60d_return_pct",
                "median_60d_return_pct", "win_rate_60d_pct"]
        lines.append("| " + " | ".join(_display(row.get(k), 0 if k in keys[1:5] else 2) for k in keys) + " |")
    lines += ["", "### First condition occurrences", "",
        "| Condition date | Ticker | Name | Condition | Conditions on date | Status | 20d % | 60d % |",
        "|---|---|---|---|---|---|---:|---:|"]
    for _, row in detail.head(200).iterrows():
        date = pd.to_datetime(row.get("condition_first_date"), errors="coerce")
        values = [date.strftime("%Y-%m-%d") if pd.notna(date) else "-", row.get("ticker"), row.get("name"),
                  row.get("condition"), row.get("condition_labels_on_first_date"), row.get("status"),
                  row.get("return_20d_pct"), row.get("return_60d_pct")]
        lines.append("| " + " | ".join(_display(v) for v in values) + " |")
    if detail.empty: lines.append("| - | - | - | - | - | No triggered conditions | - | - |")
    if len(detail) > 200: lines += ["", f"Showing the first 200 of {len(detail)} triggered events."]
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
    parser.add_argument("--holdings", type=int, nargs="+", default=list(DEFAULT_HOLDINGS))
    parser.add_argument("--github-summary", default=os.environ.get("GITHUB_STEP_SUMMARY", ""))
    args = parser.parse_args()
    if not args.holdings or any(h < 1 for h in args.holdings): parser.error("--holdings must be positive")
    if max(args.holdings) != 60: parser.error("the condition backtest must include 60 trading days")
    return args


def run(args: argparse.Namespace) -> None:
    from fixed_holding_backtest import download_prices, load_signals, select_signals
    signals, file_count, raw_count = load_signals(args.signals_dir, args.signal_pattern, args.score_min)
    marked = select_signals(signals, args.signal_mode)
    used = marked[marked["is_used"]].copy()
    LOGGER.info("CSV files=%d raw rows=%d used signals=%d", file_count, raw_count, len(used))
    path = Path(args.path_detail)
    path_detail = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": "string"}) if path.exists() else None
    prices, failures = {}, {}
    if not used.empty:
        tickers = sorted(used["ticker"].astype(str).unique())
        start = pd.to_datetime(used["signal_date"]).min() - pd.Timedelta(days=args.lookback*2)
        end = pd.Timestamp.now().tz_localize(None).normalize() + pd.Timedelta(days=2)
        prices, failures = download_prices(tickers, start, end)
        LOGGER.info("price downloads successful=%d failed=%d", len(prices), len(failures))
    detail, summary, curves, status = build_condition_backtest(used, prices, failures, path_detail, args.holdings)
    write_condition_outputs(args.output_dir, detail, summary, curves, status)
    if args.github_summary:
        with open(args.github_summary, "a", encoding="utf-8") as handle:
            handle.write(build_markdown_summary(summary, detail))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
