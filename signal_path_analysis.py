#!/usr/bin/env python3
"""Analyze score8 price paths and inverse-head-and-shoulders-like stages."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fixed_holding_backtest import download_prices, load_signals, select_signals

LOGGER = logging.getLogger("signal_path_analysis")

PATH_COLUMNS = [
    "signal_date", "ticker", "name", "score", "entry_date", "entry_price",
    "first_peak_date", "first_peak_close", "first_peak_return_pct",
    "pullback_low_date", "pullback_low_close", "pullback_pct",
    "second_peak_date", "second_peak_close", "second_peak_return_pct",
    "second_peak_vs_first_pct", "breakout_date", "breakout_day",
    "return_50d_pct", "return_60d_pct", "path_label", "status",
]
FEATURE_COLUMNS = [
    "signal_date", "ticker", "name", "score", "sell_score", "rsi", "adx14",
    "plus_di14", "minus_di14", "di_spread", "ma20_distance_pct",
    "ma50_distance_pct", "ma20_slope_10d_pct", "ma50_slope_10d_pct",
    "ma20_above_ma50", "distance_to_20d_high_pct",
    "pullback_from_60d_high_pct", "volume_5_20_ratio", "pattern_stage",
    "neckline", "distance_to_neckline_pct", "neckline_slope_pct",
    "l1_date", "h1_date", "l2_date", "h2_date", "l3_date",
]
STATUS_COLUMNS = ["signal_date", "ticker", "status", "message"]


@dataclass(frozen=True)
class PathRules:
    horizon: int = 60
    first_peak_window: int = 15
    pullback_end: int = 35
    first_rise_min_pct: float = 3.0
    pullback_min_pct: float = 3.0
    pullback_max_pct: float = 15.0
    breakout_margin_pct: float = 1.0


def normalize_prices(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    return frame.sort_index()


def pct(new: float, old: float) -> float:
    if pd.isna(new) or pd.isna(old) or old == 0:
        return np.nan
    return (float(new) / float(old) - 1.0) * 100.0


def classify_price_path(signal: pd.Series, prices: pd.DataFrame, rules: PathRules) -> tuple[dict, pd.Series | None]:
    """Use future prices only for the outcome label, never for signal features."""
    prices = normalize_prices(prices)
    signal_date = pd.Timestamp(signal["signal_date"]).normalize()
    future = prices.loc[prices.index > signal_date]
    row = {column: pd.NA for column in PATH_COLUMNS}
    row.update(signal_date=signal_date, ticker=signal.get("ticker"), name=signal.get("name"),
               score=signal.get("score"), path_label="IN_PROGRESS", status="IN_PROGRESS")
    if future.empty:
        row.update(path_label="NO_NEXT_TRADING_DAY", status="NO_NEXT_TRADING_DAY")
        return row, None
    if not {"Open", "Close"}.issubset(future.columns):
        row.update(path_label="NO_PRICE_DATA", status="NO_PRICE_DATA")
        return row, None
    entry = pd.to_numeric(pd.Series([future.iloc[0]["Open"]]), errors="coerce").iloc[0]
    if pd.isna(entry) or entry <= 0:
        row.update(path_label="NO_PRICE_DATA", status="NO_PRICE_DATA")
        return row, None
    row.update(entry_date=future.index[0], entry_price=float(entry))
    observed = future.iloc[: rules.horizon]
    curve = pd.to_numeric(observed["Close"], errors="coerce") / float(entry) * 100.0
    curve.index = range(1, len(curve) + 1)
    if len(future) >= 50:
        row["return_50d_pct"] = pct(future.iloc[49]["Close"], entry)
    if len(future) >= 60:
        row["return_60d_pct"] = pct(future.iloc[59]["Close"], entry)
    if len(future) < rules.horizon:
        return row, curve

    first = pd.to_numeric(future.iloc[: rules.first_peak_window]["Close"], errors="coerce").dropna()
    if first.empty:
        row.update(path_label="NO_PRICE_DATA", status="NO_PRICE_DATA")
        return row, curve
    first_date = first.idxmax()
    first_close = float(first.loc[first_date])
    first_pos = int(future.index.get_loc(first_date))
    pull = pd.to_numeric(future.iloc[first_pos + 1 : rules.pullback_end]["Close"], errors="coerce").dropna()
    if pull.empty:
        row.update(path_label="CONTINUOUS_RISE", status="CLOSED")
        return row, curve
    pull_date = pull.idxmin()
    pull_close = float(pull.loc[pull_date])
    pull_pos = int(future.index.get_loc(pull_date))
    second = pd.to_numeric(future.iloc[pull_pos + 1 : rules.horizon]["Close"], errors="coerce").dropna()
    if second.empty:
        row.update(path_label="SECOND_RISE_FAILED", status="CLOSED")
        return row, curve
    second_date = second.idxmax()
    second_close = float(second.loc[second_date])
    first_return = pct(first_close, entry)
    pullback = pct(pull_close, first_close)
    breakout_level = first_close * (1.0 + rules.breakout_margin_pct / 100.0)
    after_pull = pd.to_numeric(future.iloc[pull_pos + 1 : rules.horizon]["Close"], errors="coerce")
    hits = after_pull[after_pull >= breakout_level]
    breakout_date = hits.index[0] if not hits.empty else pd.NaT
    breakout_day = int(future.index.get_loc(breakout_date)) + 1 if pd.notna(breakout_date) else pd.NA

    if first_return < rules.first_rise_min_pct:
        label = "NO_FIRST_RISE"
    elif pullback > -rules.pullback_min_pct:
        label = "CONTINUOUS_RISE"
    elif pullback < -rules.pullback_max_pct:
        label = "FIRST_RISE_ONLY"
    elif second_close >= breakout_level:
        label = "DOUBLE_RISE_SUCCESS"
    else:
        label = "SECOND_RISE_FAILED"
    row.update(
        first_peak_date=first_date, first_peak_close=first_close,
        first_peak_return_pct=first_return, pullback_low_date=pull_date,
        pullback_low_close=pull_close, pullback_pct=pullback,
        second_peak_date=second_date, second_peak_close=second_close,
        second_peak_return_pct=pct(second_close, entry),
        second_peak_vs_first_pct=pct(second_close, first_close),
        breakout_date=breakout_date, breakout_day=breakout_day,
        path_label=label, status="CLOSED",
    )
    return row, curve


def confirmed_pivots(close: pd.Series, span: int = 3) -> list[tuple[pd.Timestamp, str, float]]:
    close = pd.to_numeric(close, errors="coerce").dropna()
    result: list[tuple[pd.Timestamp, str, float]] = []
    for pos in range(span, len(close) - span):
        value = float(close.iloc[pos])
        window = close.iloc[pos - span : pos + span + 1]
        kind = "H" if value == window.max() and (window == value).sum() == 1 else None
        if value == window.min() and (window == value).sum() == 1:
            kind = "L"
        if kind is None:
            continue
        point = (pd.Timestamp(close.index[pos]), kind, value)
        if result and result[-1][1] == kind:
            better = value > result[-1][2] if kind == "H" else value < result[-1][2]
            if better:
                result[-1] = point
        else:
            result.append(point)
    return result


def detect_pattern_stage(history: pd.DataFrame, pivot_span: int = 3) -> dict:
    result = {"pattern_stage": "NONE", "neckline": np.nan,
              "distance_to_neckline_pct": np.nan, "neckline_slope_pct": np.nan,
              "l1_date": pd.NaT, "h1_date": pd.NaT, "l2_date": pd.NaT,
              "h2_date": pd.NaT, "l3_date": pd.NaT}
    if history.empty or "Close" not in history:
        return result
    close = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if len(close) < 20:
        return result
    pivots = confirmed_pivots(close, pivot_span)
    current = float(close.iloc[-1])
    for size, kinds in ((5, "LHLHL"), (4, "LHLH"), (3, "LHL")):
        if len(pivots) < size:
            continue
        seq = pivots[-size:]
        if "".join(item[1] for item in seq) != kinds:
            continue
        if size == 3:
            l1, h1, l2 = seq
            if l2[2] < l1[2]:
                result.update(pattern_stage="HEAD_TO_NECKLINE", l1_date=l1[0], h1_date=h1[0],
                              l2_date=l2[0], neckline=h1[2], distance_to_neckline_pct=pct(current, h1[2]))
                return result
        elif size == 4:
            l1, h1, l2, h2 = seq
            if l2[2] < l1[2] and abs(pct(h2[2], h1[2])) <= 5.0:
                neckline = (h1[2] + h2[2]) / 2.0
                result.update(pattern_stage="RIGHT_SHOULDER", l1_date=l1[0], h1_date=h1[0],
                              l2_date=l2[0], h2_date=h2[0], neckline=neckline,
                              distance_to_neckline_pct=pct(current, neckline),
                              neckline_slope_pct=pct(h2[2], h1[2]))
                return result
        else:
            l1, h1, l2, h2, l3 = seq
            valid = l2[2] < l1[2] and pct(l3[2], l2[2]) >= 3.0 and abs(pct(h2[2], h1[2])) <= 5.0
            if valid:
                neckline = (h1[2] + h2[2]) / 2.0
                stage = "POST_BREAKOUT" if current >= neckline else "BREAKOUT_APPROACH"
                result.update(pattern_stage=stage, l1_date=l1[0], h1_date=h1[0], l2_date=l2[0],
                              h2_date=h2[0], l3_date=l3[0], neckline=neckline,
                              distance_to_neckline_pct=pct(current, neckline),
                              neckline_slope_pct=pct(h2[2], h1[2]))
                return result
    return result


def calculate_signal_features(signal: pd.Series, prices: pd.DataFrame, lookback: int = 120) -> dict:
    prices = normalize_prices(prices)
    signal_date = pd.Timestamp(signal["signal_date"]).normalize()
    history = prices.loc[prices.index <= signal_date].tail(lookback)
    close = pd.to_numeric(history.get("Close", pd.Series(dtype=float)), errors="coerce")
    volume = pd.to_numeric(history.get("Volume", pd.Series(dtype=float)), errors="coerce")
    current = close.iloc[-1] if not close.empty else np.nan
    ma20, ma50 = close.rolling(20).mean(), close.rolling(50).mean()

    def slope(series: pd.Series, periods: int = 10) -> float:
        valid = series.dropna()
        return pct(valid.iloc[-1], valid.iloc[-periods - 1]) if len(valid) > periods else np.nan

    plus = pd.to_numeric(pd.Series([signal.get("plus_di14")]), errors="coerce").iloc[0]
    minus = pd.to_numeric(pd.Series([signal.get("minus_di14")]), errors="coerce").iloc[0]
    vol20 = volume.tail(20).mean()
    row = {
        "signal_date": signal_date, "ticker": signal.get("ticker"), "name": signal.get("name"),
        "score": signal.get("score"), "sell_score": signal.get("sell_score"),
        "rsi": signal.get("rsi"), "adx14": signal.get("adx14"),
        "plus_di14": signal.get("plus_di14"), "minus_di14": signal.get("minus_di14"),
        "di_spread": plus - minus, "ma20_distance_pct": pct(current, ma20.iloc[-1]) if len(ma20) else np.nan,
        "ma50_distance_pct": pct(current, ma50.iloc[-1]) if len(ma50) else np.nan,
        "ma20_slope_10d_pct": slope(ma20), "ma50_slope_10d_pct": slope(ma50),
        "ma20_above_ma50": bool(ma20.iloc[-1] > ma50.iloc[-1]) if ma20.notna().any() and ma50.notna().any() else pd.NA,
        "distance_to_20d_high_pct": pct(current, close.tail(20).max()),
        "pullback_from_60d_high_pct": pct(current, close.tail(60).max()),
        "volume_5_20_ratio": volume.tail(5).mean() / vol20 if pd.notna(vol20) and vol20 != 0 else np.nan,
        **detect_pattern_stage(history),
    }
    return {column: row.get(column, pd.NA) for column in FEATURE_COLUMNS}


def summarize_patterns(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in paths.groupby("path_label", dropna=False):
        closed = group[group["status"].eq("CLOSED")]
        rows.append({
            "path_label": label, "signals": len(group), "closed_signals": len(closed),
            "average_50d_return_pct": pd.to_numeric(group["return_50d_pct"], errors="coerce").mean(),
            "median_50d_return_pct": pd.to_numeric(group["return_50d_pct"], errors="coerce").median(),
            "average_60d_return_pct": pd.to_numeric(group["return_60d_pct"], errors="coerce").mean(),
            "median_60d_return_pct": pd.to_numeric(group["return_60d_pct"], errors="coerce").median(),
            "average_pullback_pct": pd.to_numeric(group["pullback_pct"], errors="coerce").mean(),
            "average_breakout_day": pd.to_numeric(group["breakout_day"], errors="coerce").mean(),
        })
    return pd.DataFrame(rows)


def build_average_curves(curves: list[tuple[str, pd.Series]], horizon: int) -> pd.DataFrame:
    labels = sorted({label for label, _ in curves})
    rows = []
    for day in range(1, horizon + 1):
        row: dict[str, float | int] = {"day": day}
        values = [curve.loc[day] for _, curve in curves if day in curve.index]
        row["all"] = float(np.mean(values)) if values else np.nan
        for label in labels:
            values = [curve.loc[day] for item_label, curve in curves if item_label == label and day in curve.index]
            row[label] = float(np.mean(values)) if values else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_outputs(output: Path, paths: pd.DataFrame, features: pd.DataFrame,
                  summary: pd.DataFrame, curves: pd.DataFrame, status: pd.DataFrame) -> None:
    output.mkdir(parents=True, exist_ok=True)
    paths.to_csv(output / "signal_path_detail.csv", index=False, encoding="utf-8-sig")
    features.to_csv(output / "signal_features.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "pattern_feature_summary.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(output / "average_price_curve.csv", index=False, encoding="utf-8-sig")
    status.to_csv(output / "signal_path_status.csv", index=False, encoding="utf-8-sig")


def run(args: argparse.Namespace) -> None:
    signals, file_count, raw_count = load_signals(args.signals_dir, args.signal_pattern, args.score_min)
    marked = select_signals(signals, args.signal_mode)
    used = marked[marked["is_used"]].copy()
    LOGGER.info("CSV files=%d raw rows=%d used signals=%d", file_count, raw_count, len(used))
    rules = PathRules(args.horizon, args.first_peak_window, args.pullback_end,
                      args.first_rise_min_pct, args.pullback_min_pct,
                      args.pullback_max_pct, args.breakout_margin_pct)
    if used.empty:
        write_outputs(Path(args.output_dir), pd.DataFrame(columns=PATH_COLUMNS),
                      pd.DataFrame(columns=FEATURE_COLUMNS), pd.DataFrame(), pd.DataFrame(),
                      pd.DataFrame(columns=STATUS_COLUMNS))
        return
    tickers = sorted(used["ticker"].unique())
    start = used["signal_date"].min() - pd.Timedelta(days=args.lookback * 2)
    end = pd.Timestamp.now().tz_localize(None).normalize() + pd.Timedelta(days=2)
    prices, failures = download_prices(tickers, start, end)
    path_rows, feature_rows, status_rows, curves = [], [], [], []
    for _, signal in used.iterrows():
        ticker = signal["ticker"]
        if ticker not in prices:
            status_rows.append({"signal_date": signal["signal_date"], "ticker": ticker,
                                "status": "PRICE_DOWNLOAD_FAILED",
                                "message": failures.get(ticker, "no price data")})
            continue
        path_row, curve = classify_price_path(signal, prices[ticker], rules)
        path_rows.append(path_row)
        feature_rows.append(calculate_signal_features(signal, prices[ticker], args.lookback))
        if curve is not None:
            curves.append((str(path_row["path_label"]), curve))
        if path_row["status"] != "CLOSED":
            status_rows.append({"signal_date": signal["signal_date"], "ticker": ticker,
                                "status": path_row["status"],
                                "message": "60 trading days are not yet fully observable"})
    path_df = pd.DataFrame(path_rows).reindex(columns=PATH_COLUMNS)
    feature_df = pd.DataFrame(feature_rows).reindex(columns=FEATURE_COLUMNS)
    write_outputs(Path(args.output_dir), path_df, feature_df,
                  summarize_patterns(path_df) if not path_df.empty else pd.DataFrame(),
                  build_average_curves(curves, args.horizon),
                  pd.DataFrame(status_rows, columns=STATUS_COLUMNS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals-dir", default="fixed_holding_input")
    parser.add_argument("--signal-pattern", default="screen_*_buy.csv")
    parser.add_argument("--score-min", type=float, default=8)
    parser.add_argument("--signal-mode", choices=["all", "first_in_streak"], default="first_in_streak")
    parser.add_argument("--output-dir", default="signal_path_report")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--first-peak-window", type=int, default=15)
    parser.add_argument("--pullback-end", type=int, default=35)
    parser.add_argument("--first-rise-min-pct", type=float, default=3.0)
    parser.add_argument("--pullback-min-pct", type=float, default=3.0)
    parser.add_argument("--pullback-max-pct", type=float, default=15.0)
    parser.add_argument("--breakout-margin-pct", type=float, default=1.0)
    parser.add_argument("--lookback", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.first_peak_window < args.pullback_end < args.horizon:
        parser.error("require first_peak_window < pullback_end < horizon")
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
