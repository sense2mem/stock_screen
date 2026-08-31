# -*- coding: utf-8 -*-
"""Safe entry point for the daily stock screen.

The screening calculations remain in screen_yearend.py. This wrapper separates
workflow execution time from the actual market session used for signal labels.
"""

from collections import Counter
from pathlib import Path

import pandas as pd

import screen_yearend as screen


MIN_PRICE_DATE_COVERAGE = 0.50


def resolve_signal_date(price_dates, total_tickers):
    """Resolve the market session by majority vote across screened bar dates.

    There is deliberately no runtime-date fallback. A signal file must never be
    labeled with a weekend/holiday merely because a scheduled workflow was late.
    """
    normalized = []
    for value in price_dates:
        if value is None or pd.isna(value):
            continue
        normalized.append(pd.Timestamp(value).normalize())

    if not normalized:
        raise RuntimeError(
            "Unable to resolve signal date: no screened ticker price dates were obtained"
        )

    counts = Counter(normalized)
    signal_date, votes = counts.most_common(1)[0]
    required = max(1, int(total_tickers * MIN_PRICE_DATE_COVERAGE))

    if votes < required:
        summary = ", ".join(
            f"{day.strftime('%Y-%m-%d')}={count}"
            for day, count in counts.most_common(5)
        )
        raise RuntimeError(
            "Unable to resolve signal date with sufficient coverage: "
            f"winner={signal_date.strftime('%Y-%m-%d')} votes={votes} "
            f"required={required} total_tickers={total_tickers} top_dates=[{summary}]"
        )

    return signal_date


def score_and_observe_price_date(score_func, price_dates, df_daily):
    """Run the unchanged scorer and record the exact final bar it consumed.

    The date is appended only after the scorer returns successfully, so the
    signal-date vote reflects the same ticker observations that produced rows in
    screen_tech().
    """
    result = score_func(df_daily)

    if df_daily is not None and not df_daily.empty:
        last_date = pd.Timestamp(df_daily.index[-1]).normalize()
        price_dates.append(last_date)

    return result


def run_screen_and_capture_price_dates(tickers):
    """Run screen_tech while observing the exact bars used for score calculation."""
    price_dates = []
    original_score_func = screen.score_buy_signals

    def observed_score_func(df_daily):
        return score_and_observe_price_date(
            original_score_func,
            price_dates,
            df_daily,
        )

    screen.score_buy_signals = observed_score_func
    try:
        tech_df, runtime_end = screen.screen_tech(
            tickers,
            chunk_size=20,
            pause_sec=0.2,
            max_retry=2,
        )
    finally:
        # Never leave the imported screening module monkey-patched after this run.
        screen.score_buy_signals = original_score_func

    return tech_df, runtime_end, price_dates


def rename_error_csv(runtime_end, signal_date):
    """Keep the optional errors CSV aligned with the resolved signal date."""
    runtime_str = pd.Timestamp(runtime_end).strftime("%Y-%m-%d")
    signal_str = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    if runtime_str == signal_str:
        return

    src = Path(f"screen_{runtime_str}_errors.csv")
    dst = Path(f"screen_{signal_str}_errors.csv")
    if src.exists():
        src.replace(dst)


def main():
    tickers = screen.load_tickers_from_excel_bcol("data_e.xls", sheet_name="Sheet1")

    # Capture the final bar date from the exact DataFrame passed into the
    # unchanged BUY/SELL scorer. No second Yahoo Finance request is used.
    tech_df, runtime_end, price_dates = run_screen_and_capture_price_dates(tickers)

    runtime_str = pd.Timestamp(runtime_end).strftime("%Y-%m-%d")
    signal_date = resolve_signal_date(price_dates, total_tickers=len(tickers))
    as_of_str = signal_date.strftime("%Y-%m-%d")

    print(
        "signal-date resolution:",
        f"runtime_date={runtime_str}",
        f"actual_market_date={as_of_str}",
        f"screened_price_dates={len(price_dates)}/{len(tickers)}",
        flush=True,
    )

    if signal_date > pd.Timestamp(runtime_end).normalize():
        raise RuntimeError(
            f"Resolved signal date {as_of_str} is later than runtime date {runtime_str}"
        )

    rename_error_csv(runtime_end, signal_date)

    val_df = screen.fetch_per_pbr_batch(
        tech_df["ticker"].dropna().astype(str).tolist(),
        batch_size=200,
        pause_sec=0.2,
    )

    tech_t = set(tech_df["ticker"].astype(str).str.upper())
    val_t = set(val_df["ticker"].astype(str).str.upper())
    print("intersection(tech,val):", len(tech_t & val_t), "/", len(tech_t))
    print("val non-null pe:", val_df["pe_ttm"].notna().sum(), "/", len(val_df))

    try:
        all_rows, buy_rows, sell_rows = screen.apply_filters_and_make_trades(
            tech_df,
            val_df,
            runtime_end,
        )
    except Exception as exc:
        print("ERROR in apply_filters_and_make_trades:", repr(exc))
        print("tech_df columns:", tech_df.columns.tolist())
        print("val_df columns:", val_df.columns.tolist())
        raise

    all_rows.to_csv(f"screen_{as_of_str}_all.csv", index=False, encoding="utf-8-sig")
    buy_rows.to_csv(f"screen_{as_of_str}_buy.csv", index=False, encoding="utf-8-sig")
    sell_rows.to_csv(f"screen_{as_of_str}_sell.csv", index=False, encoding="utf-8-sig")

    print(
        "as_of:",
        as_of_str,
        "runtime_date:",
        runtime_str,
        "total:",
        len(all_rows),
        "buy:",
        len(buy_rows),
        "sell:",
        len(sell_rows),
    )

    if not sell_rows.empty:
        print(
            sell_rows[
                [
                    "ticker",
                    "sell_score",
                    "sell_reason",
                    "close",
                    "pullback_20d",
                    "dev200",
                    "rsi",
                    "adx14",
                    "days_to_earnings",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
