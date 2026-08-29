# -*- coding: utf-8 -*-
"""Safe entry point for the daily stock screen.

The screening calculations remain in screen_yearend.py. This wrapper separates
workflow execution time from the actual market session used for signal labels.
"""

from collections import Counter
from pathlib import Path
import time

import pandas as pd
import yfinance as yf

import screen_yearend as screen


PRICE_DATE_PROBE_DAYS = "10d"
PRICE_DATE_CHUNK_SIZE = 200
MIN_PRICE_DATE_COVERAGE = 0.50


def resolve_signal_date(price_dates, total_tickers):
    """Resolve the market session by majority vote across ticker price dates.

    There is deliberately no runtime-date fallback. A signal file must never be
    labeled with a weekend/holiday merely because a scheduled workflow was late.
    """
    normalized = []
    for value in price_dates:
        if value is None or pd.isna(value):
            continue
        normalized.append(pd.Timestamp(value).normalize())

    if not normalized:
        raise RuntimeError("Unable to resolve signal date: no ticker price dates were obtained")

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


def _extract_last_price_date(df_all, ticker):
    if df_all is None or df_all.empty:
        return None

    if isinstance(df_all.columns, pd.MultiIndex):
        if ticker in df_all.columns.get_level_values(0):
            d = df_all[ticker].copy()
        elif ticker in df_all.columns.get_level_values(1):
            d = df_all.xs(ticker, axis=1, level=1).copy()
        else:
            return None
    else:
        d = df_all.copy()

    d = screen._normalize_ohlcv(d, ticker)
    if d is None or d.empty or "Close" not in d.columns:
        return None

    close = pd.to_numeric(d["Close"], errors="coerce").dropna()
    if close.empty:
        return None

    return pd.Timestamp(close.index[-1]).normalize()


def fetch_price_dates(tickers, chunk_size=PRICE_DATE_CHUNK_SIZE, max_retry=2):
    """Fetch only recent daily prices and return each ticker's latest session date."""
    price_dates = []
    total_chunks = (len(tickers) + chunk_size - 1) // chunk_size

    for ci, chunk in enumerate(screen._chunks(tickers, chunk_size), start=1):
        print(
            f"price-date probe [{ci}/{total_chunks}] tickers={len(chunk)}",
            flush=True,
        )

        df_all = None
        last_err = None
        for attempt in range(max_retry + 1):
            try:
                df_all = yf.download(
                    tickers=" ".join(chunk),
                    period=PRICE_DATE_PROBE_DAYS,
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    group_by="ticker",
                )
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                print(
                    f"  price-date retry {attempt + 1}/{max_retry + 1}: {repr(exc)}",
                    flush=True,
                )
                time.sleep(1.0 + attempt)

        if df_all is None or last_err is not None:
            price_dates.extend([None] * len(chunk))
            continue

        for ticker in chunk:
            try:
                price_dates.append(_extract_last_price_date(df_all, ticker))
            except Exception as exc:
                print(f"  price-date error {ticker}: {repr(exc)}", flush=True)
                price_dates.append(None)

    return price_dates


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

    tech_df, runtime_end = screen.screen_tech(
        tickers,
        chunk_size=20,
        pause_sec=0.2,
        max_retry=2,
    )

    runtime_str = pd.Timestamp(runtime_end).strftime("%Y-%m-%d")

    # Resolve the signal date from the same screened ticker universe. This is a
    # lightweight recent-price probe and does not alter BUY/SELL calculations.
    price_dates = fetch_price_dates(tickers)
    signal_date = resolve_signal_date(price_dates, total_tickers=len(tickers))
    as_of_str = signal_date.strftime("%Y-%m-%d")

    successful_dates = sum(d is not None and not pd.isna(d) for d in price_dates)
    print(
        "signal-date resolution:",
        f"runtime_date={runtime_str}",
        f"actual_market_date={as_of_str}",
        f"price_dates={successful_dates}/{len(tickers)}",
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
