# -*- coding: utf-8 -*-
"""Safe entry point for the daily stock screen.

The screening calculations remain in screen_yearend.py.  This wrapper only
controls the output date so a delayed GitHub Actions run cannot label prior-day
market data with the next calendar date.
"""

from pathlib import Path

import pandas as pd

import screen_yearend as screen


def resolve_signal_date(runtime_end, regime):
    """Return the latest actual market-data date, with runtime date as fallback."""
    last_date = (regime or {}).get("last_date")
    if last_date is not None and not pd.isna(last_date):
        return pd.Timestamp(last_date).normalize()
    return pd.Timestamp(runtime_end).normalize()


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
    regime = screen.get_market_regime(as_of=runtime_str)
    signal_date = resolve_signal_date(runtime_end, regime)
    as_of_str = signal_date.strftime("%Y-%m-%d")

    if regime.get("last_date") is None:
        print(
            "WARNING: market last_date unavailable; "
            f"falling back to runtime date {runtime_str}",
            flush=True,
        )
    elif as_of_str != runtime_str:
        print(
            "INFO: delayed/non-trading-day execution detected: "
            f"runtime_date={runtime_str} actual_market_date={as_of_str}",
            flush=True,
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
        # Keep the original filtering/scoring behavior unchanged.  The existing
        # function already uses the market's last_date for earnings-day logic.
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
