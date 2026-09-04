# -*- coding: utf-8 -*-
"""Runtime safety hook for the daily stock screen.

Python imports ``sitecustomize`` automatically when this repository is on
``sys.path``.  Keep the hook deliberately narrow: it only patches the existing
``run_stock_screen.py`` entry point, leaving tests/backtests and other scripts
unchanged.

This avoids changing the mature screening/archiving entry point solely to add
an earnings-date data-quality guard.
"""

from pathlib import Path
import sys


def _install_daily_screen_earnings_guard():
    if Path(sys.argv[0]).name != "run_stock_screen.py":
        return

    import pandas as pd

    import screen_yearend as screen
    from earnings_date_guard import repair_earnings_dates

    original = screen.apply_filters_and_make_trades

    def guarded_apply_filters_and_make_trades(all_df, val_df, end_ts):
        repaired = repair_earnings_dates(
            val_df,
            all_df,
            as_of_date=end_ts,
            buy_score_min=screen.PULLBACK_SCORE_MIN,
            sell_score_min=screen.SELL_SCORE_MIN,
        )

        if "earn_date_raw_stale" in repaired.columns:
            stale_count = int(
                repaired["earn_date_raw_stale"]
                .fillna(False)
                .astype(bool)
                .sum()
            )
        else:
            stale_count = 0

        if "earn_date_source" in repaired.columns:
            source_counts = (
                repaired["earn_date_source"]
                .fillna("unknown")
                .astype(str)
                .value_counts()
                .to_dict()
            )
        else:
            source_counts = {}

        print(
            "earnings-date validation:",
            f"as_of={pd.Timestamp(end_ts).strftime('%Y-%m-%d')}",
            f"stale_quote_rows={stale_count}",
            f"sources={source_counts}",
            flush=True,
        )

        return original(
            all_df,
            repaired,
            end_ts,
        )

    screen.apply_filters_and_make_trades = (
        guarded_apply_filters_and_make_trades
    )


_install_daily_screen_earnings_guard()
