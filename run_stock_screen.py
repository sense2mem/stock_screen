# -*- coding: utf-8 -*-
"""Safe entry point for the daily stock screen.

The screening calculations remain in screen_yearend.py.

Safety responsibilities of this wrapper:
- separate workflow runtime date from the intended market session
- resolve the actual market session consumed by the scorer
- reject stale market data
- retry once when Yahoo returns stale bars
"""

from collections import Counter
from pathlib import Path
import time

import exchange_calendars as xcals
import pandas as pd

import screen_yearend as screen


MIN_PRICE_DATE_COVERAGE = 0.50

# Initial attempt + one retry.
SCREEN_ATTEMPTS = 2
STALE_RETRY_SECONDS = 120

# GitHub Actions nominal schedule:
#   17 11 * * 1-5
# = 20:17 JST
SCHEDULE_HOUR_JST = 20
SCHEDULE_MINUTE_JST = 17

TSE_CALENDAR = "XTKS"


def resolve_signal_date(price_dates, total_tickers):
    """Resolve the actual market session by majority vote.

    Dates come from the exact final bar passed into score_buy_signals().
    """
    normalized = []

    for value in price_dates:
        if value is None or pd.isna(value):
            continue

        normalized.append(
            pd.Timestamp(value).normalize()
        )

    if not normalized:
        raise RuntimeError(
            "Unable to resolve signal date: "
            "no screened ticker price dates were obtained"
        )

    counts = Counter(normalized)
    signal_date, votes = counts.most_common(1)[0]

    required = max(
        1,
        int(total_tickers * MIN_PRICE_DATE_COVERAGE),
    )

    if votes < required:
        summary = ", ".join(
            f"{day.strftime('%Y-%m-%d')}={count}"
            for day, count in counts.most_common(5)
        )

        raise RuntimeError(
            "Unable to resolve signal date with sufficient coverage: "
            f"winner={signal_date.strftime('%Y-%m-%d')} "
            f"votes={votes} "
            f"required={required} "
            f"total_tickers={total_tickers} "
            f"top_dates=[{summary}]"
        )

    return signal_date


def resolve_expected_market_date(now=None):
    """Return the market session this scheduled run is expected to process.

    The nominal workflow slot is 20:17 JST.

    If GitHub Actions starts after midnight but before the next 20:17 slot,
    the run is still associated with the preceding scheduled date.

    The XTKS calendar then converts that date to the latest valid TSE
    trading session, so weekends and JPX holidays do not create false
    expected dates.
    """
    if now is None:
        now_jst = pd.Timestamp.now(tz=screen.TZ)
    else:
        now_jst = pd.Timestamp(now)

        if now_jst.tzinfo is None:
            now_jst = now_jst.tz_localize(screen.TZ)
        else:
            now_jst = now_jst.tz_convert(screen.TZ)

    nominal_slot = (
        now_jst.normalize()
        + pd.Timedelta(
            hours=SCHEDULE_HOUR_JST,
            minutes=SCHEDULE_MINUTE_JST,
        )
    )

    if now_jst >= nominal_slot:
        anchor = now_jst.normalize()
    else:
        anchor = (
            now_jst.normalize()
            - pd.Timedelta(days=1)
        )

    cal = xcals.get_calendar(TSE_CALENDAR)

    sessions = cal.sessions_in_range(
        (anchor - pd.Timedelta(days=14)).date(),
        anchor.date(),
    )

    if len(sessions) == 0:
        raise RuntimeError(
            "Unable to resolve expected market date "
            "from XTKS calendar"
        )

    return pd.Timestamp(sessions[-1]).normalize()


def ensure_expected_signal_date(
    signal_date,
    expected_date,
):
    """Reject stale or otherwise unexpected market-session data."""
    actual = pd.Timestamp(signal_date).normalize()
    expected = pd.Timestamp(expected_date).normalize()

    if actual != expected:
        raise RuntimeError(
            "MARKET DATA DATE MISMATCH: "
            f"expected_market_date={expected.strftime('%Y-%m-%d')} "
            f"actual_market_date={actual.strftime('%Y-%m-%d')}"
        )


def mark_ticker_bar_freshness(
    tech_df,
    expected_date,
):
    """Mark whether each ticker has a real traded bar for expected_date."""
    out = tech_df.copy()
    expected = pd.Timestamp(
        expected_date
    ).normalize()

    if "price_date" not in out.columns:
        raise RuntimeError(
            "price_date missing from technical screen output"
        )

    price_dates = pd.to_datetime(
        out["price_date"],
        errors="coerce",
    ).dt.normalize()

    out["price_date"] = price_dates
    out["bar_is_current"] = (
        price_dates == expected
    )

    stale = ~out["bar_is_current"]

    if "skip_reason" not in out.columns:
        out["skip_reason"] = ""

    out["skip_reason"] = (
        out["skip_reason"]
        .fillna("")
        .astype(str)
    )

    cur = out.loc[
        stale,
        "skip_reason",
    ]

    out.loc[
        stale,
        "skip_reason",
    ] = cur.where(
        cur.ne(""),
        "stale_ticker_bar",
    )

    add_mask = (
        stale
        & out["skip_reason"].ne("")
        & ~out["skip_reason"].str.contains(
            r"(?:^|,)stale_ticker_bar(?:,|$)",
            regex=True,
        )
    )

    out.loc[
        add_mask,
        "skip_reason",
    ] = (
        out.loc[
            add_mask,
            "skip_reason",
        ]
        + ",stale_ticker_bar"
    )

    return out


def score_and_observe_price_date(
    score_func,
    price_dates,
    df_daily,
):
    """Run the unchanged scorer and record its exact final input bar."""
    result = score_func(df_daily)

    if df_daily is not None and not df_daily.empty:
        last_date = pd.Timestamp(
            df_daily.index[-1]
        ).normalize()

        price_dates.append(last_date)

    return result


def run_screen_and_capture_price_dates(tickers):
    """Run screen_tech while observing exact scorer input dates."""
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
        screen.score_buy_signals = original_score_func

    return tech_df, runtime_end, price_dates


def run_screen_until_expected(
    tickers,
    expected_date,
):
    """Retry the whole technical screen once when Yahoo data is stale."""
    expected = pd.Timestamp(
        expected_date
    ).normalize()

    for attempt in range(
        1,
        SCREEN_ATTEMPTS + 1,
    ):
        print(
            "screen attempt:",
            f"{attempt}/{SCREEN_ATTEMPTS}",
            f"expected_market_date={expected.strftime('%Y-%m-%d')}",
            flush=True,
        )

        (
            tech_df,
            runtime_end,
            price_dates,
        ) = run_screen_and_capture_price_dates(
            tickers
        )

        signal_date = resolve_signal_date(
            price_dates,
            total_tickers=len(tickers),
        )

        print(
            "screen attempt result:",
            f"actual_market_date={signal_date.strftime('%Y-%m-%d')}",
            f"screened_price_dates={len(price_dates)}/{len(tickers)}",
            flush=True,
        )

        if signal_date == expected:
            return (
                tech_df,
                runtime_end,
                price_dates,
                signal_date,
            )

        # Future data is not something a retry should repair.
        if signal_date > expected:
            ensure_expected_signal_date(
                signal_date,
                expected,
            )

        if attempt < SCREEN_ATTEMPTS:
            runtime_str = pd.Timestamp(
                runtime_end
            ).strftime("%Y-%m-%d")

            # Do not carry the first attempt's errors CSV into
            # a successful second attempt.
            Path(
                f"screen_{runtime_str}_errors.csv"
            ).unlink(missing_ok=True)

            print(
                "WARNING: stale market data detected; "
                f"expected={expected.strftime('%Y-%m-%d')} "
                f"actual={signal_date.strftime('%Y-%m-%d')}. "
                f"Retrying after {STALE_RETRY_SECONDS} seconds.",
                flush=True,
            )

            time.sleep(
                STALE_RETRY_SECONDS
            )

            continue

        ensure_expected_signal_date(
            signal_date,
            expected,
        )

    raise AssertionError(
        "unreachable"
    )


def rename_error_csv(
    runtime_end,
    signal_date,
):
    """Keep optional errors CSV aligned with the resolved signal date."""
    runtime_str = pd.Timestamp(
        runtime_end
    ).strftime("%Y-%m-%d")

    signal_str = pd.Timestamp(
        signal_date
    ).strftime("%Y-%m-%d")

    if runtime_str == signal_str:
        return

    src = Path(
        f"screen_{runtime_str}_errors.csv"
    )

    dst = Path(
        f"screen_{signal_str}_errors.csv"
    )

    if src.exists():
        src.replace(dst)


def main():
    tickers = screen.load_tickers_from_excel_bcol(
        "data_e.xls",
        sheet_name="Sheet1",
    )

    expected_date = resolve_expected_market_date()

    print(
        "expected market session:",
        expected_date.strftime("%Y-%m-%d"),
        flush=True,
    )

    (
        tech_df,
        runtime_end,
        price_dates,
        signal_date,
    ) = run_screen_until_expected(
        tickers,
        expected_date,
    )

    runtime_str = pd.Timestamp(
        runtime_end
    ).strftime("%Y-%m-%d")

    as_of_str = signal_date.strftime(
        "%Y-%m-%d"
    )

    print(
        "signal-date resolution:",
        f"runtime_date={runtime_str}",
        f"expected_market_date={expected_date.strftime('%Y-%m-%d')}",
        f"actual_market_date={as_of_str}",
        f"screened_price_dates={len(price_dates)}/{len(tickers)}",
        flush=True,
    )

    ensure_expected_signal_date(
        signal_date,
        expected_date,
    )

    tech_df = mark_ticker_bar_freshness(
        tech_df,
        expected_date,
    )

    current_count = int(
        tech_df["bar_is_current"].sum()
    )

    stale_count = int(
        (~tech_df["bar_is_current"]).sum()
    )

    print(
        "per-ticker market-date validation:",
        f"current={current_count}",
        f"stale={stale_count}",
        flush=True,
    )

    if signal_date > pd.Timestamp(
        runtime_end
    ).normalize():
        raise RuntimeError(
            f"Resolved signal date {as_of_str} "
            f"is later than runtime date {runtime_str}"
        )

    rename_error_csv(
        runtime_end,
        signal_date,
    )

    val_df = screen.fetch_per_pbr_batch(
        tech_df[
            "ticker"
        ].dropna().astype(str).tolist(),
        batch_size=200,
        pause_sec=0.2,
    )

    tech_t = set(
        tech_df[
            "ticker"
        ].astype(str).str.upper()
    )

    val_t = set(
        val_df[
            "ticker"
        ].astype(str).str.upper()
    )

    print(
        "intersection(tech,val):",
        len(tech_t & val_t),
        "/",
        len(tech_t),
    )

    print(
        "val non-null pe:",
        val_df["pe_ttm"].notna().sum(),
        "/",
        len(val_df),
    )

    try:
        all_rows, buy_rows, sell_rows = (
            screen.apply_filters_and_make_trades(
                tech_df,
                val_df,

                # Important:
                # earnings/regime filters must use the
                # market-session date, not a delayed runtime date.
                signal_date,
            )
        )

    except Exception as exc:
        print(
            "ERROR in apply_filters_and_make_trades:",
            repr(exc),
        )

        print(
            "tech_df columns:",
            tech_df.columns.tolist(),
        )

        print(
            "val_df columns:",
            val_df.columns.tolist(),
        )

        raise

    all_rows.to_csv(
        f"screen_{as_of_str}_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    buy_rows.to_csv(
        f"screen_{as_of_str}_buy.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sell_rows.to_csv(
        f"screen_{as_of_str}_sell.csv",
        index=False,
        encoding="utf-8-sig",
    )

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
