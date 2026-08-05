#!/usr/bin/env python3
"""Run condition-C TOPIX analysis using Yahoo! Finance Japan index history."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

import condition_c_topix_analysis as analysis
from fixed_holding_backtest import download_prices
from topix_yahoo_jp import download_index_history

LOGGER = logging.getLogger("condition_c_topix_analysis_yahoojp")


def run(args) -> None:
    path = Path(args.signals)
    signals = pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": "string"})
    if signals.empty:
        detail = pd.DataFrame(columns=analysis.DETAIL_COLUMNS)
        summary = analysis.summarize(detail)
        analysis.write_outputs(args.output_dir, detail, summary)
        return

    signals["signal_date"] = pd.to_datetime(
        signals["signal_date"], errors="coerce"
    ).dt.normalize()
    tickers = sorted(signals["ticker"].dropna().astype(str).unique())
    start = signals["signal_date"].min() - pd.Timedelta(days=7)
    end = pd.Timestamp.now().tz_localize(None).normalize() + pd.Timedelta(days=2)

    stock_prices, failures = download_prices(tickers, start, end)
    stock_prices = {
        ticker: analysis.confirmed_prices(frame)
        for ticker, frame in stock_prices.items()
    }

    benchmark = download_index_history(args.benchmark_ticker, start, end)
    benchmark = analysis.confirmed_prices(benchmark) if not benchmark.empty else benchmark
    if benchmark.empty:
        raise RuntimeError(
            f"TOPIX history download returned no rows for {args.benchmark_ticker}"
        )
    missing = analysis.REQUIRED_BENCHMARK_COLUMNS - set(benchmark.columns)
    if missing:
        raise RuntimeError(f"TOPIX history is missing columns: {sorted(missing)}")

    LOGGER.info(
        "TOPIX rows=%d range=%s..%s",
        len(benchmark), benchmark.index.min().date(), benchmark.index.max().date(),
    )
    detail = analysis.build_analysis(
        signals,
        stock_prices,
        benchmark,
        args.benchmark_ticker,
        args.stop_loss_pct,
        args.holding_days,
        failures,
    )
    summary = analysis.summarize(detail)
    analysis.write_outputs(args.output_dir, detail, summary)

    matured = detail[detail["strategy_return_pct"].notna()]
    completed = detail[detail["status"].eq("CLOSED")]
    if len(matured) > 0 and len(completed) == 0:
        statuses = detail.loc[matured.index, "status"].value_counts().to_dict()
        raise RuntimeError(
            "TOPIX comparison produced zero completed rows for matured trades: "
            f"{statuses}"
        )

    if args.github_summary:
        with open(args.github_summary, "a", encoding="utf-8") as handle:
            handle.write(
                analysis.build_markdown(
                    summary,
                    detail,
                    args.benchmark_ticker,
                    args.stop_loss_pct,
                    args.holding_days,
                )
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(analysis.parse_args())
