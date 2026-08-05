#!/usr/bin/env python3
"""Download TOPIX daily OHLC data from Yahoo! Finance Japan history pages."""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
import requests

DEFAULT_TOPIX_TICKER = "998405.T"
_HISTORY_PATTERN = re.compile(
    r'"mainDomesticIndexHistory"\s*:\s*\{[\s\S]*?'
    r'"histories"\s*:\s*(\[[\s\S]*?\])\s*,\s*"paging"',
    re.DOTALL,
)


def _number(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).replace(",", "").strip()
    if text in {"", "-", "--", "---"}:
        return float("nan")
    return float(text)


def parse_index_history_html(html: str) -> pd.DataFrame:
    """Parse embedded domestic-index history JSON from one Yahoo Japan page."""
    match = _HISTORY_PATTERN.search(html)
    if not match:
        raise ValueError("domestic index history JSON was not found")
    records = json.loads(match.group(1))
    rows = []
    for item in records:
        date = pd.to_datetime(item.get("date"), errors="coerce")
        if pd.isna(date):
            continue
        rows.append(
            {
                "Date": pd.Timestamp(date).normalize(),
                "Open": _number(item.get("openPrice")),
                "High": _number(item.get("highPrice")),
                "Low": _number(item.get("lowPrice")),
                "Close": _number(item.get("closePrice")),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    frame = pd.DataFrame(rows).set_index("Date").sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def download_index_history(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    session: requests.Session | None = None,
    max_pages: int = 100,
) -> pd.DataFrame:
    """Download daily OHLC rows for a Yahoo Japan domestic index."""
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if start_date > end_date:
        raise ValueError("start must be on or before end")

    client = session or requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "Chrome/150.0.0.0 Safari/537.36"
        )
    }
    frames: list[pd.DataFrame] = []
    seen_dates: set[pd.Timestamp] = set()

    for page in range(1, max_pages + 1):
        url = f"https://finance.yahoo.co.jp/quote/{ticker}/history"
        params = {
            "from": start_date.strftime("%Y%m%d"),
            "to": end_date.strftime("%Y%m%d"),
            "timeFrame": "d",
            "page": page,
        }
        response = client.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        frame = parse_index_history_html(response.text)
        if frame.empty:
            break

        new_dates = [date for date in frame.index if date not in seen_dates]
        if not new_dates:
            break
        seen_dates.update(new_dates)
        frames.append(frame.loc[new_dates])

        if frame.index.min() <= start_date or len(frame) < 20:
            break

    if not frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    result = result.loc[(result.index >= start_date) & (result.index <= end_date)]
    return result[["Open", "High", "Low", "Close"]]
