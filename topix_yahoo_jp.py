#!/usr/bin/env python3
"""Download TOPIX daily OHLC data from Yahoo! Finance Japan history pages."""
from __future__ import annotations

from html.parser import HTMLParser
from typing import Any

import pandas as pd
import requests

DEFAULT_TOPIX_TICKER = "998405.T"
EXPECTED_HEADERS = ("日付", "始値", "高値", "安値", "終値")
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


class _TableParser(HTMLParser):
    """Collect text cells from ordinary HTML tables without optional parsers."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"th", "td"} and self._cell_parts is not None and self._row is not None:
            text = " ".join("".join(self._cell_parts).split())
            self._row.append(text)
            self._cell_parts = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if self._row:
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def _number(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).replace(",", "").strip()
    if text in {"", "-", "--", "---"}:
        return float("nan")
    return float(text)


def parse_index_history_html(html: str) -> pd.DataFrame:
    """Parse the visible date/open/high/low/close history table."""
    parser = _TableParser()
    parser.feed(html)

    selected: list[list[str]] | None = None
    header_positions: dict[str, int] = {}
    for table in parser.tables:
        for row_index, row in enumerate(table):
            positions = {header: row.index(header) for header in EXPECTED_HEADERS if header in row}
            if len(positions) == len(EXPECTED_HEADERS):
                selected = table[row_index + 1 :]
                header_positions = positions
                break
        if selected is not None:
            break

    if selected is None:
        raise ValueError("TOPIX history table was not found")

    rows = []
    required_max = max(header_positions.values())
    for cells in selected:
        if len(cells) <= required_max:
            continue
        date = pd.to_datetime(cells[header_positions["日付"]], errors="coerce")
        if pd.isna(date):
            continue
        rows.append(
            {
                "Date": pd.Timestamp(date).normalize(),
                "Open": _number(cells[header_positions["始値"]]),
                "High": _number(cells[header_positions["高値"]]),
                "Low": _number(cells[header_positions["安値"]]),
                "Close": _number(cells[header_positions["終値"]]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    frame = pd.DataFrame(rows).set_index("Date").sort_index()
    return frame.loc[~frame.index.duplicated(keep="last"), PRICE_COLUMNS]


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
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
        "Referer": f"https://finance.yahoo.co.jp/quote/{ticker}",
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
        response = client.get(url, params=params, headers=headers, timeout=30)
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
        return pd.DataFrame(columns=PRICE_COLUMNS)

    result = pd.concat(frames).sort_index()
    result = result[~result.index.duplicated(keep="last")]
    result = result.loc[(result.index >= start_date) & (result.index <= end_date)]
    return result[PRICE_COLUMNS]
