from __future__ import annotations

import pandas as pd

from topix_yahoo_jp import parse_index_history_html


def test_parse_index_history_html() -> None:
    html = r'''
    <html><body>
      <table>
        <thead>
          <tr><th>日付</th><th>始値</th><th>高値</th><th>安値</th><th>終値</th><th>出来高</th></tr>
        </thead>
        <tbody>
          <tr><td>2026年8月4日</td><td>3,100.25</td><td>3,120.50</td><td>3,080.75</td><td>3,115.00</td><td>---</td></tr>
          <tr><td>2026年8月5日</td><td>3,116.00</td><td>3,130.00</td><td>3,090.00</td><td>3,125.50</td><td>---</td></tr>
        </tbody>
      </table>
    </body></html>
    '''

    frame = parse_index_history_html(html)

    assert list(frame.columns) == ["Open", "High", "Low", "Close"]
    assert frame.index.tolist() == [pd.Timestamp("2026-08-04"), pd.Timestamp("2026-08-05")]
    assert frame.loc[pd.Timestamp("2026-08-04"), "Open"] == 3100.25
    assert frame.loc[pd.Timestamp("2026-08-05"), "Close"] == 3125.50


def test_ignores_unrelated_tables() -> None:
    html = r'''
    <table><tr><th>項目</th><th>値</th></tr><tr><td>名称</td><td>TOPIX</td></tr></table>
    <table>
      <tr><th>日付</th><th>始値</th><th>高値</th><th>安値</th><th>終値</th></tr>
      <tr><td>2026/08/04</td><td>3,100</td><td>3,120</td><td>3,080</td><td>3,115</td></tr>
    </table>
    '''

    frame = parse_index_history_html(html)

    assert len(frame) == 1
    assert frame.iloc[0]["Close"] == 3115.0
