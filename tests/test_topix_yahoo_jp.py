from __future__ import annotations

import pandas as pd

from topix_yahoo_jp import parse_index_history_html


def test_parse_index_history_html() -> None:
    html = r'''
    <script>
    {"mainDomesticIndexHistory":{"histories":[
      {"date":"2026/08/04","openPrice":"3,100.25","highPrice":"3,120.50","lowPrice":"3,080.75","closePrice":"3,115.00"},
      {"date":"2026/08/05","openPrice":"3,116.00","highPrice":"3,130.00","lowPrice":"3,090.00","closePrice":"3,125.50"}
    ],"paging":{"page":1}}}
    </script>
    '''

    frame = parse_index_history_html(html)

    assert list(frame.columns) == ["Open", "High", "Low", "Close"]
    assert frame.index.tolist() == [pd.Timestamp("2026-08-04"), pd.Timestamp("2026-08-05")]
    assert frame.loc[pd.Timestamp("2026-08-04"), "Open"] == 3100.25
    assert frame.loc[pd.Timestamp("2026-08-05"), "Close"] == 3125.50
