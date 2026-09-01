import pandas as pd

from screen_yearend import _filter_scoring_bars


def test_zero_volume_bar_is_removed():
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 101.0],
            "Low": [99.0, 101.0],
            "Close": [101.0, 101.0],
            "Volume": [1000.0, 0.0],
        },
        index=pd.to_datetime(
            [
                "2026-08-31",
                "2026-09-01",
            ]
        ),
    )

    actual = _filter_scoring_bars(
        df,
        pd.Timestamp("2026-09-01"),
    )

    assert actual.index.tolist() == [
        pd.Timestamp("2026-08-31")
    ]


def test_valid_positive_volume_bar_is_kept():
    df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [102.0],
            "Low": [99.0],
            "Close": [101.0],
            "Volume": [1000.0],
        },
        index=pd.to_datetime(
            ["2026-09-01"]
        ),
    )

    actual = _filter_scoring_bars(
        df,
        pd.Timestamp("2026-09-01"),
    )

    assert actual.index.tolist() == [
        pd.Timestamp("2026-09-01")
    ]


def test_nan_ohlcv_bar_is_removed():
    df = pd.DataFrame(
        {
            "Open": [100.0, float("nan")],
            "High": [102.0, float("nan")],
            "Low": [99.0, float("nan")],
            "Close": [101.0, float("nan")],
            "Volume": [1000.0, float("nan")],
        },
        index=pd.to_datetime(
            [
                "2026-08-31",
                "2026-09-01",
            ]
        ),
    )

    actual = _filter_scoring_bars(
        df,
        pd.Timestamp("2026-09-01"),
    )

    assert actual.index.tolist() == [
        pd.Timestamp("2026-08-31")
    ]


def test_ticker_freshness_marks_old_bar_stale():
    from run_stock_screen import mark_ticker_bar_freshness

    df = pd.DataFrame(
        {
            "ticker": ["1111.T", "2222.T"],
            "price_date": [
                pd.Timestamp("2026-09-01"),
                pd.Timestamp("2026-08-31"),
            ],
            "skip_reason": ["", ""],
        }
    )

    actual = mark_ticker_bar_freshness(
        df,
        pd.Timestamp("2026-09-01"),
    )

    assert actual["bar_is_current"].tolist() == [
        True,
        False,
    ]

    assert actual.loc[
        actual["ticker"].eq("2222.T"),
        "skip_reason",
    ].iloc[0] == "stale_ticker_bar"


def test_ticker_freshness_preserves_existing_reason():
    from run_stock_screen import mark_ticker_bar_freshness

    df = pd.DataFrame(
        {
            "ticker": ["2222.T"],
            "price_date": [
                pd.Timestamp("2026-08-31"),
            ],
            "skip_reason": [
                "insufficient_daily",
            ],
        }
    )

    actual = mark_ticker_bar_freshness(
        df,
        pd.Timestamp("2026-09-01"),
    )

    assert not bool(actual["bar_is_current"].iloc[0])

    assert actual["skip_reason"].iloc[0] == (
        "insufficient_daily,stale_ticker_bar"
    )
