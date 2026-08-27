from datetime import date

import pandas as pd

from run_stock_screen import resolve_signal_date


def test_resolve_signal_date_uses_market_last_date():
    runtime_end = pd.Timestamp("2026-08-28")
    regime = {"last_date": date(2026, 8, 27)}

    actual = resolve_signal_date(runtime_end, regime)

    assert actual == pd.Timestamp("2026-08-27")


def test_resolve_signal_date_falls_back_to_runtime_date():
    runtime_end = pd.Timestamp("2026-08-28")
    regime = {"last_date": None}

    actual = resolve_signal_date(runtime_end, regime)

    assert actual == pd.Timestamp("2026-08-28")


def test_resolve_signal_date_accepts_timestamp():
    runtime_end = pd.Timestamp("2026-08-31")
    regime = {"last_date": pd.Timestamp("2026-08-28 15:00:00")}

    actual = resolve_signal_date(runtime_end, regime)

    assert actual == pd.Timestamp("2026-08-28")
