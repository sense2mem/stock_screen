#!/usr/bin/env python3
"""Create condition A-C signal flags and a GitHub Actions summary."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

CONDITION_COLUMNS = [
    "signal_date", "ticker", "name", "score", "sell_score", "condition_a",
    "condition_b", "condition_c", "condition_labels",
    "pullback_from_60d_high_pct", "ma20_distance_pct", "plus_di14",
    "minus_di14", "di_spread", "rsi", "adx14", "pattern_stage",
]


def build_condition_signals(features: pd.DataFrame) -> pd.DataFrame:
    """Return rows matching at least one condition, using signal-date features only."""
    frame = features.copy()
    for column in ("pullback_from_60d_high_pct", "ma20_distance_pct", "di_spread"):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["condition_a"] = frame["pullback_from_60d_high_pct"].le(-12.0).fillna(False)
    frame["condition_b"] = frame["condition_a"] & frame["ma20_distance_pct"].ge(1.0).fillna(False)
    frame["condition_c"] = (
        frame["ma20_distance_pct"].ge(1.0).fillna(False)
        & frame["di_spread"].ge(3.0).fillna(False)
    )

    def labels(row: pd.Series) -> str:
        return "|".join(
            label
            for label, column in (("A", "condition_a"), ("B", "condition_b"), ("C", "condition_c"))
            if bool(row[column])
        )

    frame["condition_labels"] = frame.apply(labels, axis=1)
    matched = frame[frame["condition_labels"].ne("")].copy().reindex(columns=CONDITION_COLUMNS)
    if not matched.empty:
        matched["signal_date"] = pd.to_datetime(matched["signal_date"], errors="coerce")
        matched = matched.sort_values(["signal_date", "ticker"], kind="stable")
    return matched.reset_index(drop=True)


def build_legacy_condition_c_signal_day(signals: pd.DataFrame) -> pd.DataFrame:
    """Return the exact conventional condition-C rows at the score8 signal close."""
    if signals.empty or "condition_c" not in signals:
        return pd.DataFrame(columns=CONDITION_COLUMNS)
    result = signals[signals["condition_c"].fillna(False)].copy()
    return result.reindex(columns=CONDITION_COLUMNS).reset_index(drop=True)


def _display(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value).replace("|", "\\|")


def build_markdown_summary(signals: pd.DataFrame) -> str:
    legacy_c = build_legacy_condition_c_signal_day(signals)
    lines = [
        "## Condition A-C signals",
        "",
        "- A: 60-day closing high pullback <= -12%",
        "- B: A and price >= 1% above MA20",
        "- C: price >= 1% above MA20 and DI spread >= 3",
        "- C uses the conventional signal-date DI values stored by the score8 screen.",
        "",
        "| Condition | Count |",
        "|---|---:|",
    ]
    for label, column in (("A", "condition_a"), ("B", "condition_b"), ("C", "condition_c")):
        count = int(signals[column].fillna(False).sum()) if column in signals else 0
        lines.append(f"| {label} | {count} |")
    lines += ["", f"- Conventional signal-day C rows written: **{len(legacy_c)}**"]
    for label, column in (("A", "condition_a"), ("B", "condition_b"), ("C", "condition_c")):
        selected = signals[signals[column].fillna(False)] if column in signals else signals.iloc[0:0]
        lines.extend(["", f"### Condition {label}", ""])
        if selected.empty:
            lines.append("No matching signals.")
            continue
        lines.extend([
            "| Signal date | Ticker | Name | Conditions | 60d-high pullback % | MA20 distance % | DI spread |",
            "|---|---|---|---|---:|---:|---:|",
        ])
        for _, row in selected.iterrows():
            date = pd.to_datetime(row.get("signal_date"), errors="coerce")
            date_text = date.strftime("%Y-%m-%d") if pd.notna(date) else "-"
            lines.append(
                "| "
                + " | ".join([
                    date_text, _display(row.get("ticker")), _display(row.get("name")),
                    _display(row.get("condition_labels")),
                    _display(row.get("pullback_from_60d_high_pct")),
                    _display(row.get("ma20_distance_pct")), _display(row.get("di_spread")),
                ])
                + " |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="signal_path_report/signal_features.csv")
    parser.add_argument("--output", default="signal_path_report/condition_signals.csv")
    parser.add_argument(
        "--condition-c-output",
        default="signal_path_report/condition_c_signal_day.csv",
        help="Dedicated conventional score8 signal-day condition-C CSV.",
    )
    parser.add_argument(
        "--github-summary", default=os.environ.get("GITHUB_STEP_SUMMARY", ""),
        help="Optional GitHub step-summary file to append.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features, encoding="utf-8-sig", dtype={"ticker": "string"})
    signals = build_condition_signals(features)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output, index=False, encoding="utf-8-sig")
    condition_c_output = Path(args.condition_c_output)
    condition_c_output.parent.mkdir(parents=True, exist_ok=True)
    build_legacy_condition_c_signal_day(signals).to_csv(
        condition_c_output, index=False, encoding="utf-8-sig"
    )
    if args.github_summary:
        with open(args.github_summary, "a", encoding="utf-8") as handle:
            handle.write(build_markdown_summary(signals))


if __name__ == "__main__":
    main()
