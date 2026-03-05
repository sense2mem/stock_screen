# -*- coding: utf-8 -*-
"""
backtest.py
-----------
GitHub Actions 上でスクリーニング結果を使ってバックテストを実行し、
HTMLレポートを生成するスクリプト。

【動作フロー】
  1. screen_YYYY-MM-DD_all.csv を過去分まとめて読み込む
  2. 買いシグナル翌日始値でエントリー、売りシグナル翌日始値でエグジット
  3. ポジションサイジング・損益を計算
  4. HTMLレポート（チャート付き）を出力する

【依存ライブラリ】
  pip install yfinance pandas numpy jinja2
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# =========================
# 設定
# =========================
RESULTS_DIR      = os.environ.get("RESULTS_DIR", "results")  # 環境変数で上書き可能
OUTPUT_DIR       = "backtest_report"  # レポート出力先
INITIAL_CAPITAL  = 1_000_000         # 初期資金（円）
MAX_POSITIONS    = 5                  # 同時保有銘柄数の上限
POSITION_SIZE_PCT = 0.10             # 1銘柄あたりの資金割合（10%）
STOP_LOSS_PCT    = -0.08             # ストップロス（-8%）
TAKE_PROFIT_PCT  =  0.20             # 利確ライン（+20%）
COMMISSION_PCT   =  0.001            # 手数料（0.1%/片道）
BUY_SCORE_MIN    = 6                 # 買いスコアの最低ライン
SELL_SCORE_MIN   = 5                 # 売りスコアの最低ライン
LOOKBACK_DAYS    = 400               # 価格データ取得期間（日）

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 1. シグナルCSVを読み込む
# =========================
def load_all_signals(results_dir: str) -> pd.DataFrame:
    """
    results/ 配下にある screen_*_all.csv を全部読み込んで結合する。
    ファイル名からas_of日付を取得してカラムに付与する。
    """
    pattern = os.path.join(results_dir, "screen_*_all.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"シグナルCSVが見つかりません: {pattern}")

    frames = []
    for fp in files:
        # ファイル名から日付を抽出: screen_2024-06-01_all.csv → 2024-06-01
        stem = Path(fp).stem  # screen_2024-06-01_all
        parts = stem.split("_")
        try:
            date_str = parts[1]  # 2024-06-01
            as_of = pd.to_datetime(date_str)
        except Exception:
            continue

        df = pd.read_csv(fp, encoding="utf-8-sig", low_memory=False)
        df["as_of"] = as_of
        frames.append(df)

    if not frames:
        raise ValueError("有効なCSVが読み込めませんでした")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["as_of"] = pd.to_datetime(all_df["as_of"])
    all_df["ticker"] = all_df["ticker"].astype(str).str.strip().str.upper()

    for col in ["score", "sell_score", "adv20_m", "close"]:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors="coerce")

    print(f"シグナルCSV: {len(files)}ファイル / 合計{len(all_df)}行")
    return all_df


# =========================
# 2. 価格データを取得する
# =========================
def fetch_prices(tickers: list, days: int = LOOKBACK_DAYS) -> dict:
    """
    yfinanceで各銘柄のOHLCVを取得してdictで返す。
    {ticker: DataFrame(Open, High, Low, Close, Volume)}
    """
    end   = datetime.today()
    start = end - timedelta(days=days)
    price_map = {}

    chunk_size = 20
    chunks = [tickers[i:i+chunk_size] for i in range(0, len(tickers), chunk_size)]

    for chunk in chunks:
        try:
            raw = yf.download(
                " ".join(chunk),
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=False,
                group_by="ticker",
            )
        except Exception as e:
            print(f"  価格取得エラー: {e}")
            continue

        for t in chunk:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if t in raw.columns.get_level_values(0):
                        df = raw[t].copy()
                    elif t in raw.columns.get_level_values(1):
                        df = raw.xs(t, axis=1, level=1).copy()
                    else:
                        continue
                else:
                    df = raw.copy()

                df = df.dropna(subset=["Open", "Close"])
                if not df.empty:
                    price_map[t] = df
            except Exception:
                continue

    print(f"価格データ取得: {len(price_map)}/{len(tickers)} 銘柄")
    return price_map


# =========================
# 3. バックテスト本体
# =========================
def run_backtest(signals_df: pd.DataFrame, price_map: dict) -> dict:
    """
    シグナルCSVと価格データをもとにバックテストを実行する。

    エントリー: 買いシグナル当日の翌営業日始値
    エグジット: ①売りシグナル翌営業日始値 ②ストップロス ③利確ライン
    """
    capital    = float(INITIAL_CAPITAL)
    positions  = {}   # {ticker: {"entry_price": float, "shares": int, "entry_date": date}}
    trade_log  = []
    equity_curve = [{"date": signals_df["as_of"].min(), "equity": capital}]

    dates = sorted(signals_df["as_of"].unique())

    for i, today in enumerate(dates):
        daily = signals_df[signals_df["as_of"] == today]

        # --- 既存ポジションの売りチェック ---
        closed_tickers = []
        for ticker, pos in list(positions.items()):
            price_df = price_map.get(ticker)
            if price_df is None:
                continue

            # 翌日の始値を取得
            future = price_df.loc[price_df.index > today]
            if future.empty:
                continue

            exit_price = float(future.iloc[0]["Open"])
            exit_date  = future.index[0]

            ret = (exit_price - pos["entry_price"]) / pos["entry_price"]

            # 売り判定: ①売りシグナル ②ストップロス ③利確
            sell_signal_row = daily[daily["ticker"] == ticker]
            has_sell_signal = (
                not sell_signal_row.empty and
                float(sell_signal_row["sell_score"].iloc[0]) >= SELL_SCORE_MIN
            )
            hit_stop   = ret <= STOP_LOSS_PCT
            hit_profit = ret >= TAKE_PROFIT_PCT

            if has_sell_signal or hit_stop or hit_profit:
                # P&L計算（手数料込み）
                proceeds = exit_price * pos["shares"] * (1 - COMMISSION_PCT)
                cost     = pos["entry_price"] * pos["shares"] * (1 + COMMISSION_PCT)
                pnl      = proceeds - cost

                capital += exit_price * pos["shares"]
                closed_tickers.append(ticker)

                if hit_stop:
                    exit_reason = "stop_loss"
                elif hit_profit:
                    exit_reason = "take_profit"
                else:
                    exit_reason = "sell_signal"

                trade_log.append({
                    "ticker":       ticker,
                    "entry_date":   pos["entry_date"],
                    "exit_date":    exit_date,
                    "entry_price":  round(pos["entry_price"], 2),
                    "exit_price":   round(exit_price, 2),
                    "shares":       pos["shares"],
                    "pnl":          round(pnl, 0),
                    "return_pct":   round(ret * 100, 2),
                    "exit_reason":  exit_reason,
                })

        for t in closed_tickers:
            del positions[t]

        # --- 新規エントリーチェック ---
        buy_candidates = daily[daily["score"].fillna(0) >= BUY_SCORE_MIN].copy()
        buy_candidates = buy_candidates.sort_values("score", ascending=False)

        for _, row in buy_candidates.iterrows():
            ticker = row["ticker"]
            if ticker in positions:
                continue
            if len(positions) >= MAX_POSITIONS:
                break

            price_df = price_map.get(ticker)
            if price_df is None:
                continue

            # 翌日始値でエントリー
            future = price_df.loc[price_df.index > today]
            if future.empty:
                continue

            entry_price = float(future.iloc[0]["Open"])
            entry_date  = future.index[0]

            # ポジションサイズ（資金の10%）
            alloc  = capital * POSITION_SIZE_PCT
            shares = int(alloc // entry_price)
            if shares <= 0:
                continue

            cost = entry_price * shares * (1 + COMMISSION_PCT)
            if cost > capital:
                continue

            capital -= entry_price * shares
            positions[ticker] = {
                "entry_price": entry_price,
                "shares":      shares,
                "entry_date":  entry_date,
            }

        # エクイティカーブ更新
        unrealized = sum(
            price_map[t].loc[price_map[t].index <= today, "Close"].iloc[-1] * pos["shares"]
            for t, pos in positions.items()
            if t in price_map and not price_map[t].loc[price_map[t].index <= today].empty
        )
        equity_curve.append({"date": today, "equity": capital + unrealized})

    trades_df = pd.DataFrame(trade_log)
    equity_df = pd.DataFrame(equity_curve).drop_duplicates("date").sort_values("date")

    return {"trades": trades_df, "equity": equity_df, "final_capital": capital}


# =========================
# 4. パフォーマンス指標の計算
# =========================
def calc_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {"error": "トレードなし"}

    wins  = trades_df[trades_df["pnl"] > 0]
    loses = trades_df[trades_df["pnl"] <= 0]

    total_pnl     = float(trades_df["pnl"].sum())
    win_rate      = len(wins) / len(trades_df) * 100
    avg_win       = float(wins["return_pct"].mean())   if not wins.empty  else 0.0
    avg_loss      = float(loses["return_pct"].mean())  if not loses.empty else 0.0
    profit_factor = (
        wins["pnl"].sum() / abs(loses["pnl"].sum())
        if not loses.empty and loses["pnl"].sum() != 0 else float("inf")
    )

    eq = equity_df["equity"].values.astype(float)
    ret_series = np.diff(eq) / eq[:-1]
    sharpe = (
        float(np.mean(ret_series) / np.std(ret_series) * np.sqrt(252))
        if ret_series.std() > 0 else 0.0
    )
    running_max = np.maximum.accumulate(eq)
    drawdown    = (eq - running_max) / running_max
    max_dd      = float(drawdown.min() * 100)

    total_return = float((eq[-1] - eq[0]) / eq[0] * 100)

    by_reason = trades_df.groupby("exit_reason")["pnl"].agg(["count","sum"]).to_dict()

    return {
        "total_trades":   len(trades_df),
        "win_rate":       round(win_rate, 1),
        "avg_win_pct":    round(avg_win, 2),
        "avg_loss_pct":   round(avg_loss, 2),
        "profit_factor":  round(profit_factor, 2),
        "sharpe":         round(sharpe, 2),
        "max_drawdown":   round(max_dd, 1),
        "total_pnl":      round(total_pnl, 0),
        "total_return":   round(total_return, 1),
        "by_exit_reason": by_reason,
    }


# =========================
# 5. HTMLレポート生成
# =========================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>バックテストレポート {{ as_of }}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body { font-family: 'Helvetica Neue', Arial, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }
  h1   { color: #38bdf8; border-bottom: 1px solid #334155; padding-bottom: 10px; }
  h2   { color: #7dd3fc; margin-top: 30px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 20px 0; }
  .card { background: #1e293b; border-radius: 10px; padding: 16px; text-align: center; border: 1px solid #334155; }
  .card .val { font-size: 2em; font-weight: bold; }
  .card .lbl { font-size: 0.85em; color: #94a3b8; margin-top: 4px; }
  .pos { color: #4ade80; } .neg { color: #f87171; } .neu { color: #facc15; }
  canvas { max-height: 320px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85em; margin-top: 12px; }
  th { background: #1e293b; padding: 8px; text-align: left; color: #94a3b8; }
  td { padding: 6px 8px; border-bottom: 1px solid #1e293b; }
  tr:hover td { background: #1e293b; }
  .win  { color: #4ade80; } .loss { color: #f87171; }
</style>
</head>
<body>
<h1>📊 バックテストレポート <span style="font-size:0.7em;color:#94a3b8">{{ as_of }}</span></h1>

<h2>サマリー</h2>
<div class="grid">
  <div class="card"><div class="val {{ ret_cls }}">{{ total_return }}%</div><div class="lbl">総リターン</div></div>
  <div class="card"><div class="val {{ pnl_cls }}">¥{{ total_pnl }}</div><div class="lbl">総損益</div></div>
  <div class="card"><div class="val neu">{{ win_rate }}%</div><div class="lbl">勝率</div></div>
  <div class="card"><div class="val neu">{{ total_trades }}回</div><div class="lbl">総トレード数</div></div>
  <div class="card"><div class="val neu">{{ profit_factor }}</div><div class="lbl">プロフィットファクター</div></div>
  <div class="card"><div class="val neu">{{ sharpe }}</div><div class="lbl">シャープレシオ</div></div>
  <div class="card"><div class="val neg">{{ max_drawdown }}%</div><div class="lbl">最大ドローダウン</div></div>
  <div class="card"><div class="val pos">{{ avg_win_pct }}%</div><div class="lbl">平均利益</div></div>
  <div class="card"><div class="val neg">{{ avg_loss_pct }}%</div><div class="lbl">平均損失</div></div>
</div>

<h2>エクイティカーブ</h2>
<canvas id="equityChart"></canvas>

<h2>月次損益</h2>
<canvas id="monthlyChart"></canvas>

<h2>トレード履歴（直近50件）</h2>
<table>
<thead><tr>
  <th>銘柄</th><th>エントリー</th><th>エグジット</th>
  <th>買値</th><th>売値</th><th>損益(円)</th><th>騰落率</th><th>理由</th>
</tr></thead>
<tbody>
{{ trade_rows }}
</tbody>
</table>

<script>
// エクイティカーブ
const eqData = {{ equity_json }};
new Chart(document.getElementById('equityChart'), {
  type: 'line',
  data: {
    labels: eqData.map(d => d.date),
    datasets: [{ label: '資産推移(円)', data: eqData.map(d => d.equity),
      borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.1)',
      fill: true, tension: 0.3, pointRadius: 0 }]
  },
  options: { plugins: { legend: { labels: { color:'#e2e8f0' } } },
    scales: { x: { ticks:{color:'#94a3b8'}, grid:{color:'#1e293b'} },
              y: { ticks:{color:'#94a3b8'}, grid:{color:'#1e293b'} } } }
});

// 月次損益
const monthly = {{ monthly_json }};
new Chart(document.getElementById('monthlyChart'), {
  type: 'bar',
  data: {
    labels: monthly.map(d => d.month),
    datasets: [{ label: '月次損益(円)', data: monthly.map(d => d.pnl),
      backgroundColor: monthly.map(d => d.pnl >= 0 ? 'rgba(74,222,128,0.7)' : 'rgba(248,113,113,0.7)') }]
  },
  options: { plugins: { legend: { labels: { color:'#e2e8f0' } } },
    scales: { x: { ticks:{color:'#94a3b8'}, grid:{color:'#1e293b'} },
              y: { ticks:{color:'#94a3b8'}, grid:{color:'#1e293b'} } } }
});
</script>
</body></html>
"""

def make_html_report(metrics: dict, trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> str:
    as_of = datetime.today().strftime("%Y-%m-%d")

    # エクイティJSON
    equity_json = equity_df.copy()
    equity_json["date"] = equity_df["date"].astype(str)
    eq_list = equity_json[["date","equity"]].to_dict(orient="records")

    # 月次損益
    if not trades_df.empty and "exit_date" in trades_df.columns:
        t = trades_df.copy()
        t["month"] = pd.to_datetime(t["exit_date"]).dt.to_period("M").astype(str)
        monthly = t.groupby("month")["pnl"].sum().reset_index()
        monthly.columns = ["month","pnl"]
        monthly_list = monthly.to_dict(orient="records")
    else:
        monthly_list = []

    # トレード行HTML
    rows_html = ""
    if trades_df.empty or "exit_date" not in trades_df.columns:
        rows_html = '<tr><td colspan="8" style="text-align:center;color:#94a3b8">トレードなし（シグナルCSVが1日分のみの場合、翌日価格と照合できないためトレードが発生しません）</td></tr>'
    for _, r in (trades_df.tail(50).sort_values("exit_date", ascending=False).iterrows() if not trades_df.empty and "exit_date" in trades_df.columns else []):
        cls = "win" if r["pnl"] > 0 else "loss"
        rows_html += (
            f'<tr class="{cls}">'
            f'<td>{r["ticker"]}</td>'
            f'<td>{str(r["entry_date"])[:10]}</td>'
            f'<td>{str(r["exit_date"])[:10]}</td>'
            f'<td>¥{r["entry_price"]:,.0f}</td>'
            f'<td>¥{r["exit_price"]:,.0f}</td>'
            f'<td>¥{r["pnl"]:,.0f}</td>'
            f'<td>{r["return_pct"]:+.2f}%</td>'
            f'<td>{r["exit_reason"]}</td>'
            f'</tr>\n'
        )

    tr  = metrics.get("total_return", 0)
    pnl = metrics.get("total_pnl", 0)

    html = HTML_TEMPLATE
    replacements = {
        "{{ as_of }}":         as_of,
        "{{ total_return }}":  str(tr),
        "{{ ret_cls }}":       "pos" if tr >= 0 else "neg",
        "{{ total_pnl }}":     f"{pnl:,.0f}",
        "{{ pnl_cls }}":       "pos" if pnl >= 0 else "neg",
        "{{ win_rate }}":      str(metrics.get("win_rate", 0)),
        "{{ total_trades }}":  str(metrics.get("total_trades", 0)),
        "{{ profit_factor }}": str(metrics.get("profit_factor", 0)),
        "{{ sharpe }}":        str(metrics.get("sharpe", 0)),
        "{{ max_drawdown }}":  str(metrics.get("max_drawdown", 0)),
        "{{ avg_win_pct }}":   str(metrics.get("avg_win_pct", 0)),
        "{{ avg_loss_pct }}":  str(metrics.get("avg_loss_pct", 0)),
        "{{ trade_rows }}":    rows_html,
        "{{ equity_json }}":   json.dumps(eq_list),
        "{{ monthly_json }}":  json.dumps(monthly_list),
    }
    for k, v in replacements.items():
        html = html.replace(k, v)

    return html


# =========================
# メイン
# =========================
if __name__ == "__main__":
    print("=== バックテスト開始 ===")

    # 1. シグナル読み込み
    signals_df = load_all_signals(RESULTS_DIR)

    # 2. 価格データ取得（買いシグナルが出た銘柄のみ）
    buy_tickers = (
        signals_df[signals_df["score"].fillna(0) >= BUY_SCORE_MIN]["ticker"]
        .unique().tolist()
    )
    print(f"対象銘柄数: {len(buy_tickers)}")
    price_map = fetch_prices(buy_tickers)

    # 3. バックテスト実行
    result    = run_backtest(signals_df, price_map)
    trades_df = result["trades"]
    equity_df = result["equity"]

    # 4. 指標計算
    metrics = calc_metrics(trades_df, equity_df)
    print("\n=== パフォーマンス ===")
    for k, v in metrics.items():
        if k != "by_exit_reason":
            print(f"  {k}: {v}")

    # 5. CSV出力
    as_of = datetime.today().strftime("%Y-%m-%d")
    trades_df.to_csv(
        os.path.join(OUTPUT_DIR, f"trades_{as_of}.csv"),
        index=False, encoding="utf-8-sig"
    )
    equity_df.to_csv(
        os.path.join(OUTPUT_DIR, f"equity_{as_of}.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 6. HTMLレポート出力
    html = make_html_report(metrics, trades_df, equity_df)
    report_path = os.path.join(OUTPUT_DIR, f"report_{as_of}.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ レポート生成完了: {report_path}")
    print(f"   総トレード: {metrics.get('total_trades')}件")
    print(f"   総リターン: {metrics.get('total_return')}%")
    print(f"   最大DD:    {metrics.get('max_drawdown')}%")
