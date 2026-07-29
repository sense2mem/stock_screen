# stock_screen
日本株スクリーニングを平日20時(JST)に自動実行し、結果CSVを出力するスクリプト。

## score8 固定保有バックテスト

`fixed_holding_backtest.py` は、日次の `screen_YYYY-MM-DD_buy.csv` に含まれる
score8以上の候補を、シグナル翌取引日の始値で買い、5・10・20・40営業日目の
終値で売る独立したバックテストです。保有期間中（エントリー日・エグジット日を含む）
のMFE/MAEも計算します。

```bash
python fixed_holding_backtest.py \
  --signals-dir fixed_holding_input \
  --signal-pattern "screen_*_buy.csv" \
  --score-min 8 \
  --holdings 5 10 20 40 \
  --signal-mode first_in_streak \
  --output-dir fixed_holding_report/first_in_streak
```

`--signal-mode all` は全シグナルを独立評価します。`first_in_streak` は、読み込んだ
スクリーニング日の並びで同じtickerが連続した場合に先頭だけを評価します。

出力先には、取引明細 `fixed_holding_detail.csv`、保有期間別集計
`fixed_holding_summary.csv`、未確定・失敗明細 `fixed_holding_status.csv`、および
採否を含む `signals_used.csv` がUTF-8 BOM付きで作成されます。GitHub Actionsでは
両モードを実行し、artifactと `data` ブランチの
`archive/YYYY-MM-DD/fixed_holding/` および `archive/latest/fixed_holding/` に保存します。
