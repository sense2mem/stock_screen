# stock_screen
日本株スクリーニングを平日20時(JST)に自動実行し、結果CSVを出力するスクリプト。

## score8 固定保有バックテスト

`fixed_holding_backtest.py` は、日次の `screen_YYYY-MM-DD_buy.csv` に含まれる
score8以上の候補を、シグナル翌取引日の始値で買い、5・10・20・40・50・60営業日目の
終値で売る独立したバックテストです。保有期間中（エントリー日・エグジット日を含む）
のMFE/MAEも計算します。

```bash
python fixed_holding_backtest.py \
  --signals-dir fixed_holding_input \
  --signal-pattern "screen_*_buy.csv" \
  --score-min 8 \
  --holdings 5 10 20 40 50 60 \
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

## score8 シグナル後の価格経路分析

`signal_path_analysis.py` は、score8以上のシグナルについて、シグナル翌営業日の始値を100として
60営業日の価格経路を分析する独立ツールです。既存のスクリーニング条件や売買ロジックは変更しません。

主な価格経路分類は次のとおりです。

- `DOUBLE_RISE_SUCCESS`: 第1上昇後に3〜15%調整し、第1高値を終値で1%以上上抜け
- `SECOND_RISE_FAILED`: 調整後に再上昇したが第1高値を上抜けられない
- `CONTINUOUS_RISE`: 3%以上の調整を伴わず上昇継続
- `FIRST_RISE_ONLY`: 第1上昇後の下落が15%を超える
- `NO_FIRST_RISE`: 最初の15営業日で3%以上上昇しない
- `IN_PROGRESS`: 60営業日が未経過

シグナル日以前の株価だけを使い、逆三尊に似た形成段階も次のタグで出力します。

- `HEAD_TO_NECKLINE`
- `RIGHT_SHOULDER`
- `BREAKOUT_APPROACH`
- `POST_BREAKOUT`
- `NONE`

```bash
python signal_path_analysis.py \
  --signals-dir fixed_holding_input \
  --score-min 8 \
  --signal-mode first_in_streak \
  --output-dir signal_path_report
```

出力ファイルは次の5種類です。

- `signal_path_detail.csv`: 第1高値、調整安値、第2高値、上抜け日、50/60日リターン
- `signal_features.csv`: RSI、ADX、DI差、移動平均線、高値距離、出来高比率、逆三尊段階
- `pattern_feature_summary.csv`: 価格経路分類別の件数と50/60日成績
- `average_price_curve.csv`: 買付価格を100とした分類別平均価格推移
- `signal_path_status.csv`: 未確定・価格取得失敗の明細

初期判定値はコマンドライン引数で変更できます。分析結果を確認するまでは、通常のscore8判定への加点や除外には使用しません。
