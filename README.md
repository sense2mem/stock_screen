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

出力ファイルは次の6種類です。

- `signal_path_detail.csv`: 第1高値、調整安値、第2高値、上抜け日、50/60日リターン
- `signal_features.csv`: RSI、ADX、DI差、移動平均線、高値距離、出来高比率、逆三尊段階
- `pattern_feature_summary.csv`: 価格経路分類別の件数と50/60日成績
- `average_price_curve.csv`: 買付価格を100とした分類別平均価格推移
- `signal_path_status.csv`: 未確定・価格取得失敗の明細
- `condition_signals.csv`: シグナル日時点で条件A〜Cのいずれかに該当するシグナルと判定列

条件A〜Cは次の固定ルールです。

- 条件A: `pullback_from_60d_high_pct <= -12`
- 条件B: 条件A、かつ `ma20_distance_pct >= 1`
- 条件C: `ma20_distance_pct >= 1`、かつ `di_spread >= 3`

`condition_signals.csv` には `condition_a`、`condition_b`、`condition_c` と、該当条件を
`A|B|C` 形式で示す `condition_labels` が入ります。手動GitHub ActionsのSummaryにも、
条件別件数とシグナル日・ticker・銘柄名・主要指標の一覧を表示します。

## 条件A〜Cの初回成立バックテスト

`condition_backtest.py` は、score8シグナル日以降の日足を順番に確認し、条件A・B・Cが
それぞれ初めて成立した日を記録します。条件成立は当日終値確定後に判定し、翌営業日の始値を
エントリー価格として、5・10・20・40・50・60営業日後の終値リターンを計算します。

```bash
python condition_backtest.py \
  --signals-dir fixed_holding_input \
  --signal-pattern "screen_*_buy.csv" \
  --score-min 8 \
  --signal-mode first_in_streak \
  --output-dir signal_path_report \
  --path-detail signal_path_report/signal_path_detail.csv \
  --holdings 5 10 20 40 50 60
```

判定時点ごとに60日高値、20日移動平均、Wilder方式の+DI/-DIを再計算するため、
シグナル日後に条件が新しく成立したケースも検証できます。未来データは条件判定に使用しません。

出力ファイルは次の4種類です。

- `condition_backtest_detail.csv`: 条件初回成立日、翌営業日始値、各保有期間リターン、60日MFE/MAE
- `condition_backtest_summary.csv`: 条件別の成立件数、60日完了件数、平均・中央値・勝率
- `condition_average_price_curve.csv`: 条件別および条件組み合わせ別の平均価格曲線
- `condition_backtest_status.csv`: 未成立、価格取得失敗、60営業日未完了の明細

`condition_group` は同一成立日に点灯した条件を、`A_only`、`A_B`、`A_B_C`、`C_only`
などで表します。同じ日・同じ銘柄でAとBが同時成立した場合、条件別集計では両方に含めますが、
`all_unique_entries` と組み合わせ別平均曲線では1回のエントリーとして重複を除きます。

この処理は手動の `Signal path analysis (manual)` ワークフロー内で実行され、通常のscore8判定、
平日の本番ワークフロー、`data` ブランチの内容は変更しません。分析結果を確認するまでは、
通常のscore8判定への加点や除外には使用しません。
