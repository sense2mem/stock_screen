# stock_screen OpenClaw分析レポート

## 1. 概要

`stock_screen` は、日本株の銘柄スクリーニングを自動実行するためのリポジトリである。

主な目的は、JPXの上場銘柄リストを入力として、株価データ、テクニカル指標、簡易ファンダメンタル情報、決算予定、地合い判定を組み合わせ、買い候補・売り候補・全銘柄評価結果をCSVとして出力することにある。

GitHub Actions により、平日20:00 JSTに自動実行される構成になっており、実行結果は artifact として保存されるほか、`data` ブランチへアーカイブされる想定である。また、生成済みスクリーニングCSVを利用して簡易バックテストを行い、HTMLレポートを生成する機能も含まれている。

## 2. リポジトリ構成

確認した主な構成は以下の通り。

```text
stock_screen/
├── README.md
├── requirements.txt
├── screen_yearend.py
├── backtest.py
├── run_screen.bat
├── .gitignore
├── workflows
├── openclaw_analyze_and_commit.sh
├── .github/
│   └── workflows/
│       └── stock_screen.yml
└── docs/
    └── openclaw_analysis.md
```

分析対象からは、依頼条件に従い `.git/`、仮想環境、キャッシュ、ログ、大量CSV、大量データ、秘密情報系ファイルを除外した。

補足:

- `.github/workflows/stock_screen.yml` が実際のGitHub Actions定義である。
- `workflows` は内容のないファイルだった。
- `openclaw_analyze_and_commit.sh` はOpenClawによる分析・Git操作を補助するスクリプトだが、今回の依頼では実行していない。
- `.gitignore` では `*.csv`、`*.xls`、`*.xlsx`、`.env`、ログ、仮想環境などが除外対象になっている。

## 3. 主要ファイル

### `README.md`

リポジトリの概要を短く説明している。

内容は、日本株スクリーニングを平日20時に自動実行し、結果CSVを出力するスクリプトである、という説明に留まっている。

### `screen_yearend.py`

スクリーニング処理の中心となるファイル。

主な役割:

- JPX銘柄リストの読み込み
- yfinanceによる株価データ取得
- RSI、MACD、SMA、ATR/NATR、ADX、Hurst指数などの計算
- 買いスコア、売りスコアの算出
- Yahoo Finance系エンドポイントからPER、PBR、EPS、時価総額、決算予定を取得
- TOPIX連動ETF `1306.T` による地合い判定
- 流動性、EPS、時価総額、決算ブラックアウトによるフィルタ
- `screen_YYYY-MM-DD_all.csv`、`buy.csv`、`sell.csv` の出力

主要関数:

- `rsi()`
- `macd()`
- `hurst_exponent()`
- `calc_atr_natr()`
- `calc_adx()`
- `score_buy_signals()`
- `screen_tech()`
- `load_tickers_from_excel_bcol()`
- `fetch_per_pbr_batch()`
- `get_market_regime()`
- `apply_filters_and_make_buy()`
- `apply_filters_and_make_trades()`

### `backtest.py`

スクリーニング結果CSVを用いた簡易バックテストとHTMLレポート生成を行うファイル。

主な役割:

- `screen_*_all.csv` の読み込み
- 買いシグナル翌営業日始値でのエントリー
- 売りシグナル、損切り、利確によるエグジット
- 損益、勝率、プロフィットファクター、シャープレシオ、最大ドローダウンなどの計算
- `backtest_report/` 配下へのCSV・HTMLレポート出力

主要設定:

- 初期資金: 1,000,000円
- 最大保有銘柄数: 5
- 1銘柄あたり資金割合: 10%
- 損切り: -8%
- 利確: +20%
- 手数料: 片道0.1%
- 買いスコア閾値: 6
- 売りスコア閾値: 5

### `.github/workflows/stock_screen.yml`

GitHub Actionsの自動実行設定。

主な内容:

- 平日20:00 JSTにスケジュール実行
- 手動実行 `workflow_dispatch` に対応
- Python 3.11を使用
- `requirements.txt` の依存関係をインストール
- JPXから `data_e.xls` をダウンロード
- `screen_yearend.py` を実行
- 生成CSVをartifactとしてアップロード
- `backtest.py` を実行
- バックテストレポートをartifactとしてアップロード
- `data` ブランチへスクリーニング結果とバックテスト結果をアーカイブ

### `requirements.txt`

Python依存関係を固定している。

主な依存:

- `pandas`
- `numpy`
- `yfinance`
- `requests`
- `xlrd`
- `beautifulsoup4`
- `curl_cffi`

### `run_screen.bat`

Windowsローカル実行用のバッチファイル。

主な処理:

- 固定パス `C:\Users\kenic\Desktop\stock_screen` へ移動
- `.venv` を有効化
- `logs/` に日時付きログを保存
- `screen_yearend.py` を実行

## 4. 処理フロー

全体の処理フローは以下の通り。

```text
GitHub Actions またはローカル実行
        ↓
JPXから data_e.xls を取得またはローカル配置
        ↓
data_e.xls から銘柄コードを読み込み
        ↓
yfinanceで株価OHLCVを取得
        ↓
各銘柄のテクニカル指標を計算
        ↓
買いスコア・売りスコアを算出
        ↓
Yahoo Finance系APIからPER/PBR/EPS/時価総額/決算予定を取得
        ↓
TOPIX連動ETF 1306.T で地合い判定
        ↓
流動性・EPS・時価総額・決算ブラックアウトでフィルタ
        ↓
全件・買い候補・売り候補CSVを出力
        ↓
スクリーニングCSVを使ってバックテスト
        ↓
取引履歴・資産推移CSVとHTMLレポートを生成
        ↓
GitHub Actions artifact と data ブランチへ保存
```

## 5. 入力データ・出力データ

### 入力データ

| 種別 | 内容 | 主な参照元 |
|---|---|---|
| 銘柄一覧 | JPX上場銘柄リスト | `data_e.xls` |
| 株価データ | Open, High, Low, Close, Adj Close, Volume | yfinance |
| ファンダ情報 | PER、PBR、EPS、時価総額など | Yahoo Finance quote API |
| 決算予定 | 次回決算日 | Yahoo Finance quote API |
| 地合い判定 | TOPIX連動ETFの価格と200日移動平均 | `1306.T` |
| バックテスト入力 | 過去の `screen_*_all.csv` | `RESULTS_DIR` 配下 |

### 出力データ

`screen_yearend.py` の主な出力:

```text
screen_YYYY-MM-DD_all.csv
screen_YYYY-MM-DD_buy.csv
screen_YYYY-MM-DD_sell.csv
screen_YYYY-MM-DD_errors.csv  # エラー発生時
```

`backtest.py` の主な出力:

```text
backtest_report/trades_YYYY-MM-DD.csv
backtest_report/equity_YYYY-MM-DD.csv
backtest_report/report_YYYY-MM-DD.html
```

GitHub Actionsでは、これらをartifactとして保存し、さらに `data` ブランチの `archive/` 配下へ日付別・latest別に保存する設計になっている。

### 外部依存

- JPX公式サイトのExcelファイル
- yfinance / Yahoo Finance系データ取得
- GitHub Actions実行環境
- Python 3.11
- pip依存パッケージ

外部データ取得に依存しているため、サイト仕様変更、レート制限、通信失敗、データ欠損の影響を受ける可能性がある。

## 6. スクリーニングロジック

### 買いスコア

`screen_yearend.py` では、買い条件を8項目でスコア化している。

買いスコアの閾値:

```text
PULLBACK_SCORE_MIN = 6
```

主な買い条件:

1. 上昇トレンド
   - 株価がSMA200を上回る
   - SMA50がSMA200を上回る
2. SMA200が上向き
3. 200日移動平均からの乖離が適正
   - 0%以上15%以下
4. 押し目ゾーン
   - SMA75付近以上
   - SMA50近辺以下
5. RSIが押し目レンジ
   - 35以上55以下
6. 反発シグナル
   - SMA25上抜け、またはMACD買い風
7. 52週高値に対して一定以上の強さ
   - 52週高値の80%以上
8. ADXによる上昇トレンド確認
   - ADX 20以上
   - +DI > -DI

### ファンダメンタル・流動性フィルタ

主な閾値:

```text
MIN_EPS_TTM = 0.0
MIN_ADV20_M = 200
MIN_MKT_CAP = 30e9
```

意味:

- EPSがマイナスでないこと
- 20日平均売買代金が2億円以上であること
- 時価総額が300億円以上であること

### 地合い判定

`MARKET_TICKER = "1306.T"` をTOPIX代替として使用し、200日移動平均との関係でリスクオン/オフを判定している。

`BEAR_MODE = "tighten"` のため、地合いが悪い場合は完全停止ではなく、買い条件を追加で厳しくする設計である。

### 決算ブラックアウト

決算前後の一定期間を買い候補から除外する設計。

主な設定:

```text
EARNINGS_BLACKOUT_PRE_BDAYS = 3
EARNINGS_BLACKOUT_POST_BDAYS = 1
DROP_IF_EARNINGS_UNKNOWN = False
```

### 売りスコア

売りスコアの閾値:

```text
SELL_SCORE_MIN = 5
```

主な売り条件:

- トレンド崩れ
- SMA25を2日連続で下抜け
- MACD売り風
- RSI過熱後の反落
- 直近20日高値からのトレーリングストップ
- DI弱気化
- 下落日に出来高急増
- NATRスパイク
- 決算直前の退避

売り候補は `screen_YYYY-MM-DD_sell.csv` に出力される。

## 7. 実行方法

### GitHub Actionsでの実行

`.github/workflows/stock_screen.yml` により、以下のタイミングで実行される。

- 平日20:00 JST
- 手動実行 `workflow_dispatch`

実行内容:

1. 依存関係をインストール
2. JPXから `data_e.xls` を取得
3. `screen_yearend.py` を実行
4. CSVをartifact保存
5. `backtest.py` を実行
6. レポートをartifact保存
7. `data` ブランチへアーカイブ

### ローカル実行

Windowsでは `run_screen.bat` が用意されている。

前提:

- `C:\Users\kenic\Desktop\stock_screen` にリポジトリがあること
- `.venv` が存在し、有効化できること
- 必要に応じて `data_e.xls` が配置されていること

実行内容:

```text
python screen_yearend.py
```

ログは `logs/` 配下に保存される。

### バックテスト実行

`backtest.py` は `RESULTS_DIR` 配下の `screen_*_all.csv` を読み込む。

例:

```text
RESULTS_DIR=. python backtest.py
```

ただし、バックテストは複数日のシグナルCSVがあるほど意味を持つため、当日分のみでは十分な検証にならない可能性がある。

## 8. 現在の課題

### READMEの情報不足

READMEが非常に短く、以下の情報が不足している。

- セットアップ方法
- ローカル実行方法
- GitHub Actionsの説明
- 出力CSVの説明
- スクリーニング条件の概要
- バックテストの使い方
- 注意事項、免責

### 外部データ依存が強い

JPX、yfinance、Yahoo Finance系エンドポイントに依存している。

想定リスク:

- サイト仕様変更
- レート制限
- 通信失敗
- 銘柄ごとのデータ欠損
- 決算予定やファンダ情報の未取得

### バックテストの検証精度

GitHub Actions上では、カレントディレクトリの当日CSVだけを対象にバックテストしている可能性がある。

`backtest.py` 自体は過去の `screen_*_all.csv` をまとめて読む設計だが、過去CSVを明示的に取得してから実行する流れにはなっていない。

そのため、現状のActions上のバックテストは、履歴検証としては不十分になる可能性がある。

### バックテストの資金計算

`backtest.py` では手数料を損益計算に反映している一方、資金残高の増減では手数料込みの支出・控除後の売却代金が十分に反映されていない可能性がある。

これはエクイティカーブや最終資産の精度に影響する可能性がある。

### Actionsの依存関係インストールに重複がある

`requirements.txt` で依存関係をインストールした後、Actions内で再度 `yfinance pandas numpy` をインストールしている。

依存管理を `requirements.txt` に寄せた方がよい。

### `BEAR_MODE = "tighten"` の条件に旧ロジックの名残がある

買いスコアからは `vol_quiet` が外されているが、地合い悪化時の追加条件では `vol_quiet` が使われている。

意図した設計なら問題ないが、ロジックの整合性確認が必要である。

### ローカル実行パスが固定されている

`run_screen.bat` はWindowsの固定パスを前提としている。

別環境ではそのまま使えない。

## 9. 改善提案

### README整備

以下をREADMEに追加すると、保守性が上がる。

- 概要
- セットアップ手順
- ローカル実行手順
- GitHub Actions実行フロー
- 入力・出力ファイル一覧
- スクリーニング条件概要
- バックテスト方法
- dataブランチの運用方法
- 注意事項、免責

### バックテストの履歴データ利用

Actionsでバックテストする際に、`data` ブランチの過去アーカイブCSVを取得してから `backtest.py` に渡す構成にすると、検証の意味が大きくなる。

候補:

- `data` ブランチの `archive/YYYY-MM-DD/screen_*_all.csv` を読み込む
- バックテスト専用workflowを分離する
- 直近N日、直近Nか月など期間指定できるようにする

### 手数料を資金管理へ反映

バックテストでは、エントリー時・エグジット時の資金増減に手数料を明示的に反映した方がよい。

例:

- エントリー時は `entry_price * shares * (1 + COMMISSION_PCT)` を減算
- エグジット時は `exit_price * shares * (1 - COMMISSION_PCT)` を加算

### 外部データ取得の耐性強化

以下を追加すると運用時の安定性が上がる。

- 取得失敗率のサマリー出力
- データソース別の失敗件数
- 再取得対象リストの保存
- Yahoo/yfinance失敗時の代替方針
- レート制限を意識した待機・リトライ制御

### 依存関係整理

- `requirements.txt` とActionsの追加インストールを統一する
- 実際に不要な依存がないか確認する
- yfinanceなど外部仕様変更に弱いライブラリは、動作確認済みバージョンを明示する

### 設定値の外出し

現在はスクリーニング閾値がPythonコード内に直接定義されている。

将来的には以下のような設定ファイル化が考えられる。

- `config.yml`
- `screen_config.json`
- 環境変数

ただし、現段階ではコードが小さいため、無理に分離しなくてもよい。

## 10. OpenClawで自動化できる作業

OpenClawで自動化しやすい作業は以下である。

1. 実行ログ・エラーCSVの要約
   - 取得失敗銘柄
   - エラー種別
   - 前回比で悪化した点

2. 毎日のスクリーニング結果サマリー
   - 買い候補件数
   - 売り候補件数
   - 上位候補の理由
   - 地合い判定

3. バックテストレポートの要約
   - 勝率
   - PF
   - 最大DD
   - 直近の悪化・改善

4. READMEや運用ドキュメントの更新補助
   - 実コードとの差分確認
   - 設定値の一覧化
   - Actionsフローの説明更新

5. GitHub Actions失敗時の原因分析
   - 依存関係エラー
   - 外部データ取得エラー
   - タイムアウト
   - CSV未生成

6. スクリーニング条件の変更案作成
   - 条件変更前後の影響整理
   - 閾値変更候補の提示
   - バックテスト結果との比較

7. 定期的な品質チェック
   - 依存関係の古さ確認
   - 外部API仕様変更の兆候確認
   - 出力CSVカラムの変化検出

## 11. 次に対応すべき優先順位

優先度順に並べると以下が妥当である。

1. バックテストの資金計算を確認・修正する
   - 手数料がエクイティカーブに正しく反映されているか確認する。

2. 過去CSVを使ったバックテスト運用にする
   - 当日CSVのみでは検証として弱いため、`data` ブランチの履歴を使えるようにする。

3. READMEを整備する
   - 使い方、出力、スクリーニング条件、注意事項を明文化する。

4. 外部データ取得失敗時のログ・サマリーを強化する
   - Yahoo/yfinance/JPX依存のため、失敗の見える化が重要。

5. Actionsとrequirementsの依存関係を整理する
   - 重複インストールを減らし、再現性を上げる。

6. 地合い悪化時の `tighten` 条件を再確認する
   - `vol_quiet` を使い続ける意図があるか確認する。

7. ローカル実行手順を環境依存しにくくする
   - 固定パス前提の `run_screen.bat` を補助ドキュメントでカバーする。

## 12. 今回確認できなかった点

以下は今回の分析では確認していない、または確認できなかった。

- 実際の外部API通信結果
- JPX、Yahoo Finance、yfinanceの現在の応答状況
- 実際のGitHub Actions実行ログ
- `data` ブランチ上の過去アーカイブ内容
- 生成済みCSVの実データ内容
- 本番運用上の通知先や利用フロー
- 実際の投資成績や売買判断への利用状況
- 秘密情報、認証情報、`.env` 系ファイルの内容

今回の依頼条件に従い、本番データ、本番DB、外部API実行、Git操作、秘密情報ファイルの読み取りは行っていない。
