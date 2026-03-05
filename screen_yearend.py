# -*- coding: utf-8 -*-
# ローカルPCで実行（ネット接続あり想定）
# pip install yfinance pandas numpy requests
# ※ data_e.xls を読むなら環境により xlrd が必要です:
# pip install xlrd==2.0.1

# =============================================================================
# 主な修正点サマリー
# =============================================================================
# [FIX-01] cond_not_near_breakout を廃止 → cond_near_52w_high（強い株を選ぶ）に変更
# [FIX-02] vol_quiet を買い条件から除外 → cond_adx_bull（ADX≥20 かつ +DI>-DI）に変更
# [FIX-03] PULLBACK_SCORE_MIN を 7→6 に緩和
# [FIX-04] cond_pullback_zone の上限を SMA25→SMA50 に変更（cond_rebound との論理矛盾を解消）
# [FIX-05] Hurst指数・ADX を買いスコアに活用
# [FIX-06] sell_cross_sma25_down を「2日連続下抜け」に強化（ノイズ低減）
# [FIX-07] RSI_OVERBOUGHT を 70→75 に引き上げ（早期利確を抑制）
# [FIX-08] SELL_SCORE_MIN を 4→5 に引き上げ（ノイズ売りを抑制）
# [FIX-09] sell_natr_spike で close_adj（調整済み終値）を使用
# [FIX-10] get_market_regime で auto_adjust=True かつ Close で一本化
# [FIX-11] ADV 計算を Close（非調整）× Volume に変更（実際の取引代金に近づける）
# [FIX-12] 週足変数 w（デッドコード）を削除
# [FIX-13] sell_score のコメントを 8点満点に整合させる（ext_dev200 を除外を明示）
# [FIX-14] エントリー・出口は「翌日始値」を想定するコメントを追加
# [FIX-15] merge 時に suffixes を明示してカラム衝突リスクを排除
# [FIX-16] _ts_to_date の返り値を pd.Timestamp に統一
# =============================================================================

import time
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================
# 設定
# =========================
LOOKBACK_DAYS = 900
TZ = "Asia/Tokyo"

MARKET_TICKER = "1306.T"     # TOPIX連動ETF（代用）
MARKET_MA_DAYS = 200

# 【最優先】ファンダ地雷除去
MIN_EPS_TTM = 0.0
MIN_ADV20_M = 200            # 百万円（=2億円/日）
MIN_MKT_CAP = 30e9           # 300億円

# 【環境認識】地合いが悪い時の動作
BEAR_MODE = "tighten"        # "ignore" or "tighten"

# Pullbackスコア（0〜8点）
# [FIX-03] 7→6 に緩和（シグナル数を現実的に）
PULLBACK_SCORE_MIN = 6

# Yahoo quote (PER/PBR) 取得用
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
YAHOO_CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
YAHOO_COOKIE_WARMUP = "https://fc.yahoo.com"

_regime_cache = {"as_of": None, "risk_on": True, "close": None, "ma200": None}

# =========================
# SELL（出口）シグナル設定
# =========================
# [FIX-08] 4→5 に引き上げ（ノイズ売りシグナルを抑制）
SELL_SCORE_MIN = 5
TRAIL_STOP_20D_PCT = 0.08    # 直近20日高値から -8% でトレーリングストップ
EXT_DEV200_MAX = 0.18        # 200MA乖離 +18% 以上は利確優先（スコアには含めず理由列のみ）
# [FIX-07] RSI閾値を 70→75 に引き上げ（早期利確を抑制）
RSI_OVERBOUGHT = 75
RSI_EXTREME = 82
ADX_SELL_MIN = 20
NATR_SPIKE_MULT = 1.4        # NATRが60日平均の1.4倍超なら"荒れ"
SELL_EXIT_BEFORE_EARNINGS = True
EARNINGS_EXIT_PRE_BDAYS = 1  # 決算の何営業日前までに逃げるか

# 決算ブラックアウト
EARNINGS_BLACKOUT_PRE_BDAYS = 3
EARNINGS_BLACKOUT_POST_BDAYS = 1
DROP_IF_EARNINGS_UNKNOWN = False

# =========================
# 指標計算
# =========================
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    m = ema_fast - ema_slow
    s = m.ewm(span=signal, adjust=False).mean()
    h = m - s
    return m, s, h

def hurst_exponent(price, max_lag=50, eps=1e-12):
    s = price.dropna()
    if len(s) < max_lag + 5:
        return np.nan
    ts = np.log(s.values)
    xs, ys = [], []
    for lag in range(2, max_lag):
        diff = ts[lag:] - ts[:-lag]
        t = np.std(diff)
        if np.isfinite(t) and t > eps:
            xs.append(lag)
            ys.append(t)
    if len(xs) < 6:
        return np.nan
    poly = np.polyfit(np.log(xs), np.log(ys), 1)
    return poly[0]

def _rma(series: pd.Series, n: int) -> pd.Series:
    """Wilder's RMA (EMA alpha=1/n)"""
    return series.ewm(alpha=1.0/n, adjust=False).mean()

def calc_atr_natr(df: pd.DataFrame, period: int = 14):
    if df is None or df.empty:
        return (np.nan, np.nan)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = _rma(tr, period)
    atr_last = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else np.nan
    c_last   = float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else np.nan
    natr_last = (atr_last / c_last * 100.0) if (np.isfinite(atr_last) and np.isfinite(c_last) and c_last != 0) else np.nan
    return (atr_last, natr_last)

def calc_pullback_from_high(close_adj: pd.Series, window: int = 20):
    if close_adj is None or close_adj.dropna().empty or len(close_adj.dropna()) < window:
        return np.nan
    hh = close_adj.rolling(window).max().iloc[-1]
    if pd.isna(hh) or hh == 0:
        return np.nan
    return float(close_adj.iloc[-1] / hh - 1.0)

def calc_adx(df: pd.DataFrame, period: int = 14):
    if df is None or df.empty or len(df) < period + 2:
        return (np.nan, np.nan, np.nan)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    close = df["Close"].astype(float)
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr      = _rma(tr, period)
    plus_di  = 100.0 * (_rma(plus_dm,  period) / atr.replace(0, np.nan))
    minus_di = 100.0 * (_rma(minus_dm, period) / atr.replace(0, np.nan))
    dx  = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _rma(dx, period)
    adx_last = float(adx.iloc[-1])      if pd.notna(adx.iloc[-1])      else np.nan
    pdi_last = float(plus_di.iloc[-1])  if pd.notna(plus_di.iloc[-1])  else np.nan
    mdi_last = float(minus_di.iloc[-1]) if pd.notna(minus_di.iloc[-1]) else np.nan
    return (adx_last, pdi_last, mdi_last)


def score_buy_signals(df_daily):
    """
    買いスコア（score）と売りスコア（sell_score）を計算して返す。

    約定タイミングの考え方：
      - シグナルは当日の「終値確定後」に発生
      - 実際のエントリー・エグジットは「翌営業日の始値」を想定すること  [FIX-14]
        （バックテスト時は open.shift(-1) で評価すること）
    """
    # [FIX-11] ADV計算用に非調整Close も保持
    close_raw = df_daily["Close"].astype(float)
    close_adj = df_daily["Adj Close"] if "Adj Close" in df_daily.columns else close_raw
    vol = df_daily["Volume"]

    MIN_DAYS = 260
    if len(close_adj.dropna()) < MIN_DAYS:
        last_close = float(close_adj.iloc[-1]) if len(close_adj) else np.nan
        return {
            "score": 0,
            "gc_multi": False, "rsi": np.nan, "macd_buy_like": False,
            "vol_surge": False, "vol_quiet": False, "vol_down": False,
            "hurst": np.nan, "close": last_close, "close_adj": last_close,
            "adv20": np.nan, "adv20_m": np.nan,
            "hh252": np.nan, "pct_of_52w_high": np.nan, "max_dd_252": np.nan,
            "dev200": np.nan, "atr14": np.nan, "natr14": np.nan,
            "pullback_20d": np.nan, "pullback_60d": np.nan,
            "adx14": np.nan, "plus_di14": np.nan, "minus_di14": np.nan,
            "skip_reason": "insufficient_daily",
            "sell_score": 0, "sell_reason": "insufficient_daily",
            "sell_trend_break": False, "sell_cross_sma25_down": False,
            "sell_macd_sell_like": False, "sell_rsi_reversal": False,
            "sell_trailing_stop": False, "sell_ext_dev200": False,
            "sell_di_bear": False, "sell_vol_distribution": False,
            "sell_natr_spike": False,
        }

    # =====================
    # 共通指標
    # =====================
    sma25  = close_adj.rolling(25).mean()
    sma50  = close_adj.rolling(50).mean()
    sma75  = close_adj.rolling(75).mean()
    sma200 = close_adj.rolling(200).mean()

    r = rsi(close_adj, 14)
    m, s, h = macd(close_adj)

    h2, m2, s2 = h.dropna(), m.dropna(), s.dropna()
    macd_buy = False
    if len(h2) >= 2 and len(m2) >= 1 and len(s2) >= 1:
        macd_buy = bool((h2.iloc[-1] > h2.iloc[-2]) and (m2.iloc[-1] > s2.iloc[-1]))

    ret    = close_adj.pct_change()
    vol20v = ret.rolling(20).std()
    vol_down = bool(vol20v.iloc[-1] < vol20v.rolling(60).mean().iloc[-1]) if pd.notna(vol20v.iloc[-1]) else False

    vol20_mean = vol.rolling(20).mean().iloc[-1]
    vol_surge  = bool(vol.iloc[-1] > vol20_mean * 1.5) if pd.notna(vol20_mean) else False
    vol_quiet  = bool(vol.iloc[-1] <= vol20_mean * 1.2) if pd.notna(vol20_mean) else False

    H = hurst_exponent(close_adj.tail(250))
    H = float(np.clip(H, 0.0, 1.0)) if np.isfinite(H) else np.nan

    # [FIX-11] ADV は非調整終値 × Volume（実際の取引代金に近い値）
    adv20   = (close_raw * vol).rolling(20).mean().iloc[-1]
    adv20_m = adv20 / 1e6 if pd.notna(adv20) else np.nan

    hh252         = close_adj.rolling(252).max().iloc[-1]
    pct_of_52w_high = (close_adj.iloc[-1] / hh252) if (pd.notna(hh252) and hh252 != 0) else np.nan

    roll_max  = close_adj.rolling(252).max()
    dd        = close_adj / roll_max - 1.0
    max_dd_252 = dd.iloc[-252:].min() if len(dd.dropna()) >= 252 else np.nan

    # ATR/NATR・Pullback・ADX は買いスコアにも使う [FIX-05]
    atr14, natr14       = calc_atr_natr(df_daily, period=14)
    pullback_20d        = calc_pullback_from_high(close_adj, window=20)
    pullback_60d        = calc_pullback_from_high(close_adj, window=60)
    adx14, plus_di14, minus_di14 = calc_adx(df_daily, period=14)

    # =====================
    # BUY スコア（8条件）
    # =====================

    # 条件1: 上昇トレンド（SMA50 > SMA200 かつ 株価 > SMA200）
    cond_trend = bool(
        pd.notna(sma200.iloc[-1]) and pd.notna(sma50.iloc[-1]) and
        (close_adj.iloc[-1] > sma200.iloc[-1]) and
        (sma50.iloc[-1] > sma200.iloc[-1])
    )

    # 条件2: SMA200 が上向き
    cond_sma200_up = False
    if len(close_adj) >= 220 and pd.notna(sma200.iloc[-1]) and pd.notna(sma200.iloc[-20]):
        cond_sma200_up = bool(sma200.iloc[-1] > sma200.iloc[-20])

    # 条件3: 200MA乖離が 0〜15%（過熱でなく、かつ水面上）
    dev200 = np.nan
    cond_dev200_ok = False
    if pd.notna(sma200.iloc[-1]) and sma200.iloc[-1] != 0:
        dev200 = float(close_adj.iloc[-1] / sma200.iloc[-1] - 1.0)
        cond_dev200_ok = bool(0.0 <= dev200 <= 0.15)   # 上限を 12%→15% に緩和

    # 条件4: 押し目ゾーン（SMA75 付近〜SMA50 以下）
    # [FIX-04] 上限を SMA25→SMA50 に変更（cond_rebound との矛盾を解消）
    cond_pullback_zone = False
    if pd.notna(sma75.iloc[-1]) and pd.notna(sma50.iloc[-1]) and sma75.iloc[-1] > 0:
        cond_pullback_zone = bool(
            (close_adj.iloc[-1] >= sma75.iloc[-1] * 0.97) and
            (close_adj.iloc[-1] <= sma50.iloc[-1] * 1.02)  # ← SMA25 から SMA50 に変更
        )

    # 条件5: RSI が押し目レンジ（35〜55）
    cond_rsi_pullback = bool(35 <= float(r.iloc[-1]) <= 55) if pd.notna(r.iloc[-1]) else False

    # 条件6: 反発シグナル（SMA25上抜け or MACD買い）
    cond_cross_sma25_today = False
    if len(close_adj.dropna()) >= 2 and pd.notna(sma25.iloc[-2]) and pd.notna(sma25.iloc[-1]):
        cond_cross_sma25_today = bool(
            (close_adj.iloc[-2] <= sma25.iloc[-2]) and
            (close_adj.iloc[-1] >  sma25.iloc[-1])
        )
    cond_rebound = bool(cond_cross_sma25_today or macd_buy)

    # 条件7: 52週高値の 80% 以上（強い株を選ぶ）
    # [FIX-01] <= 0.91 を廃止 → >= 0.80 に反転（モメンタム重視）
    cond_near_52w_high = bool(pct_of_52w_high >= 0.80) if pd.notna(pct_of_52w_high) else False

    # 条件8: ADX≥20 かつ +DI > -DI（上昇トレンドの強さ確認）
    # [FIX-02] vol_quiet を廃止 → トレンド強度で代替
    # [FIX-05] ADX を買いスコアに組み込み
    cond_adx_bull = False
    if np.isfinite(adx14) and np.isfinite(plus_di14) and np.isfinite(minus_di14):
        cond_adx_bull = bool((adx14 >= 20) and (plus_di14 > minus_di14))

    buy_conds = [
        cond_trend,          # 1. SMA整列
        cond_sma200_up,      # 2. SMA200上向き
        cond_dev200_ok,      # 3. 200MA乖離適正
        cond_pullback_zone,  # 4. 押し目ゾーン
        cond_rsi_pullback,   # 5. RSI押し目
        cond_rebound,        # 6. 反発シグナル
        cond_near_52w_high,  # 7. 52週高値付近の強い株 [FIX-01]
        cond_adx_bull,       # 8. ADX上昇トレンド確認 [FIX-02][FIX-05]
    ]
    score = int(sum(buy_conds))

    # =====================
    # SELL スコア（8条件）
    # =====================
    # ※ ext_dev200 は理由列のみに記録し、スコアには含めない [FIX-13]

    # 1) トレンド崩れ
    sell_trend_break = False
    if pd.notna(sma200.iloc[-1]) and pd.notna(sma50.iloc[-1]):
        sell_trend_break = bool(
            (close_adj.iloc[-1] < sma200.iloc[-1]) or
            (sma50.iloc[-1] < sma200.iloc[-1])
        )

    # 2) SMA25 を 2日連続で下抜け（1日だけはノイズとして無視）
    # [FIX-06] 1日→2日連続に強化
    sell_cross_sma25_down = False
    if len(close_adj.dropna()) >= 3 and pd.notna(sma25.iloc[-3]) and pd.notna(sma25.iloc[-2]) and pd.notna(sma25.iloc[-1]):
        sell_cross_sma25_down = bool(
            (close_adj.iloc[-3] >= sma25.iloc[-3]) and  # 3日前は上
            (close_adj.iloc[-2] <  sma25.iloc[-2]) and  # 2日前から下
            (close_adj.iloc[-1] <  sma25.iloc[-1])      # 昨日も下（2日連続確認）
        )

    # 3) MACD sell-like
    sell_macd_sell_like = False
    if len(h2) >= 2 and len(m2) >= 1 and len(s2) >= 1:
        sell_macd_sell_like = bool((h2.iloc[-1] < h2.iloc[-2]) and (m2.iloc[-1] < s2.iloc[-1]))

    # 4) RSI 過熱→失速 or 極端
    # [FIX-07] RSI_OVERBOUGHT を 70→75 に引き上げ
    sell_rsi_reversal = False
    if len(r.dropna()) >= 2:
        r1 = float(r.iloc[-1]) if pd.notna(r.iloc[-1]) else np.nan
        r0 = float(r.iloc[-2]) if pd.notna(r.iloc[-2]) else np.nan
        if np.isfinite(r0) and np.isfinite(r1):
            sell_rsi_reversal = bool(
                (r0 >= RSI_OVERBOUGHT and r1 < RSI_OVERBOUGHT) or
                (r1 >= RSI_EXTREME)
            )

    # 5) トレーリング（直近20日高値から -8%）
    sell_trailing_stop = bool(np.isfinite(pullback_20d) and (pullback_20d <= -TRAIL_STOP_20D_PCT))

    # 6) 200MA乖離が過熱（利確）← スコアには入れない、理由列のみ [FIX-13]
    sell_ext_dev200 = bool(np.isfinite(dev200) and (dev200 >= EXT_DEV200_MAX))

    # 7) DIが弱気（-DI優位）かつ ADX≥20
    sell_di_bear = False
    if np.isfinite(adx14) and np.isfinite(plus_di14) and np.isfinite(minus_di14):
        sell_di_bear = bool((minus_di14 > plus_di14) and (adx14 >= ADX_SELL_MIN))

    # 8) 下落日に出来高急増（分配/売り浴びせ）
    sell_vol_distribution = False
    if len(close_adj.dropna()) >= 2:
        sell_vol_distribution = bool((close_adj.iloc[-1] < close_adj.iloc[-2]) and vol_surge)

    # 9) NATRスパイク（60日平均比）
    # [FIX-09] close_adj を使用
    sell_natr_spike = False
    try:
        high_s  = df_daily["High"].astype(float)
        low_s   = df_daily["Low"].astype(float)
        prev_c  = close_adj.shift(1)
        tr_s = pd.concat([
            (high_s - low_s).abs(),
            (high_s - prev_c).abs(),
            (low_s  - prev_c).abs(),
        ], axis=1).max(axis=1)
        atr_series  = _rma(tr_s, 14)
        natr_series = (atr_series / close_adj.replace(0, np.nan)) * 100.0  # close_adj に修正
        natr60 = natr_series.rolling(60).mean().iloc[-1]
        if np.isfinite(natr14) and np.isfinite(natr60) and natr60 > 0:
            sell_natr_spike = bool(natr14 > natr60 * NATR_SPIKE_MULT)
    except Exception:
        sell_natr_spike = False

    sell_flags = {
        "trend_break":       sell_trend_break,
        "cross_sma25_down":  sell_cross_sma25_down,
        "macd_sell_like":    sell_macd_sell_like,
        "rsi_reversal":      sell_rsi_reversal,
        "trailing_stop":     sell_trailing_stop,
        "ext_dev200":        sell_ext_dev200,      # 理由列のみ（スコア外）[FIX-13]
        "di_bear":           sell_di_bear,
        "vol_distribution":  sell_vol_distribution,
        "natr_spike":        sell_natr_spike,
    }

    # 8点満点（ext_dev200 はスコア外） [FIX-13]
    sell_conds_for_score = [
        sell_trend_break,
        sell_cross_sma25_down,
        sell_macd_sell_like,
        sell_rsi_reversal,
        sell_trailing_stop,
        sell_di_bear,
        sell_vol_distribution,
        sell_natr_spike,
    ]
    sell_score  = int(sum(sell_conds_for_score))
    sell_reason = ";".join([k for k, v in sell_flags.items() if v])

    return {
        # ---- buy互換 ----
        "score": score,
        "gc_multi": cond_trend,
        "rsi": float(r.iloc[-1]) if pd.notna(r.iloc[-1]) else np.nan,
        "macd_buy_like": bool(macd_buy),
        "vol_surge": bool(vol_surge),
        "vol_quiet": bool(vol_quiet),
        "vol_down":  bool(vol_down),
        "hurst": float(H) if np.isfinite(H) else np.nan,
        "close":     float(close_adj.iloc[-1]),
        "close_adj": float(close_adj.iloc[-1]),
        "adv20":   float(adv20)   if pd.notna(adv20)   else np.nan,
        "adv20_m": float(adv20_m) if pd.notna(adv20_m) else np.nan,
        "hh252":           float(hh252)           if pd.notna(hh252)           else np.nan,
        "pct_of_52w_high": float(pct_of_52w_high) if pd.notna(pct_of_52w_high) else np.nan,
        "max_dd_252":      float(max_dd_252)       if pd.notna(max_dd_252)       else np.nan,
        "dev200":    float(dev200)    if np.isfinite(dev200)    else np.nan,
        "atr14":     float(atr14)     if np.isfinite(atr14)     else np.nan,
        "natr14":    float(natr14)    if np.isfinite(natr14)    else np.nan,
        "pullback_20d": float(pullback_20d) if np.isfinite(pullback_20d) else np.nan,
        "pullback_60d": float(pullback_60d) if np.isfinite(pullback_60d) else np.nan,
        "adx14":      float(adx14)      if np.isfinite(adx14)      else np.nan,
        "plus_di14":  float(plus_di14)  if np.isfinite(plus_di14)  else np.nan,
        "minus_di14": float(minus_di14) if np.isfinite(minus_di14) else np.nan,
        "skip_reason": "",
        # ---- sell追加 ----
        "sell_score":          sell_score,
        "sell_reason":         sell_reason,
        "sell_trend_break":      bool(sell_trend_break),
        "sell_cross_sma25_down": bool(sell_cross_sma25_down),
        "sell_macd_sell_like":   bool(sell_macd_sell_like),
        "sell_rsi_reversal":     bool(sell_rsi_reversal),
        "sell_trailing_stop":    bool(sell_trailing_stop),
        "sell_ext_dev200":       bool(sell_ext_dev200),
        "sell_di_bear":          bool(sell_di_bear),
        "sell_vol_distribution": bool(sell_vol_distribution),
        "sell_natr_spike":       bool(sell_natr_spike),
    }


def _normalize_ohlcv(df, ticker: str):
    if df is None or df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)
        if ticker in lv0:
            df = df[ticker].copy()
        elif ticker in lv1:
            df = df.xs(ticker, axis=1, level=1).copy()
        else:
            df.columns = df.columns.get_level_values(-1)
    rename = {}
    for c in df.columns:
        if isinstance(c, str):
            cc = c.strip()
            if cc.lower() == "adj close":
                rename[c] = "Adj Close"
            else:
                rename[c] = cc[:1].upper() + cc[1:]
    if rename:
        df = df.rename(columns=rename)
    return df


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


# =========================
# テクニカルスクリーニング（yfinanceで一括DL）
# =========================
def screen_tech(tickers, chunk_size=20, pause_sec=0.2, max_retry=2):
    rows   = []
    errors = []

    end   = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)

    total        = len(tickers)
    total_chunks = (total + chunk_size - 1) // chunk_size
    print(f"tickers={total}, chunks={total_chunks}, range={start.date()}..{end.date()}", flush=True)

    for ci, chunk in enumerate(_chunks(tickers, chunk_size), start=1):
        print(f"[{ci}/{total_chunks}] downloading {len(chunk)} tickers...", flush=True)

        df_all   = None
        last_err = None

        for attempt in range(max_retry + 1):
            try:
                df_all = yf.download(
                    tickers=" ".join(chunk),
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    group_by="ticker",
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"  retry {attempt+1}/{max_retry+1}: {repr(e)}", flush=True)
                time.sleep(1.0 + attempt)

        if df_all is None or isinstance(last_err, Exception):
            for t in chunk:
                errors.append((t, f"download_failed: {repr(last_err)}"))
            time.sleep(pause_sec)
            continue

        print("  downloaded. processing...", flush=True)

        for t in chunk:
            try:
                if isinstance(df_all.columns, pd.MultiIndex):
                    if t in df_all.columns.get_level_values(0):
                        d = df_all[t].copy()
                    elif t in df_all.columns.get_level_values(1):
                        d = df_all.xs(t, axis=1, level=1).copy()
                    else:
                        errors.append((t, "ticker_not_found_in_chunk"))
                        continue
                else:
                    d = df_all.copy()

                d = _normalize_ohlcv(d, t)
                if d is None or d.empty:
                    errors.append((t, "no_data"))
                    continue

                needed = {"Open", "High", "Low", "Close", "Volume"}
                if not needed.issubset(set(d.columns)):
                    errors.append((t, f"missing_ohlcv: {list(d.columns)}"))
                    continue

                d = d.dropna(subset=["Open","High","Low","Close","Volume"])
                d = d.loc[d.index <= end]

                # [FIX-12] 週足変数 w（デッドコード）を削除

                meta = score_buy_signals(d)
                meta["ticker"] = t
                rows.append(meta)

            except Exception as e:
                errors.append((t, repr(e)))

        print(f"  done. rows={len(rows)} errors={len(errors)}", flush=True)
        time.sleep(pause_sec)

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=[
            "ticker",
            "score","gc_multi","rsi","macd_buy_like",
            "vol_surge","vol_quiet","vol_down","hurst",
            "close","close_adj","adv20","adv20_m",
            "hh252","pct_of_52w_high","max_dd_252","dev200",
            "atr14","natr14","pullback_20d","pullback_60d",
            "adx14","plus_di14","minus_di14",
            "skip_reason",
            "sell_score","sell_reason",
            "sell_trend_break","sell_cross_sma25_down","sell_macd_sell_like",
            "sell_rsi_reversal","sell_trailing_stop","sell_ext_dev200",
            "sell_di_bear","sell_vol_distribution","sell_natr_spike",
        ])

    if errors:
        as_of_str = pd.to_datetime(end).strftime("%Y-%m-%d")
        pd.DataFrame(errors, columns=["ticker","error"]).to_csv(
            f"screen_{as_of_str}_errors.csv", index=False, encoding="utf-8-sig"
        )

    return out, end


# =========================
# 銘柄一覧読み込み
# =========================
def load_tse_tickers_from_jpx_xls(xls_path: str, domestic_common_only: bool = True):
    df = pd.read_excel(xls_path)
    code_col = "コード" if "コード" in df.columns else "Code"
    mkt_col  = "市場・商品区分" if "市場・商品区分" in df.columns else "Market / Product category"
    d = df.copy()
    if domestic_common_only and mkt_col in d.columns:
        allow = [
            "プライム（内国株式）",
            "スタンダード（内国株式）",
            "グロース（内国株式）",
        ]
        d = d[d[mkt_col].isin(allow)]
    codes = (
        d[code_col].dropna().astype(int).astype(str).str.zfill(4).unique().tolist()
    )
    return [f"{c}.T" for c in codes]


def load_tickers_from_excel_bcol(xls_path: str, sheet_name: str = "Sheet1", add_suffix: str = ".T"):
    s = pd.read_excel(xls_path, sheet_name=sheet_name, usecols="B", header=None).iloc[:, 0]
    codes = (
        s.astype(str).str.strip().str.extract(r"(\d{4})", expand=False).dropna().unique().tolist()
    )
    return [f"{c}{add_suffix}" for c in codes]


# =========================
# Yahoo (PER/PBR) まとめ取得
# =========================
def _make_session():
    s = requests.Session()
    retry_kwargs = dict(
        total=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
    )
    try:
        retry = Retry(**retry_kwargs, allowed_methods=("GET",))
    except TypeError:
        retry = Retry(**retry_kwargs, method_whitelist=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def _get_cookie_and_crumb(session, headers, timeout_sec=20):
    session.get(YAHOO_COOKIE_WARMUP, headers=headers, timeout=timeout_sec)
    r = session.get(YAHOO_CRUMB_URL, headers=headers, timeout=timeout_sec)
    crumb = (r.text or "").strip()
    if (not crumb) or ("<html" in crumb.lower()):
        raise RuntimeError("Failed to obtain Yahoo crumb")
    return crumb

def fetch_per_pbr_batch(symbols, batch_size=100, pause_sec=0.2, timeout_sec=20):
    cols = ["ticker","skip_reason","mkt_price","pe_ttm","pbr","eps_ttm","bps","market_cap",
            "earn_ts","earn_start_ts","earn_end_ts"]
    if not symbols:
        return pd.DataFrame(columns=cols)

    symbols = [str(x).strip().upper() for x in symbols]
    s = _make_session()
    headers = {
        "User-Agent":      "Mozilla/5.0",
        "Accept":          "application/json,text/plain,*/*",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer":         "https://finance.yahoo.com/",
    }

    try:
        crumb = _get_cookie_and_crumb(s, headers, timeout_sec=timeout_sec)
    except Exception as e:
        return pd.DataFrame([{
            "ticker": sym,
            "skip_reason": f"crumb_error:{type(e).__name__}:{str(e)}"[:160],
            "mkt_price": None, "pe_ttm": None, "pbr": None,
            "eps_ttm": None, "bps": None, "market_cap": None,
            "earn_ts": None, "earn_start_ts": None, "earn_end_ts": None,
        } for sym in symbols], columns=cols)

    out_rows = []
    for i in range(0, len(symbols), batch_size):
        batch  = symbols[i:i+batch_size]
        params = {"symbols": ",".join(batch), "crumb": crumb}

        batch_error = None
        results     = []
        try:
            r = s.get(YAHOO_QUOTE_URL, params=params, headers=headers, timeout=timeout_sec)
            if r.status_code == 401:
                crumb = _get_cookie_and_crumb(s, headers, timeout_sec=timeout_sec)
                params["crumb"] = crumb
                r = s.get(YAHOO_QUOTE_URL, params=params, headers=headers, timeout=timeout_sec)
            if r.status_code != 200:
                batch_error = f"http_{r.status_code}"
            data    = r.json()
            results = data.get("quoteResponse", {}).get("result", []) or []
        except Exception as e:
            batch_error = f"exception_{type(e).__name__}:{str(e)}"[:160]
            results = []

        by_symbol = {str(q.get("symbol","")).upper(): q for q in results if q.get("symbol")}

        for sym in batch:
            q = by_symbol.get(sym)
            if q is None:
                out_rows.append({
                    "ticker": sym, "skip_reason": batch_error or "no_quote_result",
                    "mkt_price": None, "pe_ttm": None, "pbr": None,
                    "eps_ttm": None, "bps": None, "market_cap": None,
                    "earn_ts": None, "earn_start_ts": None, "earn_end_ts": None,
                })
                continue

            price      = q.get("regularMarketPrice")
            pe         = q.get("trailingPE")
            pb         = q.get("priceToBook")
            eps        = q.get("epsTrailingTwelveMonths") or q.get("trailingEps")
            bps        = q.get("bookValue")
            mcap       = q.get("marketCap")
            earn_ts    = q.get("earningsTimestamp")
            earn_start = q.get("earningsTimestampStart")
            earn_end   = q.get("earningsTimestampEnd")

            if (pe is None) and (price is not None) and (eps not in (None,0,0.0)):
                pe = price / eps
            if (pb is None) and (price is not None) and (bps not in (None,0,0.0)):
                pb = price / bps

            out_rows.append({
                "ticker": sym, "skip_reason": batch_error,
                "mkt_price": price, "pe_ttm": pe, "pbr": pb,
                "eps_ttm": eps, "bps": bps, "market_cap": mcap,
                "earn_ts": earn_ts, "earn_start_ts": earn_start, "earn_end_ts": earn_end,
            })

        time.sleep(pause_sec)

    return pd.DataFrame(out_rows, columns=cols).drop_duplicates("ticker", keep="first")


# =========================
# 地合い（Market Regime）
# =========================
def get_market_regime(as_of=None):
    # [FIX-10] auto_adjust=True で取得し Close を使う（Adj Close と一致するため一本化）
    global _regime_cache
    if _regime_cache["as_of"] == as_of and _regime_cache["close"] is not None:
        return _regime_cache

    try:
        df = yf.download(
            MARKET_TICKER,
            period="2y",
            interval="1d",
            auto_adjust=True,   # ← False→True に変更
            progress=False
        )
        if df is None or df.empty:
            raise ValueError("market df empty")

        c     = df["Close"].dropna()
        ma200 = c.rolling(MARKET_MA_DAYS).mean()
        risk_on = bool(c.iloc[-1] > ma200.iloc[-1])

        _regime_cache = {
            "as_of":     as_of,
            "risk_on":   risk_on,
            "close":     float(c.iloc[-1]),
            "ma200":     float(ma200.iloc[-1]),
            "last_date": pd.to_datetime(c.index[-1]).date(),
        }
        return _regime_cache

    except Exception as e:
        return {"as_of": as_of, "risk_on": True, "close": None, "ma200": None, "err": str(e)}


# =========================
# merge後のフィルタ＆買い抽出
# =========================
def apply_filters_and_make_buy(all_df: pd.DataFrame, val_df: pd.DataFrame, end_ts):
    all_df = all_df.copy()
    val_df = val_df.copy()

    for df_ in (all_df, val_df):
        if "ticker" not in df_.columns:
            df_["ticker"] = ""
        df_["ticker"] = df_["ticker"].astype(str).str.strip().str.upper()

    val_df = val_df.drop_duplicates(subset=["ticker"], keep="first").copy()
    all_df = all_df[all_df["ticker"].ne("")].copy()
    val_df = val_df[val_df["ticker"].ne("")].copy()

    if "skip_reason" in val_df.columns:
        val_df = val_df.rename(columns={"skip_reason": "skip_reason_yahoo"})

    for c in ["eps_ttm","market_cap","pe_ttm","pbr","mkt_price","earn_ts","earn_start_ts","earn_end_ts"]:
        if c in val_df.columns:
            val_df[c] = pd.to_numeric(val_df[c], errors="coerce")

    for c in ["adv20_m","score","pe_ttm","pbr","eps_ttm","market_cap","mkt_price","sell_score"]:
        if c in all_df.columns:
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    all_df = all_df.drop_duplicates(subset=["ticker"], keep="first").copy()

    # [FIX-15] suffixes を明示してカラム衝突リスクを排除
    merged = all_df.merge(
        val_df, on="ticker", how="left",
        validate="m:1",
        suffixes=("_tech", "_yahoo")
    )

    # skip_reason の正規化
    if "skip_reason" not in merged.columns:
        if "skip_reason_tech" in merged.columns:
            merged = merged.rename(columns={"skip_reason_tech": "skip_reason"})
        else:
            merged["skip_reason"] = ""
    merged["skip_reason"] = merged["skip_reason"].fillna("").astype(str)

    required_tech_cols = ["adv20_m", "score", "macd_buy_like", "vol_quiet", "skip_reason"]
    for c in required_tech_cols:
        if c not in merged.columns:
            if c in ("macd_buy_like", "vol_quiet"):
                merged[c] = False
            elif c == "skip_reason":
                merged[c] = ""
            else:
                merged[c] = np.nan

    if "skip_reason_yahoo" not in merged.columns:
        if "skip_reason_y" in merged.columns:
            merged = merged.rename(columns={"skip_reason_y": "skip_reason_yahoo"})
        else:
            merged["skip_reason_yahoo"] = ""
    merged["skip_reason_yahoo"] = merged["skip_reason_yahoo"].fillna("").astype(str)

    def _append_reason(mask, reason: str):
        cur = merged.loc[mask, "skip_reason"]
        merged.loc[mask, "skip_reason"] = np.where(cur.eq(""), reason, cur + "," + reason)

    # 地合い
    regime  = get_market_regime(as_of=pd.to_datetime(end_ts).strftime("%Y-%m-%d"))
    risk_on = bool(regime.get("risk_on", True))
    merged["market_risk_on"] = risk_on
    merged["market_close"]   = regime.get("close")
    merged["market_ma200"]   = regime.get("ma200")

    # ファンダ地雷除去
    eps_ok  = merged["eps_ttm"].notna()    & (merged["eps_ttm"]    > MIN_EPS_TTM)
    adv_ok  = merged["adv20_m"].notna()   & (merged["adv20_m"]    >= MIN_ADV20_M)
    mcap_ok = merged["market_cap"].notna() & (merged["market_cap"] >= MIN_MKT_CAP)
    merged["fund_ok"] = eps_ok & adv_ok & mcap_ok

    _append_reason(~eps_ok,  "fund_neg_eps")
    _append_reason(~adv_ok,  "fund_low_adv20")
    _append_reason(~mcap_ok, "fund_small_mcap")

    # 決算日関連
    merged["earn_ts_pick"] = (
        merged["earn_start_ts"].fillna(merged["earn_ts"]).fillna(merged["earn_end_ts"])
    )

    def _ts_to_date(ts):
        # [FIX-16] pd.Timestamp に統一（dateオブジェクトとの混在を回避）
        if ts is None or (isinstance(ts, float) and pd.isna(ts)):
            return pd.NaT
        try:
            ts = float(ts)
            if ts > 1e11:
                ts = ts / 1000.0
            return pd.to_datetime(ts, unit="s", utc=True).tz_convert(TZ).normalize().tz_localize(None)
        except Exception:
            return pd.NaT

    merged["next_earn_date"] = merged["earn_ts_pick"].apply(_ts_to_date)
    as_of_ts   = pd.Timestamp(regime.get("last_date") or pd.to_datetime(end_ts).date())
    as_of_date = as_of_ts.date()

    def _busdays(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        a64 = np.datetime64(a if isinstance(a, np.datetime64) else str(a)[:10])
        b64 = np.datetime64(b if isinstance(b, np.datetime64) else str(b)[:10])
        if b64 >= a64:
            return int(np.busday_count(a64, b64))
        else:
            return -int(np.busday_count(b64, a64))

    merged["days_to_earnings"] = merged["next_earn_date"].apply(
        lambda d: _busdays(as_of_date, d.date() if pd.notna(d) else None)
    )

    mask_unknown  = merged["next_earn_date"].isna()
    mask_blackout = merged["days_to_earnings"].between(
        -EARNINGS_BLACKOUT_POST_BDAYS,
         EARNINGS_BLACKOUT_PRE_BDAYS
    )
    _append_reason(mask_blackout, "earnings_blackout")
    if DROP_IF_EARNINGS_UNKNOWN:
        _append_reason(mask_unknown, "earnings_unknown")

    # BUY抽出
    base_mask = (
        merged["skip_reason"].eq("") &
        merged["fund_ok"] &
        merged["score"].ge(PULLBACK_SCORE_MIN)
    )

    if (not risk_on) and BEAR_MODE == "ignore":
        buy = merged.iloc[0:0].copy()
    elif (not risk_on) and BEAR_MODE == "tighten":
        tighten_mask = merged["macd_buy_like"].fillna(False) & merged["vol_quiet"].fillna(False)
        buy = merged[base_mask & tighten_mask].copy()
    else:
        buy = merged[base_mask].copy()

    buy = buy.sort_values(["score","adv20_m"], ascending=False)
    return merged, buy


def apply_filters_and_make_trades(all_df: pd.DataFrame, val_df: pd.DataFrame, end_ts):
    merged, buy = apply_filters_and_make_buy(all_df, val_df, end_ts)

    if "sell_score" not in merged.columns:
        merged["sell_score"] = 0
    if "sell_reason" not in merged.columns:
        merged["sell_reason"] = ""

    sell_mask = merged["sell_score"].fillna(0).ge(SELL_SCORE_MIN)

    if SELL_EXIT_BEFORE_EARNINGS and "days_to_earnings" in merged.columns:
        earn_exit = merged["days_to_earnings"].between(0, EARNINGS_EXIT_PRE_BDAYS)
        sell_mask = sell_mask | earn_exit
        sr = merged.loc[earn_exit, "sell_reason"].fillna("")
        merged.loc[earn_exit, "sell_reason"] = np.where(
            sr.eq(""), "before_earnings", sr + ";before_earnings"
        )

    sell_rows  = merged[sell_mask].copy()
    sort_cols  = [c for c in ["sell_score","adv20_m"] if c in sell_rows.columns]
    if sort_cols:
        sell_rows = sell_rows.sort_values(sort_cols, ascending=False)

    return merged, buy, sell_rows


# =========================
# main
# =========================
if __name__ == "__main__":
    # NOTE: エントリー・エグジットは「翌営業日の始値」を想定すること [FIX-14]
    # バックテスト時は open.shift(-1) を約定価格として使用してください。

    tickers = load_tickers_from_excel_bcol("data_e.xls", sheet_name="Sheet1")

    tech_df, end = screen_tech(tickers, chunk_size=20, pause_sec=0.2, max_retry=2)
    as_of_str = pd.to_datetime(end).strftime("%Y-%m-%d")

    val_df = fetch_per_pbr_batch(
        tech_df["ticker"].dropna().astype(str).tolist(),
        batch_size=200,
        pause_sec=0.2
    )

    tech_t = set(tech_df["ticker"].astype(str).str.upper())
    val_t  = set(val_df["ticker"].astype(str).str.upper())
    print("intersection(tech,val):", len(tech_t & val_t), "/", len(tech_t))
    print("val non-null pe:", val_df["pe_ttm"].notna().sum(), "/", len(val_df))

    try:
        all_rows2, buy_rows, sell_rows = apply_filters_and_make_trades(tech_df, val_df, end)
    except Exception as e:
        print("ERROR in apply_filters_and_make_trades:", repr(e))
        print("tech_df columns:", tech_df.columns.tolist())
        print("val_df columns:",  val_df.columns.tolist())
        raise

    all_rows2.to_csv(f"screen_{as_of_str}_all.csv",   index=False, encoding="utf-8-sig")
    buy_rows.to_csv(f"screen_{as_of_str}_buy.csv",    index=False, encoding="utf-8-sig")
    sell_rows.to_csv(f"screen_{as_of_str}_sell.csv",  index=False, encoding="utf-8-sig")

    print("as_of:", as_of_str, "total:", len(all_rows2), "buy:", len(buy_rows), "sell:", len(sell_rows))
    if not sell_rows.empty:
        print(sell_rows[[
            "ticker","sell_score","sell_reason","close",
            "pullback_20d","dev200","rsi","adx14","days_to_earnings"
        ]].head(30).to_string(index=False))
