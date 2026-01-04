# -*- coding: utf-8 -*-
# ローカルPCで実行（ネット接続あり想定）
# pip install yfinance pandas numpy requests
# ※ data_e.xls を読むなら環境により xlrd が必要です:
# pip install xlrd==2.0.1

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
PULLBACK_SCORE_MIN = 7       # 最初は厳しめ。買いが少なすぎるなら 5〜6 推奨

# Yahoo quote (PER/PBR) 取得用
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
YAHOO_CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
YAHOO_COOKIE_WARMUP = "https://fc.yahoo.com"

_regime_cache = {"as_of": None, "risk_on": True, "close": None, "ma200": None}

# 「決算ブラックアウト」追加用
EARNINGS_BLACKOUT_PRE_BDAYS = 3   # 決算の3営業日前から
EARNINGS_BLACKOUT_POST_BDAYS = 1  # 決算翌営業日まで
DROP_IF_EARNINGS_UNKNOWN = False  # 決算日不明を落とすなら True

# =========================
# 指標
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
    """
    ATR / NATR を計算して最後の値を返す
    NATR = ATR / Close * 100
    """
    if df is None or df.empty:
        return (np.nan, np.nan)

    high = df["High"].astype(float)
    low  = df["Low"].astype(float)
    close = df["Close"].astype(float)

    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = _rma(tr, period)
    atr_last = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else np.nan

    c_last = float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else np.nan
    natr_last = (atr_last / c_last * 100.0) if (np.isfinite(atr_last) and np.isfinite(c_last) and c_last != 0) else np.nan
    return (atr_last, natr_last)

def calc_pullback_from_high(close_adj: pd.Series, window: int = 20):
    """
    直近window日高値(終値ベース)からの下落率
    例: -0.05 = 直近高値から -5%
    """
    if close_adj is None or close_adj.dropna().empty or len(close_adj.dropna()) < window:
        return np.nan
    hh = close_adj.rolling(window).max().iloc[-1]
    if pd.isna(hh) or hh == 0:
        return np.nan
    return float(close_adj.iloc[-1] / hh - 1.0)

def calc_adx(df: pd.DataFrame, period: int = 14):
    """
    ADX(14) と +DI / -DI を計算して最後の値を返す
    """
    if df is None or df.empty or len(df) < period + 2:
        return (np.nan, np.nan, np.nan)

    high = df["High"].astype(float)
    low  = df["Low"].astype(float)
    close = df["Close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = _rma(tr, period)
    plus_di = 100.0 * (_rma(plus_dm, period) / atr.replace(0, np.nan))
    minus_di = 100.0 * (_rma(minus_dm, period) / atr.replace(0, np.nan))

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = _rma(dx, period)

    adx_last = float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else np.nan
    pdi_last = float(plus_di.iloc[-1]) if pd.notna(plus_di.iloc[-1]) else np.nan
    mdi_last = float(minus_di.iloc[-1]) if pd.notna(minus_di.iloc[-1]) else np.nan
    return (adx_last, pdi_last, mdi_last)

def score_buy_signals(df_daily, df_weekly):
    close_adj = df_daily["Adj Close"] if "Adj Close" in df_daily.columns else df_daily["Close"]
    vol = df_daily["Volume"]

    MIN_DAYS = 260
    if len(close_adj.dropna()) < MIN_DAYS:
        last_close = float(close_adj.iloc[-1]) if len(close_adj) else np.nan
        return {
            "score": 0,
            "gc_multi": False,
            "rsi": np.nan,
            "macd_buy_like": False,
            "vol_surge": False,
            "vol_quiet": False,
            "vol_down": False,
            "hurst": np.nan,
            "close": last_close,
            "close_adj": last_close,
            "adv20": np.nan,
            "adv20_m": np.nan,
            "hh252": np.nan,
            "pct_of_52w_high": np.nan,
            "max_dd_252": np.nan,
            "dev200": np.nan,
            "skip_reason": "insufficient_daily",
        }

    sma25  = close_adj.rolling(25).mean()
    sma50  = close_adj.rolling(50).mean()
    sma75  = close_adj.rolling(75).mean()
    sma200 = close_adj.rolling(200).mean()

    r = rsi(close_adj, 14)
    m, s, h = macd(close_adj)

    macd_buy = False
    h2, m2, s2 = h.dropna(), m.dropna(), s.dropna()
    if len(h2) >= 2 and len(m2) >= 1 and len(s2) >= 1:
        macd_buy = (h2.iloc[-1] > h2.iloc[-2]) and (m2.iloc[-1] > s2.iloc[-1])

    ret = close_adj.pct_change()
    vol20 = ret.rolling(20).std()
    vol_down = bool(vol20.iloc[-1] < vol20.rolling(60).mean().iloc[-1]) if pd.notna(vol20.iloc[-1]) else False

    vol20_mean = vol.rolling(20).mean().iloc[-1]
    vol_surge = bool(vol.iloc[-1] > vol20_mean * 1.5) if pd.notna(vol20_mean) else False
    vol_quiet = bool(vol.iloc[-1] <= vol20_mean * 1.2) if pd.notna(vol20_mean) else False

    H = hurst_exponent(close_adj.tail(250))
    H = float(np.clip(H, 0.0, 1.0)) if np.isfinite(H) else np.nan

    adv20 = (close_adj * vol).rolling(20).mean().iloc[-1]
    adv20_m = adv20 / 1e6 if pd.notna(adv20) else np.nan

    hh252 = close_adj.rolling(252).max().iloc[-1]
    pct_of_52w_high = (close_adj.iloc[-1] / hh252) if (pd.notna(hh252) and hh252 != 0) else np.nan

    roll_max = close_adj.rolling(252).max()
    dd = close_adj / roll_max - 1.0
    max_dd_252 = dd.iloc[-252:].min() if len(dd.dropna()) >= 252 else np.nan

    # --- Pullback特化（0〜8点） ---
    cond_trend = bool(
        pd.notna(sma200.iloc[-1]) and pd.notna(sma50.iloc[-1]) and
        (close_adj.iloc[-1] > sma200.iloc[-1]) and
        (sma50.iloc[-1] > sma200.iloc[-1])
    )

    cond_sma200_up = False
    if len(close_adj) >= 220 and pd.notna(sma200.iloc[-1]) and pd.notna(sma200.iloc[-20]):
        cond_sma200_up = bool(sma200.iloc[-1] > sma200.iloc[-20])

    dev200 = np.nan
    cond_dev200_ok = False
    if pd.notna(sma200.iloc[-1]) and sma200.iloc[-1] != 0:
        dev200 = float(close_adj.iloc[-1] / sma200.iloc[-1] - 1.0)
        cond_dev200_ok = bool(0.0 <= dev200 <= 0.12)

    band_low  = sma75.iloc[-1]
    band_high = sma25.iloc[-1]
    cond_pullback_zone = False
    if pd.notna(band_low) and pd.notna(band_high) and band_low > 0:
        cond_pullback_zone = bool(
            (close_adj.iloc[-1] >= band_low * 0.97) and
            (close_adj.iloc[-1] <= band_high * 1.03)
        )

    cond_rsi_pullback = bool(35 <= float(r.iloc[-1]) <= 55) if pd.notna(r.iloc[-1]) else False

    cond_cross_sma25_today = False
    if len(close_adj.dropna()) >= 2 and pd.notna(sma25.iloc[-2]) and pd.notna(sma25.iloc[-1]):
        cond_cross_sma25_today = bool(
            (close_adj.iloc[-2] <= sma25.iloc[-2]) and
            (close_adj.iloc[-1] >  sma25.iloc[-1])
        )
    cond_rebound = bool(cond_cross_sma25_today or macd_buy)

    cond_not_near_breakout = bool(pct_of_52w_high <= 0.91) if pd.notna(pct_of_52w_high) else False

    # === 追加：ATR/NATR（実務向けボラ）===
    atr14, natr14 = calc_atr_natr(df_daily, period=14)

    # === 追加：直近高値からの下落率（押し目の深さ）===
    pullback_20d = calc_pullback_from_high(close_adj, window=20)
    pullback_60d = calc_pullback_from_high(close_adj, window=60)

    # === 追加：ADX（トレンド強度）===
    adx14, plus_di14, minus_di14 = calc_adx(df_daily, period=14)


    conds = [
        cond_trend,
        cond_dev200_ok,
        cond_sma200_up,
        cond_pullback_zone,
        cond_rsi_pullback,
        cond_rebound,
        vol_quiet,
        cond_not_near_breakout,
    ]
    score = int(sum(conds))

    return {
        "score": score,
        "gc_multi": cond_trend,
        "rsi": float(r.iloc[-1]) if pd.notna(r.iloc[-1]) else np.nan,
        "macd_buy_like": bool(macd_buy),
        "vol_surge": bool(vol_surge),
        "vol_quiet": bool(vol_quiet),
        "vol_down": bool(vol_down),
        "hurst": float(H) if np.isfinite(H) else np.nan,
        "close": float(close_adj.iloc[-1]),
        "close_adj": float(close_adj.iloc[-1]),
        "adv20": float(adv20) if pd.notna(adv20) else np.nan,
        "adv20_m": float(adv20_m) if pd.notna(adv20_m) else np.nan,
        "hh252": float(hh252) if pd.notna(hh252) else np.nan,
        "pct_of_52w_high": float(pct_of_52w_high) if pd.notna(pct_of_52w_high) else np.nan,
        "max_dd_252": float(max_dd_252) if pd.notna(max_dd_252) else np.nan,
        "dev200": float(dev200) if np.isfinite(dev200) else np.nan,
        "atr14": float(atr14) if np.isfinite(atr14) else np.nan,
        "natr14": float(natr14) if np.isfinite(natr14) else np.nan,
        "pullback_20d": float(pullback_20d) if np.isfinite(pullback_20d) else np.nan,
        "pullback_60d": float(pullback_60d) if np.isfinite(pullback_60d) else np.nan,
        "adx14": float(adx14) if np.isfinite(adx14) else np.nan,
        "plus_di14": float(plus_di14) if np.isfinite(plus_di14) else np.nan,
        "minus_di14": float(minus_di14) if np.isfinite(minus_di14) else np.nan,
        "skip_reason": "",
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
    rows = []
    errors = []

    end = pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)

    total = len(tickers)
    total_chunks = (total + chunk_size - 1) // chunk_size
    print(f"tickers={total}, chunks={total_chunks}, range={start.date()}..{end.date()}", flush=True)

    for ci, chunk in enumerate(_chunks(tickers, chunk_size), start=1):
        print(f"[{ci}/{total_chunks}] downloading {len(chunk)} tickers...", flush=True)

        df_all = None
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

                w = (
                    d.resample("W-FRI")
                     .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                     .dropna()
                )

                meta = score_buy_signals(d, w)
                meta["ticker"] = t
                rows.append(meta)

            except Exception as e:
                errors.append((t, repr(e)))

        print(f"  done. rows={len(rows)} errors={len(errors)}", flush=True)
        time.sleep(pause_sec)

    out = pd.DataFrame(rows)
    # ★ rows が空でも列を保証（mainで tech_df["ticker"] が落ちないように）
    if out.empty:
        out = pd.DataFrame(columns=[
            "ticker",
            "score","gc_multi","rsi","macd_buy_like",
            "vol_surge","vol_quiet","vol_down","hurst",
            "close","close_adj","adv20","adv20_m",
            "hh252","pct_of_52w_high","max_dd_252","dev200",
            "atr14","natr14","pullback_20d","pullback_60d",
            "adx14","plus_di14","minus_di14",
            "skip_reason"
        ])

    if errors:
        as_of_str = pd.to_datetime(end).strftime("%Y-%m-%d")
        pd.DataFrame(errors, columns=["ticker", "error"]).to_csv(
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
        # 古い urllib3 向け
        retry = Retry(**retry_kwargs, method_whitelist=("GET",))

    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def _get_cookie_and_crumb(session: requests.Session, headers: dict, timeout_sec: int = 20) -> str:
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
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        "Referer": "https://finance.yahoo.com/",
    }

    try:
        crumb = _get_cookie_and_crumb(s, headers, timeout_sec=timeout_sec)
    except Exception as e:
        return pd.DataFrame([{
            "ticker": sym,
            "skip_reason": f"crumb_error:{type(e).__name__}:{str(e)}"[:160],
            "mkt_price": None, "pe_ttm": None, "pbr": None,
            "eps_ttm": None, "bps": None, "market_cap": None,
            "earn_ts": None, "earn_start_ts": None, "earn_end_ts": None,  # ★追加
        } for sym in symbols], columns=cols)

    out_rows = []

    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        params = {"symbols": ",".join(batch), "crumb": crumb}

        batch_error = None
        results = []
        try:
            r = s.get(YAHOO_QUOTE_URL, params=params, headers=headers, timeout=timeout_sec)
            if r.status_code == 401:
                crumb = _get_cookie_and_crumb(s, headers, timeout_sec=timeout_sec)
                params["crumb"] = crumb
                r = s.get(YAHOO_QUOTE_URL, params=params, headers=headers, timeout=timeout_sec)

            if r.status_code != 200:
                batch_error = f"http_{r.status_code}"

            data = r.json()
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

            price = q.get("regularMarketPrice")
            pe = q.get("trailingPE")
            pb = q.get("priceToBook")
            eps = q.get("epsTrailingTwelveMonths") or q.get("trailingEps")
            bps = q.get("bookValue")
            mcap = q.get("marketCap")
            earn_ts = q.get("earningsTimestamp")
            earn_start = q.get("earningsTimestampStart")
            earn_end = q.get("earningsTimestampEnd")

            if (pe is None) and (price is not None) and (eps not in (None,0,0.0)):
                pe = price / eps
            if (pb is None) and (price is not None) and (bps not in (None,0,0.0)):
                pb = price / bps

            out_rows.append({
                "ticker": sym,
                "skip_reason": batch_error,
                "mkt_price": price,
                "pe_ttm": pe,
                "pbr": pb,
                "eps_ttm": eps,
                "bps": bps,
                "market_cap": mcap,
                "earn_ts": earn_ts,
                "earn_start_ts": earn_start,
                "earn_end_ts": earn_end,
            })

        time.sleep(pause_sec)

    return pd.DataFrame(out_rows, columns=cols).drop_duplicates("ticker", keep="first")


# =========================
# 地合い（Market Regime）
# =========================
def get_market_regime(as_of=None):
    global _regime_cache
    if _regime_cache["as_of"] == as_of and _regime_cache["close"] is not None:
        return _regime_cache

    try:
        df = yf.download(
            MARKET_TICKER,
            period="2y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )
        if df is None or df.empty:
            raise ValueError("market df empty")

        c = df["Close"].dropna()
        ma200 = c.rolling(MARKET_MA_DAYS).mean()
        risk_on = bool(c.iloc[-1] > ma200.iloc[-1])

        _regime_cache = {
            "as_of": as_of,
            "risk_on": risk_on,
            "close": float(c.iloc[-1]),
            "ma200": float(ma200.iloc[-1]),
            "last_date": pd.to_datetime(c.index[-1]).date(),  # ★追加
        }
        return _regime_cache

    except Exception as e:
        return {"as_of": as_of, "risk_on": True, "close": None, "ma200": None, "err": str(e)}

# =========================
# merge後のフィルタ＆買い抽出（ここが本丸）
# =========================
def apply_filters_and_make_buy(all_df: pd.DataFrame, val_df: pd.DataFrame, end_ts):
    all_df = all_df.copy()
    val_df = val_df.copy()
    # ticker正規化
    for df_ in (all_df, val_df):
        if "ticker" not in df_.columns:
            df_["ticker"] = ""
        df_["ticker"] = df_["ticker"].astype(str).str.strip().str.upper()

    val_df = val_df.drop_duplicates(subset=["ticker"], keep="first").copy()

    # 空tickerは事故りやすいので除去（merge validate対策）
    all_df = all_df[all_df["ticker"].ne("")].copy()
    val_df = val_df[val_df["ticker"].ne("")].copy()

    # ★ 衝突回避：Yahoo側 skip_reason を別名にする
    if "skip_reason" in val_df.columns:
        val_df = val_df.rename(columns={"skip_reason": "skip_reason_yahoo"})

    # 数値列の型を揃える（Yahooがobjectになりがち）
    for c in ["eps_ttm","market_cap","pe_ttm","pbr","mkt_price",
              "earn_ts","earn_start_ts","earn_end_ts"]:
        if c in val_df.columns:
            val_df[c] = pd.to_numeric(val_df[c], errors="coerce")

    for c in ["adv20_m","score","pe_ttm","pbr","eps_ttm","market_cap","mkt_price"]:
        if c in all_df.columns:
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    # 念のため tech側も ticker 重複を潰す（入力Excelの重複対策）
    all_df = all_df.drop_duplicates(subset=["ticker"], keep="first").copy()

    # merge（ここ1回だけ）
    merged = all_df.merge(val_df, on="ticker", how="left", validate="m:1")

    # skip_reason（tech側）
    if "skip_reason" not in merged.columns:
        if "skip_reason_x" in merged.columns:
            merged = merged.rename(columns={"skip_reason_x": "skip_reason"})
        else:
            merged["skip_reason"] = ""
    merged["skip_reason"] = merged["skip_reason"].fillna("").astype(str)

    # ---- 必要列の存在保証（tech側も） ----
    required_tech_cols = ["adv20_m", "score", "macd_buy_like", "vol_quiet", "skip_reason"]
    for c in required_tech_cols:
        if c not in merged.columns:
            if c in ("macd_buy_like", "vol_quiet"):
                merged[c] = False
            elif c == "skip_reason":
                merged[c] = ""
            else:
                merged[c] = np.nan

    # Yahoo側理由列（デバッグ用）
    if "skip_reason_yahoo" not in merged.columns:
        if "skip_reason_y" in merged.columns:
            merged = merged.rename(columns={"skip_reason_y": "skip_reason_yahoo"})
        else:
            merged["skip_reason_yahoo"] = ""
    merged["skip_reason_yahoo"] = merged["skip_reason_yahoo"].fillna("").astype(str)

    # skip_reason 追記ヘルパ（★1回だけ定義）
    def _append_reason(mask, reason: str):
        cur = merged.loc[mask, "skip_reason"]
        merged.loc[mask, "skip_reason"] = np.where(cur.eq(""), reason, cur + "," + reason)

    # 地合い
    regime = get_market_regime(as_of=pd.to_datetime(end_ts).strftime("%Y-%m-%d"))
    risk_on = bool(regime.get("risk_on", True))
    merged["market_risk_on"] = risk_on
    merged["market_close"] = regime.get("close")
    merged["market_ma200"] = regime.get("ma200")

    # ファンダ地雷除去（ベクタ）
    eps_ok  = merged["eps_ttm"].notna() & (merged["eps_ttm"] > MIN_EPS_TTM)
    adv_ok  = merged["adv20_m"].notna() & (merged["adv20_m"] >= MIN_ADV20_M)
    mcap_ok = merged["market_cap"].notna() & (merged["market_cap"] >= MIN_MKT_CAP)
    merged["fund_ok"] = eps_ok & adv_ok & mcap_ok

    _append_reason(~eps_ok,  "fund_neg_eps")
    _append_reason(~adv_ok,  "fund_low_adv20")
    _append_reason(~mcap_ok, "fund_small_mcap")

    # ===== 決算日フィルタ（★ここを buy抽出より前に移動）=====
    merged["earn_ts_pick"] = merged["earn_start_ts"].fillna(merged["earn_ts"]).fillna(merged["earn_end_ts"])

    def _ts_to_date(ts):
        if ts is None or (isinstance(ts, float) and pd.isna(ts)):
            return pd.NaT
        try:
            # 数値化（"12345" みたいな文字列も吸収）
            ts = float(ts)

            # epochミリ秒っぽい場合は秒に直す（閾値はざっくり）
            # 10^11秒は西暦5138年なので現実的にありえない → ms扱いで割る
            if ts > 1e11:
                ts = ts / 1000.0

            return pd.to_datetime(ts, unit="s", utc=True).tz_convert(TZ).date()
        except Exception:
            return pd.NaT

    merged["next_earn_date"] = merged["earn_ts_pick"].apply(_ts_to_date)
    as_of_date = regime.get("last_date") or pd.to_datetime(end_ts).date()

    def _busdays(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        a64 = np.datetime64(a)
        b64 = np.datetime64(b)
        if b64 >= a64:
            return int(np.busday_count(a64, b64))
        else:
            return -int(np.busday_count(b64, a64))

    merged["days_to_earnings"] = merged["next_earn_date"].apply(lambda d: _busdays(as_of_date, d))

    mask_unknown  = merged["next_earn_date"].isna()
    mask_blackout = merged["days_to_earnings"].between(-EARNINGS_BLACKOUT_PRE_BDAYS, EARNINGS_BLACKOUT_POST_BDAYS)

    _append_reason(mask_blackout, "earnings_blackout")
    if DROP_IF_EARNINGS_UNKNOWN:
        _append_reason(mask_unknown, "earnings_unknown")

    # ===== buy抽出（★決算フィルタ後の skip_reason を使う）=====
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

# =========================
# main
# =========================
if __name__ == "__main__":
    # tickers = load_tse_tickers_from_jpx_xls("data_e.xls", domestic_common_only=True)
    tickers = load_tickers_from_excel_bcol("data_e.xls", sheet_name="Sheet1")

    tech_df, end = screen_tech(tickers, chunk_size=20, pause_sec=0.2, max_retry=2)
    as_of_str = pd.to_datetime(end).strftime("%Y-%m-%d")

    val_df = fetch_per_pbr_batch(
        tech_df["ticker"].dropna().astype(str).tolist(),
        batch_size=200,
        pause_sec=0.2
    )

    # 交差チェック
    tech_t = set(tech_df["ticker"].astype(str).str.upper())
    val_t  = set(val_df["ticker"].astype(str).str.upper())
    print("intersection(tech,val):", len(tech_t & val_t), "/", len(tech_t))
    print("val non-null pe:", val_df["pe_ttm"].notna().sum(), "/", len(val_df))

    # ★ここで all_rows2 を作る（重要）
    try:
        all_rows2, buy_rows = apply_filters_and_make_buy(tech_df, val_df, end)
    except Exception as e:
        print("ERROR in apply_filters_and_make_buy:", repr(e))
        print("tech_df columns:", tech_df.columns.tolist())
        print("val_df columns:", val_df.columns.tolist())
        raise

    # ★ all_rows2 が定義された後で print（重要）
    print([c for c in all_rows2.columns if "skip_reason" in c])
    print(all_rows2[["ticker", "skip_reason"]].head(5).to_string(index=False))

    all_rows2.to_csv(f"screen_{as_of_str}_all.csv", index=False, encoding="utf-8-sig")
    buy_rows.to_csv(f"screen_{as_of_str}_buy.csv", index=False, encoding="utf-8-sig")

    print("as_of:", as_of_str, "total:", len(all_rows2), "buy:", len(buy_rows))
    if not buy_rows.empty:
        print(buy_rows[["ticker","score","pe_ttm","pbr","adv20_m","pct_of_52w_high","market_risk_on"]]
              .head(30).to_string(index=False))
