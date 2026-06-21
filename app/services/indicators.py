import math
import numpy as np
import pandas as pd
import pandas_ta as ta


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(val, decimals: int = 2):
    """Return rounded float or None if NaN/None."""
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, decimals)
    except (TypeError, ValueError):
        return None


def _compute_supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
    """Compute SuperTrend indicator manually (robust across pandas_ta versions)."""
    hl2 = (df["High"] + df["Low"]) / 2
    atr = ta.atr(df["High"], df["Low"], df["Close"], length=period)

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    direction = pd.Series(index=df.index, dtype=float)
    supertrend = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if i == 0:
            direction.iloc[i] = 1.0
            supertrend.iloc[i] = lower.iloc[i]
            continue

        prev_dir = direction.iloc[i - 1]
        prev_st = supertrend.iloc[i - 1]
        close = df["Close"].iloc[i]

        cur_lower = lower.iloc[i]
        cur_upper = upper.iloc[i]

        # Adjust bands to ensure continuity
        if cur_lower > prev_st or prev_dir == -1:
            cur_lower = cur_lower
        else:
            cur_lower = max(cur_lower, prev_st) if prev_dir == 1 else cur_lower

        if cur_upper < prev_st or prev_dir == 1:
            cur_upper = cur_upper
        else:
            cur_upper = min(cur_upper, prev_st) if prev_dir == -1 else cur_upper

        if prev_dir == 1:
            direction.iloc[i] = 1.0 if close >= cur_lower else -1.0
        else:
            direction.iloc[i] = -1.0 if close <= cur_upper else 1.0

        supertrend.iloc[i] = cur_lower if direction.iloc[i] == 1.0 else cur_upper

    df["SuperTrend"] = supertrend
    df["SuperTrend_Dir"] = direction   # 1 = uptrend, -1 = downtrend
    return df


def _compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Ichimoku Cloud components."""
    high, low, close = df["High"], df["Low"], df["Close"]

    df["ICH_Tenkan"] = (high.rolling(9).max() + low.rolling(9).min()) / 2
    df["ICH_Kijun"] = (high.rolling(26).max() + low.rolling(26).min()) / 2

    # Cloud values at current bar (unshifted — for signal evaluation)
    df["ICH_SpanA"] = (df["ICH_Tenkan"] + df["ICH_Kijun"]) / 2
    df["ICH_SpanB"] = (high.rolling(52).max() + low.rolling(52).min()) / 2

    # Chikou span: current close plotted 26 bars back (use as-is for comparison)
    df["ICH_Chikou"] = close.shift(-26)

    df["ICH_CloudTop"] = df[["ICH_SpanA", "ICH_SpanB"]].max(axis=1)
    df["ICH_CloudBot"] = df[["ICH_SpanA", "ICH_SpanB"]].min(axis=1)
    return df


def _compute_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Classic pivot points from previous bar."""
    ph = df["High"].shift(1)
    pl = df["Low"].shift(1)
    pc = df["Close"].shift(1)

    df["PP"] = (ph + pl + pc) / 3
    df["PP_R1"] = 2 * df["PP"] - pl
    df["PP_S1"] = 2 * df["PP"] - ph
    df["PP_R2"] = df["PP"] + (ph - pl)
    df["PP_S2"] = df["PP"] - (ph - pl)
    df["PP_R3"] = ph + 2 * (df["PP"] - pl)
    df["PP_S3"] = pl - 2 * (ph - df["PP"])
    return df


# ─── Main indicator computation ───────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute ALL technical indicators; return enriched DataFrame + latest summary."""

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["RSI"] = ta.rsi(df["Close"], length=14)

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_Signal"] = macd["MACDs_12_26_9"]
    df["MACD_Hist"] = macd["MACDh_12_26_9"]

    # ── Moving Averages ───────────────────────────────────────────────────────
    df["SMA_20"]  = ta.sma(df["Close"], length=20)
    df["SMA_50"]  = ta.sma(df["Close"], length=50)
    df["SMA_200"] = ta.sma(df["Close"], length=200)

    df["EMA_12"]  = ta.ema(df["Close"], length=12)
    df["EMA_20"]  = ta.ema(df["Close"], length=20)
    df["EMA_26"]  = ta.ema(df["Close"], length=26)
    df["EMA_50"]  = ta.ema(df["Close"], length=50)
    df["EMA_100"] = ta.ema(df["Close"], length=100)
    df["EMA_200"] = ta.ema(df["Close"], length=200)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = ta.bbands(df["Close"], length=20)
    df["BB_Upper"] = bb[[c for c in bb.columns if "BBU" in c][0]]
    df["BB_Lower"] = bb[[c for c in bb.columns if "BBL" in c][0]]
    df["BB_Mid"]   = bb[[c for c in bb.columns if "BBM" in c][0]]

    # ── ATR ───────────────────────────────────────────────────────────────────
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # ── ADX ───────────────────────────────────────────────────────────────────
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=14)
    if adx is not None:
        adx_col = [c for c in adx.columns if c.startswith("ADX_")]
        dmp_col = [c for c in adx.columns if c.startswith("DMP_")]
        dmn_col = [c for c in adx.columns if c.startswith("DMN_")]
        df["ADX"] = adx[adx_col[0]] if adx_col else np.nan
        df["DMP"] = adx[dmp_col[0]] if dmp_col else np.nan
        df["DMN"] = adx[dmn_col[0]] if dmn_col else np.nan
    else:
        df["ADX"] = df["DMP"] = df["DMN"] = np.nan

    # ── SuperTrend ────────────────────────────────────────────────────────────
    df = _compute_supertrend(df, period=7, multiplier=3.0)

    # ── VWAP (rolling 20-period — meaningful for daily charts) ────────────────
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical_price * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()

    # ── Ichimoku Cloud ────────────────────────────────────────────────────────
    df = _compute_ichimoku(df)

    # ── OBV ───────────────────────────────────────────────────────────────────
    df["OBV"] = ta.obv(df["Close"], df["Volume"])
    df["OBV_MA20"] = df["OBV"].rolling(20).mean()

    # ── MFI ───────────────────────────────────────────────────────────────────
    df["MFI"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)

    # ── Stochastic ────────────────────────────────────────────────────────────
    stoch = ta.stoch(df["High"], df["Low"], df["Close"])
    if stoch is not None:
        sk = [c for c in stoch.columns if "STOCHk" in c]
        sd = [c for c in stoch.columns if "STOCHd" in c]
        df["STOCH_K"] = stoch[sk[0]] if sk else np.nan
        df["STOCH_D"] = stoch[sd[0]] if sd else np.nan
    else:
        df["STOCH_K"] = df["STOCH_D"] = np.nan

    # ── CCI ───────────────────────────────────────────────────────────────────
    df["CCI"] = ta.cci(df["High"], df["Low"], df["Close"], length=14)

    # ── Williams %R ───────────────────────────────────────────────────────────
    df["WILLIAMS_R"] = ta.willr(df["High"], df["Low"], df["Close"], length=14)

    # ── Keltner Channel ───────────────────────────────────────────────────────
    kc = ta.kc(df["High"], df["Low"], df["Close"])
    if kc is not None:
        kcu = [c for c in kc.columns if "KCU" in c]
        kcl = [c for c in kc.columns if "KCL" in c]
        kcb = [c for c in kc.columns if "KCB" in c]
        df["KC_Upper"] = kc[kcu[0]] if kcu else np.nan
        df["KC_Lower"] = kc[kcl[0]] if kcl else np.nan
        df["KC_Mid"]   = kc[kcb[0]] if kcb else np.nan
    else:
        df["KC_Upper"] = df["KC_Lower"] = df["KC_Mid"] = np.nan

    # ── Pivot Points ──────────────────────────────────────────────────────────
    df = _compute_pivot_points(df)

    # ── Volume ────────────────────────────────────────────────────────────────
    df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Spike"] = (df["Volume"] > df["Volume_MA20"] * 1.5).astype(int)

    # ── Returns ───────────────────────────────────────────────────────────────
    df["Daily_Return"] = df["Close"].pct_change()
    df["Cum_Return"]   = (1 + df["Daily_Return"]).cumprod() - 1

    # ── Sharpe Ratio (rolling 252-bar, annualised) ────────────────────────────
    roll_mean = df["Daily_Return"].rolling(252).mean()
    roll_std  = df["Daily_Return"].rolling(252).std()
    df["Sharpe"] = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252)

    # ── Value at Risk — 95 % (rolling 252-bar) ───────────────────────────────
    df["VaR_95"] = df["Daily_Return"].rolling(252).quantile(0.05)

    # ── Max Drawdown (rolling 252-bar) ───────────────────────────────────────
    rolling_max = df["Close"].rolling(252).max()
    df["Max_Drawdown"] = ((df["Close"] - rolling_max) / rolling_max) * 100

    # ── Doji Candlestick ──────────────────────────────────────────────────────
    body         = (df["Close"] - df["Open"]).abs()
    candle_range = (df["High"] - df["Low"]).replace(0, np.nan)
    df["Doji"] = (body / candle_range < 0.1).astype(int)

    # ── Hammer ────────────────────────────────────────────────────────────────
    lower_shadow = df[["Open", "Close"]].min(axis=1) - df["Low"]
    upper_shadow = df["High"] - df[["Open", "Close"]].max(axis=1)
    df["Hammer"] = (
        (lower_shadow >= 2 * body) &
        (upper_shadow <= 0.1 * body.replace(0, np.nan)) &
        (body > 0)
    ).astype(int)

    # ── Shooting Star ─────────────────────────────────────────────────────────
    df["ShootingStar"] = (
        (upper_shadow >= 2 * body) &
        (lower_shadow <= 0.1 * body.replace(0, np.nan)) &
        (body > 0)
    ).astype(int)

    # ── Bullish / Bearish Engulfing ────────────────────────────────────────────
    prev_open  = df["Open"].shift(1)
    prev_close = df["Close"].shift(1)
    df["BullEngulf"] = (
        (prev_close < prev_open) &                          # previous bearish
        (df["Close"] > df["Open"]) &                        # current bullish
        (df["Open"] <= prev_close) &
        (df["Close"] >= prev_open)
    ).astype(int)
    df["BearEngulf"] = (
        (prev_close > prev_open) &                          # previous bullish
        (df["Close"] < df["Open"]) &                        # current bearish
        (df["Open"] >= prev_close) &
        (df["Close"] <= prev_open)
    ).astype(int)

    # ── Fibonacci Retracement (rolling 52-week) ───────────────────────────────
    window       = min(252, len(df))
    period_high  = df["High"].rolling(window).max()
    period_low   = df["Low"].rolling(window).min()
    fib_range    = period_high - period_low
    df["Fib_0"]   = period_low
    df["Fib_236"] = period_low + 0.236 * fib_range
    df["Fib_382"] = period_low + 0.382 * fib_range
    df["Fib_500"] = period_low + 0.500 * fib_range
    df["Fib_618"] = period_low + 0.618 * fib_range
    df["Fib_100"] = period_high

    # ── Drop rows missing core indicators ─────────────────────────────────────
    df.dropna(subset=["RSI", "MACD", "ATR"], inplace=True)

    # ── Summary (latest bar) ──────────────────────────────────────────────────
    latest = df.iloc[-1]

    summary = {
        # Core momentum
        "rsi":          _safe_float(latest["RSI"]),
        "macd":         _safe_float(latest["MACD"], 4),
        "macd_signal":  _safe_float(latest["MACD_Signal"], 4),
        "macd_hist":    _safe_float(latest["MACD_Hist"], 4),
        # Moving averages
        "sma_20":       _safe_float(latest["SMA_20"]),
        "sma_50":       _safe_float(latest["SMA_50"]),
        "sma_200":      _safe_float(latest["SMA_200"]),
        "ema_12":       _safe_float(latest["EMA_12"]),
        "ema_20":       _safe_float(latest["EMA_20"]),
        "ema_26":       _safe_float(latest["EMA_26"]),
        "ema_50":       _safe_float(latest["EMA_50"]),
        "ema_100":      _safe_float(latest["EMA_100"]),
        "ema_200":      _safe_float(latest["EMA_200"]),
        # Bollinger
        "bb_upper":     _safe_float(latest["BB_Upper"]),
        "bb_lower":     _safe_float(latest["BB_Lower"]),
        "bb_mid":       _safe_float(latest["BB_Mid"]),
        # Trend / volatility
        "atr":          _safe_float(latest["ATR"]),
        "adx":          _safe_float(latest["ADX"]),
        "dmp":          _safe_float(latest["DMP"]),
        "dmn":          _safe_float(latest["DMN"]),
        "supertrend":   _safe_float(latest["SuperTrend"]),
        "supertrend_dir": int(latest["SuperTrend_Dir"]) if not math.isnan(float(latest["SuperTrend_Dir"])) else None,
        "vwap":         _safe_float(latest["VWAP"]),
        # Ichimoku
        "ich_tenkan":   _safe_float(latest["ICH_Tenkan"]),
        "ich_kijun":    _safe_float(latest["ICH_Kijun"]),
        "ich_cloud_top":_safe_float(latest["ICH_CloudTop"]),
        "ich_cloud_bot":_safe_float(latest["ICH_CloudBot"]),
        # Volume-based
        "obv":          _safe_float(latest["OBV"], 0),
        "obv_ma20":     _safe_float(latest["OBV_MA20"], 0),
        "mfi":          _safe_float(latest["MFI"]),
        "volume_ma20":  int(latest["Volume_MA20"]) if not math.isnan(float(latest["Volume_MA20"])) else None,
        "volume_spike": bool(latest["Volume_Spike"]),
        # Oscillators
        "stoch_k":      _safe_float(latest["STOCH_K"]),
        "stoch_d":      _safe_float(latest["STOCH_D"]),
        "cci":          _safe_float(latest["CCI"]),
        "williams_r":   _safe_float(latest["WILLIAMS_R"]),
        # Keltner
        "kc_upper":     _safe_float(latest["KC_Upper"]),
        "kc_lower":     _safe_float(latest["KC_Lower"]),
        "kc_mid":       _safe_float(latest["KC_Mid"]),
        # Pivot points
        "pivot": {
            "pp":   _safe_float(latest["PP"]),
            "r1":   _safe_float(latest["PP_R1"]),
            "r2":   _safe_float(latest["PP_R2"]),
            "r3":   _safe_float(latest["PP_R3"]),
            "s1":   _safe_float(latest["PP_S1"]),
            "s2":   _safe_float(latest["PP_S2"]),
            "s3":   _safe_float(latest["PP_S3"]),
        },
        # Fibonacci
        "fib_levels": {
            "0":    _safe_float(latest["Fib_0"]),
            "23.6": _safe_float(latest["Fib_236"]),
            "38.2": _safe_float(latest["Fib_382"]),
            "50.0": _safe_float(latest["Fib_500"]),
            "61.8": _safe_float(latest["Fib_618"]),
            "100":  _safe_float(latest["Fib_100"]),
        },
        # Returns & risk
        "cum_return_pct":  _safe_float(latest["Cum_Return"] * 100),
        "sharpe":          _safe_float(latest["Sharpe"], 3),
        "var_95_pct":      _safe_float(latest["VaR_95"] * 100, 3),
        "max_drawdown_pct":_safe_float(latest["Max_Drawdown"]),
        # Candlestick patterns
        "doji":           bool(latest["Doji"]),
        "hammer":         bool(latest["Hammer"]),
        "shooting_star":  bool(latest["ShootingStar"]),
        "bull_engulf":    bool(latest["BullEngulf"]),
        "bear_engulf":    bool(latest["BearEngulf"]),
    }

    return df, summary
