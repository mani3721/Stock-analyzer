import math
import pandas as pd

# ─── Default config ────────────────────────────────────────────────────────────
DEFAULT_CRITERIA_CONFIG: dict[str, dict] = {
    # Trend
    "EMA20":       {"enabled": True},
    "EMA50":       {"enabled": True},
    "EMA100":      {"enabled": True},
    "EMA200":      {"enabled": True},
    "SMA50":       {"enabled": True},
    "SMA200":      {"enabled": True},
    "SUPERTREND":  {"enabled": True},
    "ADX":         {"enabled": True, "min_strength": 25},
    "VWAP":        {"enabled": True},
    "ICHIMOKU":    {"enabled": True},
    # Momentum
    "RSI":         {"enabled": True, "oversold": 30, "overbought": 70},
    "MACD":        {"enabled": True},
    "STOCHASTIC":  {"enabled": True, "oversold": 20, "overbought": 80},
    "CCI":         {"enabled": True, "oversold": -100, "overbought": 100},
    "WILLIAMS_R":  {"enabled": True, "oversold": -80, "overbought": -20},
    # Volume
    "VOLUME_SMA":  {"enabled": True, "spike_threshold": 1.5},
    "OBV":         {"enabled": True},
    "MFI":         {"enabled": True, "oversold": 20, "overbought": 80},
    # Volatility
    "BOLLINGER":   {"enabled": True},
    "ATR":         {"enabled": True},
    "KELTNER":     {"enabled": True},
    # Support & Resistance
    "FIB":         {"enabled": True, "proximity_pct": 2.0},
    "PIVOT":       {"enabled": True, "proximity_pct": 0.5},
    # Candlestick
    "DOJI":        {"enabled": True},
    "HAMMER":      {"enabled": True},
    "SHOOTING_STAR": {"enabled": True},
    "ENGULFING":   {"enabled": True},
    # Fundamentals
    "PE":          {"enabled": True, "max": 25},
    "ROE":         {"enabled": True, "min": 15},
    "REVENUE_GROWTH": {"enabled": True, "min": 10},
    "DEBT_EQUITY": {"enabled": True, "max": 1.5},
    "EPS_GROWTH":  {"enabled": True, "min": 10},
    # Risk
    "SHARPE":      {"enabled": True, "min": 0.5},
    "VAR":         {"enabled": True, "max_loss_pct": -2.0},
    "MAX_DRAWDOWN":{"enabled": True, "max_pct": -20},
    # Legacy keys (backward compat)
    "EMA_Cross":   {"enabled": True},
    "Bollinger":   {"enabled": True},
    "Volume":      {"enabled": True, "spike_threshold": 1.5},
}

AVAILABLE_CRITERIA = [
    # Trend
    "EMA20", "EMA50", "EMA100", "EMA200", "SMA50", "SMA200",
    "SUPERTREND", "ADX", "VWAP", "ICHIMOKU",
    # Momentum
    "RSI", "MACD", "STOCHASTIC", "CCI", "WILLIAMS_R",
    # Volume
    "VOLUME_SMA", "OBV", "MFI",
    # Volatility
    "BOLLINGER", "ATR", "KELTNER",
    # Support & Resistance
    "FIB", "PIVOT",
    # Candlestick
    "DOJI", "HAMMER", "SHOOTING_STAR", "ENGULFING",
    # Fundamentals
    "PE", "ROE", "REVENUE_GROWTH", "DEBT_EQUITY", "EPS_GROWTH",
    # Risk
    "SHARPE", "VAR", "MAX_DRAWDOWN",
]

# Keys that need AI / sentiment results — evaluated after main pipeline
POST_ANALYSIS_CRITERIA = {"NEWS_SENTIMENT", "BUY_PROB", "SELL_PROB", "CONFIDENCE"}


# ─── Utility ──────────────────────────────────────────────────────────────────

def _nan(val) -> bool:
    try:
        return val is None or math.isnan(float(val))
    except (TypeError, ValueError):
        return True


# ─── Trend evaluators ─────────────────────────────────────────────────────────

def _evaluate_ema20(latest, cfg):
    price, ema = latest["Close"], latest["EMA_20"]
    if _nan(ema):
        return "HOLD", "EMA20 not available"
    return ("BUY", f"Price {price:.2f} > EMA20 {ema:.2f} — Uptrend") if price > ema else \
           ("SELL", f"Price {price:.2f} < EMA20 {ema:.2f} — Downtrend")


def _evaluate_ema50(latest, cfg):
    price, ema = latest["Close"], latest["EMA_50"]
    if _nan(ema):
        return "HOLD", "EMA50 not available"
    return ("BUY", f"Price {price:.2f} > EMA50 {ema:.2f} — Uptrend") if price > ema else \
           ("SELL", f"Price {price:.2f} < EMA50 {ema:.2f} — Downtrend")


def _evaluate_ema100(latest, cfg):
    price, ema = latest["Close"], latest.get("EMA_100")
    if _nan(ema):
        return "HOLD", "EMA100 not available (need 100+ bars)"
    return ("BUY", f"Price {price:.2f} > EMA100 {ema:.2f} — Uptrend") if price > ema else \
           ("SELL", f"Price {price:.2f} < EMA100 {ema:.2f} — Downtrend")


def _evaluate_ema200(latest, cfg):
    price, ema = latest["Close"], latest.get("EMA_200")
    if _nan(ema):
        return "HOLD", "EMA200 not available (need 200+ bars)"
    return ("BUY", f"Price {price:.2f} > EMA200 {ema:.2f} — Long-term uptrend") if price > ema else \
           ("SELL", f"Price {price:.2f} < EMA200 {ema:.2f} — Long-term downtrend")


def _evaluate_sma50(latest, cfg):
    price, sma = latest["Close"], latest["SMA_50"]
    if _nan(sma):
        return "HOLD", "SMA50 not available"
    return ("BUY", f"Price {price:.2f} > SMA50 {sma:.2f}") if price > sma else \
           ("SELL", f"Price {price:.2f} < SMA50 {sma:.2f}")


def _evaluate_sma200(latest, cfg):
    price, sma = latest["Close"], latest.get("SMA_200")
    if _nan(sma):
        return "HOLD", "SMA200 not available (need 200+ bars)"
    return ("BUY", f"Price {price:.2f} > SMA200 {sma:.2f} — Long-term uptrend") if price > sma else \
           ("SELL", f"Price {price:.2f} < SMA200 {sma:.2f} — Long-term downtrend")


def _evaluate_ema_cross(latest, cfg):
    e12, e26 = latest["EMA_12"], latest["EMA_26"]
    return ("BUY", f"EMA12 {e12:.2f} > EMA26 {e26:.2f} — Golden cross") if e12 > e26 else \
           ("SELL", f"EMA12 {e12:.2f} < EMA26 {e26:.2f} — Death cross")


def _evaluate_supertrend(latest, cfg):
    direction = latest.get("SuperTrend_Dir")
    st = latest.get("SuperTrend")
    if _nan(direction):
        return "HOLD", "SuperTrend not available"
    price = latest["Close"]
    if direction == 1:
        return "BUY", f"SuperTrend UP — price {price:.2f} above support {st:.2f}"
    return "SELL", f"SuperTrend DOWN — price {price:.2f} below resistance {st:.2f}"


def _evaluate_adx(latest, cfg):
    adx = latest.get("ADX")
    dmp = latest.get("DMP")
    dmn = latest.get("DMN")
    if _nan(adx):
        return "HOLD", "ADX not available"
    min_strength = cfg.get("min_strength", 25)
    strength = "Strong" if adx >= min_strength else "Weak"
    if adx >= min_strength:
        if not _nan(dmp) and not _nan(dmn):
            if dmp > dmn:
                return "BUY", f"ADX={adx:.1f} ({strength}) with DI+{dmp:.1f} > DI-{dmn:.1f} — Bullish trend"
            return "SELL", f"ADX={adx:.1f} ({strength}) with DI-{dmn:.1f} > DI+{dmp:.1f} — Bearish trend"
    return "HOLD", f"ADX={adx:.1f} — {strength} trend / no clear direction"


def _evaluate_vwap(latest, cfg):
    vwap = latest.get("VWAP")
    if _nan(vwap):
        return "HOLD", "VWAP not available"
    price = latest["Close"]
    if price > vwap:
        return "BUY", f"Price {price:.2f} > VWAP {vwap:.2f} — Bullish"
    return "SELL", f"Price {price:.2f} < VWAP {vwap:.2f} — Bearish"


def _evaluate_ichimoku(latest, cfg):
    top = latest.get("ICH_CloudTop")
    bot = latest.get("ICH_CloudBot")
    tenkan = latest.get("ICH_Tenkan")
    kijun = latest.get("ICH_Kijun")
    if _nan(top) or _nan(bot):
        return "HOLD", "Ichimoku Cloud not available"
    price = latest["Close"]
    if price > top:
        return "BUY", f"Price {price:.2f} above Cloud ({bot:.2f}–{top:.2f}) — Bullish"
    elif price < bot:
        return "SELL", f"Price {price:.2f} below Cloud ({bot:.2f}–{top:.2f}) — Bearish"
    return "HOLD", f"Price {price:.2f} inside Cloud ({bot:.2f}–{top:.2f}) — Neutral"


# ─── Momentum evaluators ──────────────────────────────────────────────────────

def _evaluate_rsi(latest, cfg):
    rsi = latest["RSI"]
    os, ob = cfg.get("oversold", 30), cfg.get("overbought", 70)
    if rsi < os:
        return "BUY", f"RSI={rsi:.1f} — Oversold (below {os})"
    elif rsi > ob:
        return "SELL", f"RSI={rsi:.1f} — Overbought (above {ob})"
    return "HOLD", f"RSI={rsi:.1f} — Neutral ({os}–{ob})"


def _evaluate_macd(latest, cfg):
    m, s = latest["MACD"], latest["MACD_Signal"]
    return ("BUY", f"MACD {m:.4f} > Signal {s:.4f} — Bullish") if m > s else \
           ("SELL", f"MACD {m:.4f} < Signal {s:.4f} — Bearish")


def _evaluate_stochastic(latest, cfg):
    k = latest.get("STOCH_K")
    d = latest.get("STOCH_D")
    if _nan(k):
        return "HOLD", "Stochastic not available"
    os, ob = cfg.get("oversold", 20), cfg.get("overbought", 80)
    if k < os:
        return "BUY", f"Stoch-K={k:.1f} — Oversold (below {os})"
    elif k > ob:
        return "SELL", f"Stoch-K={k:.1f} — Overbought (above {ob})"
    if not _nan(d):
        if k > d:
            return "BUY", f"Stoch-K={k:.1f} crossed above D={d:.1f} — Bullish"
        return "SELL", f"Stoch-K={k:.1f} crossed below D={d:.1f} — Bearish"
    return "HOLD", f"Stoch-K={k:.1f} — Neutral"


def _evaluate_cci(latest, cfg):
    cci = latest.get("CCI")
    if _nan(cci):
        return "HOLD", "CCI not available"
    os, ob = cfg.get("oversold", -100), cfg.get("overbought", 100)
    if cci < os:
        return "BUY", f"CCI={cci:.1f} — Oversold (below {os})"
    elif cci > ob:
        return "SELL", f"CCI={cci:.1f} — Overbought (above {ob})"
    return "HOLD", f"CCI={cci:.1f} — Neutral"


def _evaluate_williams_r(latest, cfg):
    wr = latest.get("WILLIAMS_R")
    if _nan(wr):
        return "HOLD", "Williams %R not available"
    os, ob = cfg.get("oversold", -80), cfg.get("overbought", -20)
    if wr < os:
        return "BUY", f"Williams %R={wr:.1f} — Oversold (below {os})"
    elif wr > ob:
        return "SELL", f"Williams %R={wr:.1f} — Overbought (above {ob})"
    return "HOLD", f"Williams %R={wr:.1f} — Neutral"


# ─── Volume evaluators ────────────────────────────────────────────────────────

def _evaluate_volume_sma(latest, cfg):
    threshold = cfg.get("spike_threshold", 1.5)
    ratio = latest["Volume"] / latest["Volume_MA20"]
    if ratio > threshold:
        return "BUY", f"Volume {ratio:.1f}x MA20 — Spike (>{threshold}x)"
    return "HOLD", f"Volume {ratio:.1f}x MA20 — Normal"


def _evaluate_obv(latest, cfg):
    obv = latest.get("OBV")
    obv_ma = latest.get("OBV_MA20")
    if _nan(obv) or _nan(obv_ma):
        return "HOLD", "OBV not available"
    if obv > obv_ma:
        return "BUY", f"OBV {obv:,.0f} > OBV-MA20 {obv_ma:,.0f} — Accumulation"
    return "SELL", f"OBV {obv:,.0f} < OBV-MA20 {obv_ma:,.0f} — Distribution"


def _evaluate_mfi(latest, cfg):
    mfi = latest.get("MFI")
    if _nan(mfi):
        return "HOLD", "MFI not available"
    os, ob = cfg.get("oversold", 20), cfg.get("overbought", 80)
    if mfi < os:
        return "BUY", f"MFI={mfi:.1f} — Oversold (below {os})"
    elif mfi > ob:
        return "SELL", f"MFI={mfi:.1f} — Overbought (above {ob})"
    return "HOLD", f"MFI={mfi:.1f} — Neutral"


# ─── Volatility evaluators ────────────────────────────────────────────────────

def _evaluate_bollinger(latest, cfg):
    price, upper, lower = latest["Close"], latest["BB_Upper"], latest["BB_Lower"]
    if price < lower:
        return "BUY", f"Price {price:.2f} below BB Lower {lower:.2f} — Oversold"
    elif price > upper:
        return "SELL", f"Price {price:.2f} above BB Upper {upper:.2f} — Overbought"
    return "HOLD", f"Price within Bollinger Bands ({lower:.2f}–{upper:.2f})"


def _evaluate_atr(latest, cfg):
    atr, price = latest["ATR"], latest["Close"]
    pct = (atr / price) * 100
    if pct > 3.0:
        return "SELL", f"ATR={atr:.2f} ({pct:.1f}%) — High volatility"
    elif pct < 1.0:
        return "BUY", f"ATR={atr:.2f} ({pct:.1f}%) — Low volatility (stable)"
    return "HOLD", f"ATR={atr:.2f} ({pct:.1f}%) — Normal volatility"


def _evaluate_keltner(latest, cfg):
    upper = latest.get("KC_Upper")
    lower = latest.get("KC_Lower")
    if _nan(upper) or _nan(lower):
        return "HOLD", "Keltner Channel not available"
    price = latest["Close"]
    if price < lower:
        return "BUY", f"Price {price:.2f} below Keltner Lower {lower:.2f} — Oversold"
    elif price > upper:
        return "SELL", f"Price {price:.2f} above Keltner Upper {upper:.2f} — Overbought"
    return "HOLD", f"Price within Keltner Channel ({lower:.2f}–{upper:.2f})"


# ─── Support & Resistance evaluators ─────────────────────────────────────────

def _evaluate_fib(latest, cfg):
    price = latest["Close"]
    proximity = cfg.get("proximity_pct", 2.0) / 100
    levels = {
        "0%":    latest.get("Fib_0"),
        "23.6%": latest.get("Fib_236"),
        "38.2%": latest.get("Fib_382"),
        "50.0%": latest.get("Fib_500"),
        "61.8%": latest.get("Fib_618"),
        "100%":  latest.get("Fib_100"),
    }
    for label, level in levels.items():
        if _nan(level) or level == 0:
            continue
        if abs(price - level) / level <= proximity:
            if label in ("0%", "23.6%", "38.2%"):
                return "BUY", f"Price {price:.2f} near Fib support {label} ({level:.2f})"
            elif label in ("61.8%", "100%"):
                return "SELL", f"Price {price:.2f} near Fib resistance {label} ({level:.2f})"
            return "HOLD", f"Price {price:.2f} near Fib 50% ({level:.2f})"
    return "HOLD", f"Price {price:.2f} not near any Fibonacci level"


def _evaluate_pivot(latest, cfg):
    price = latest["Close"]
    proximity = cfg.get("proximity_pct", 0.5) / 100
    pp  = latest.get("PP")
    r1  = latest.get("PP_R1")
    r2  = latest.get("PP_R2")
    s1  = latest.get("PP_S1")
    s2  = latest.get("PP_S2")
    if _nan(pp):
        return "HOLD", "Pivot Points not available"
    for label, level in [("R2", r2), ("R1", r1)]:
        if not _nan(level) and abs(price - level) / level <= proximity:
            return "SELL", f"Price {price:.2f} near Pivot {label} {level:.2f} — Resistance"
    for label, level in [("S1", s1), ("S2", s2)]:
        if not _nan(level) and abs(price - level) / level <= proximity:
            return "BUY", f"Price {price:.2f} near Pivot {label} {level:.2f} — Support"
    if not _nan(pp) and abs(price - pp) / pp <= proximity:
        return "HOLD", f"Price {price:.2f} near Pivot PP {pp:.2f}"
    if price > (r1 or price):
        return "BUY", f"Price {price:.2f} above R1 {r1:.2f} — Breakout"
    if price < (s1 or price):
        return "SELL", f"Price {price:.2f} below S1 {s1:.2f} — Breakdown"
    return "HOLD", f"Price {price:.2f} between S1 {s1:.2f} and R1 {r1:.2f}"


# ─── Candlestick evaluators ───────────────────────────────────────────────────

def _evaluate_doji(latest, cfg):
    if latest.get("Doji", 0):
        return "HOLD", "Doji — market indecision, wait for confirmation"
    return "HOLD", "No Doji detected"


def _evaluate_hammer(latest, cfg):
    if latest.get("Hammer", 0):
        return "BUY", "Hammer candle — potential bullish reversal"
    return "HOLD", "No Hammer detected"


def _evaluate_shooting_star(latest, cfg):
    if latest.get("ShootingStar", 0):
        return "SELL", "Shooting Star — potential bearish reversal"
    return "HOLD", "No Shooting Star detected"


def _evaluate_engulfing(latest, cfg):
    if latest.get("BullEngulf", 0):
        return "BUY", "Bullish Engulfing — strong reversal signal"
    if latest.get("BearEngulf", 0):
        return "SELL", "Bearish Engulfing — strong reversal signal"
    return "HOLD", "No Engulfing pattern detected"


# ─── Risk evaluators ─────────────────────────────────────────────────────────

def _evaluate_sharpe(latest, cfg):
    sharpe = latest.get("Sharpe")
    if _nan(sharpe):
        return "HOLD", "Sharpe not available (need 252+ bars)"
    min_s = cfg.get("min", 0.5)
    if sharpe >= min_s:
        return "BUY", f"Sharpe={sharpe:.2f} ≥ {min_s} — Good risk-adjusted return"
    elif sharpe < 0:
        return "SELL", f"Sharpe={sharpe:.2f} — Negative risk-adjusted return"
    return "HOLD", f"Sharpe={sharpe:.2f} — Below threshold ({min_s})"


def _evaluate_var(latest, cfg):
    var = latest.get("VaR_95")
    if _nan(var):
        return "HOLD", "VaR not available (need 252+ bars)"
    max_loss = cfg.get("max_loss_pct", -2.0)
    var_pct = var * 100
    if var_pct > max_loss:
        return "BUY", f"VaR-95%={var_pct:.2f}% — Within acceptable risk ({max_loss}%)"
    return "SELL", f"VaR-95%={var_pct:.2f}% — Exceeds risk threshold ({max_loss}%)"


def _evaluate_max_drawdown(latest, cfg):
    dd = latest.get("Max_Drawdown")
    if _nan(dd):
        return "HOLD", "Max Drawdown not available"
    max_pct = cfg.get("max_pct", -20)
    if dd > max_pct:
        return "BUY", f"Max Drawdown={dd:.1f}% — Within tolerance ({max_pct}%)"
    return "SELL", f"Max Drawdown={dd:.1f}% — Exceeds threshold ({max_pct}%)"


# ─── Fundamental evaluators (need stock_info) ─────────────────────────────────

def _evaluate_pe(latest, cfg, stock_info=None):
    pe = (stock_info or {}).get("trailingPE") or (stock_info or {}).get("forwardPE")
    if _nan(pe):
        return "HOLD", "P/E not available from Yahoo Finance"
    max_pe = cfg.get("max", 25)
    if pe < max_pe:
        return "BUY", f"P/E={pe:.1f} < {max_pe} — Reasonable valuation"
    elif pe > max_pe * 2:
        return "SELL", f"P/E={pe:.1f} — Significantly overvalued (>{max_pe*2:.0f})"
    return "HOLD", f"P/E={pe:.1f} — Above threshold ({max_pe}) but not extreme"


def _evaluate_roe(latest, cfg, stock_info=None):
    roe = (stock_info or {}).get("returnOnEquity")
    if _nan(roe):
        return "HOLD", "ROE not available from Yahoo Finance"
    roe_pct = roe * 100
    min_roe = cfg.get("min", 15)
    if roe_pct >= min_roe:
        return "BUY", f"ROE={roe_pct:.1f}% ≥ {min_roe}% — Strong returns"
    return "SELL", f"ROE={roe_pct:.1f}% < {min_roe}% — Weak returns"


def _evaluate_revenue_growth(latest, cfg, stock_info=None):
    rg = (stock_info or {}).get("revenueGrowth")
    if _nan(rg):
        return "HOLD", "Revenue Growth not available from Yahoo Finance"
    rg_pct = rg * 100
    min_rg = cfg.get("min", 10)
    if rg_pct >= min_rg:
        return "BUY", f"Revenue Growth={rg_pct:.1f}% ≥ {min_rg}% — Growing"
    elif rg_pct < 0:
        return "SELL", f"Revenue Growth={rg_pct:.1f}% — Declining revenue"
    return "HOLD", f"Revenue Growth={rg_pct:.1f}% — Below target ({min_rg}%)"


def _evaluate_debt_equity(latest, cfg, stock_info=None):
    de = (stock_info or {}).get("debtToEquity")
    if _nan(de):
        return "HOLD", "Debt/Equity not available from Yahoo Finance"
    max_de = cfg.get("max", 1.5)
    if de <= max_de:
        return "BUY", f"D/E={de:.2f} ≤ {max_de} — Conservative leverage"
    elif de > max_de * 2:
        return "SELL", f"D/E={de:.2f} — High leverage risk (>{max_de*2:.1f})"
    return "HOLD", f"D/E={de:.2f} — Above threshold ({max_de})"


def _evaluate_eps_growth(latest, cfg, stock_info=None):
    eg = (stock_info or {}).get("earningsGrowth")
    if _nan(eg):
        return "HOLD", "EPS Growth not available from Yahoo Finance"
    eg_pct = eg * 100
    min_eg = cfg.get("min", 10)
    if eg_pct >= min_eg:
        return "BUY", f"EPS Growth={eg_pct:.1f}% ≥ {min_eg}% — Earnings expanding"
    elif eg_pct < 0:
        return "SELL", f"EPS Growth={eg_pct:.1f}% — Earnings declining"
    return "HOLD", f"EPS Growth={eg_pct:.1f}% — Below target ({min_eg}%)"


# ─── Registries ───────────────────────────────────────────────────────────────

CRITERIA_EVALUATORS: dict[str, callable] = {
    # Trend
    "EMA20":        _evaluate_ema20,
    "EMA50":        _evaluate_ema50,
    "EMA100":       _evaluate_ema100,
    "EMA200":       _evaluate_ema200,
    "SMA50":        _evaluate_sma50,
    "SMA200":       _evaluate_sma200,
    "SUPERTREND":   _evaluate_supertrend,
    "ADX":          _evaluate_adx,
    "VWAP":         _evaluate_vwap,
    "ICHIMOKU":     _evaluate_ichimoku,
    # Momentum
    "RSI":          _evaluate_rsi,
    "MACD":         _evaluate_macd,
    "STOCHASTIC":   _evaluate_stochastic,
    "CCI":          _evaluate_cci,
    "WILLIAMS_R":   _evaluate_williams_r,
    # Volume
    "VOLUME_SMA":   _evaluate_volume_sma,
    "OBV":          _evaluate_obv,
    "MFI":          _evaluate_mfi,
    # Volatility
    "BOLLINGER":    _evaluate_bollinger,
    "ATR":          _evaluate_atr,
    "KELTNER":      _evaluate_keltner,
    # Support & Resistance
    "FIB":          _evaluate_fib,
    "PIVOT":        _evaluate_pivot,
    # Candlestick
    "DOJI":         _evaluate_doji,
    "HAMMER":       _evaluate_hammer,
    "SHOOTING_STAR":_evaluate_shooting_star,
    "ENGULFING":    _evaluate_engulfing,
    # Risk
    "SHARPE":       _evaluate_sharpe,
    "VAR":          _evaluate_var,
    "MAX_DRAWDOWN": _evaluate_max_drawdown,
    # Legacy aliases
    "EMA_Cross":    _evaluate_ema_cross,
    "Bollinger":    _evaluate_bollinger,
    "Volume":       _evaluate_volume_sma,
}

FUNDAMENTAL_EVALUATORS: dict[str, callable] = {
    "PE":             _evaluate_pe,
    "ROE":            _evaluate_roe,
    "REVENUE_GROWTH": _evaluate_revenue_growth,
    "DEBT_EQUITY":    _evaluate_debt_equity,
    "EPS_GROWTH":     _evaluate_eps_growth,
}


# ─── Post-analysis criteria ────────────────────────────────────────────────────

def evaluate_post_analysis_criteria(
    criteria_config: dict | None,
    ai_signal: str,
    ai_confidence: float,
    ai_proba: list,       # [sell_prob, hold_prob, buy_prob]
    sentiment_score: float,
    sentiment_label: str,
) -> dict:
    if not criteria_config:
        return {}
    signals = {}
    for key in POST_ANALYSIS_CRITERIA:
        cfg = criteria_config.get(key, {})
        if not cfg.get("enabled", False):
            continue
        if key == "NEWS_SENTIMENT":
            t = cfg.get("min_score", 0.05)
            if sentiment_score > t:
                signals[key] = {"signal": "BUY",  "reason": f"Sentiment {sentiment_score:.3f} → {sentiment_label}"}
            elif sentiment_score < -t:
                signals[key] = {"signal": "SELL", "reason": f"Sentiment {sentiment_score:.3f} → {sentiment_label}"}
            else:
                signals[key] = {"signal": "HOLD", "reason": f"Sentiment {sentiment_score:.3f} → Neutral"}
        elif key == "BUY_PROB":
            prob = ai_proba[2] * 100
            t = cfg.get("min", 50)
            signals[key] = {"signal": "BUY" if prob >= t else "HOLD",
                            "reason": f"AI BUY probability {prob:.1f}%"}
        elif key == "SELL_PROB":
            prob = ai_proba[0] * 100
            t = cfg.get("min", 50)
            signals[key] = {"signal": "SELL" if prob >= t else "HOLD",
                            "reason": f"AI SELL probability {prob:.1f}%"}
        elif key == "CONFIDENCE":
            t = cfg.get("min", 60)
            signals[key] = {
                "signal": ai_signal if ai_confidence >= t else "HOLD",
                "reason": f"AI confidence {ai_confidence:.1f}% (threshold {t}%)",
            }
    return signals


# ─── AND / OR / NOT Custom Rules Engine ───────────────────────────────────────

# Maps the human-readable indicator name → df column name
INDICATOR_COLUMN_MAP: dict[str, str] = {
    "PRICE": "Close", "CLOSE": "Close",
    "OPEN": "Open", "HIGH": "High", "LOW": "Low", "VOLUME": "Volume",
    "RSI": "RSI",
    "MACD": "MACD", "MACD_SIGNAL": "MACD_Signal", "MACD_HIST": "MACD_Hist",
    "EMA12": "EMA_12", "EMA20": "EMA_20", "EMA26": "EMA_26",
    "EMA50": "EMA_50", "EMA100": "EMA_100", "EMA200": "EMA_200",
    "SMA20": "SMA_20", "SMA50": "SMA_50", "SMA200": "SMA_200",
    "BB_UPPER": "BB_Upper", "BB_LOWER": "BB_Lower", "BB_MID": "BB_Mid",
    "ATR": "ATR", "ADX": "ADX", "DMP": "DMP", "DMN": "DMN",
    "SUPERTREND": "SuperTrend", "SUPERTREND_DIR": "SuperTrend_Dir",
    "VWAP": "VWAP",
    "ICH_TENKAN": "ICH_Tenkan", "ICH_KIJUN": "ICH_Kijun",
    "ICH_CLOUD_TOP": "ICH_CloudTop", "ICH_CLOUD_BOT": "ICH_CloudBot",
    "OBV": "OBV", "OBV_MA20": "OBV_MA20",
    "MFI": "MFI",
    "STOCH_K": "STOCH_K", "STOCH_D": "STOCH_D",
    "CCI": "CCI", "WILLIAMS_R": "WILLIAMS_R",
    "KC_UPPER": "KC_Upper", "KC_LOWER": "KC_Lower", "KC_MID": "KC_Mid",
    "PP": "PP", "PP_R1": "PP_R1", "PP_R2": "PP_R2",
    "PP_S1": "PP_S1", "PP_S2": "PP_S2",
    "FIB_236": "Fib_236", "FIB_382": "Fib_382", "FIB_500": "Fib_500",
    "FIB_618": "Fib_618",
    "VOLUME_MA20": "Volume_MA20",
    "SHARPE": "Sharpe", "VAR_95": "VaR_95", "MAX_DRAWDOWN": "Max_Drawdown",
}


def _resolve_value(val, latest: pd.Series):
    """Resolve a value that may be a number or an indicator name."""
    if isinstance(val, str):
        col = INDICATOR_COLUMN_MAP.get(val.upper(), val)
        raw = latest.get(col)
        if raw is None:
            return None
        try:
            f = float(raw)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _eval_leaf(node: dict, latest: pd.Series) -> bool:
    """Evaluate a single leaf condition."""
    indicator = node.get("indicator", "")
    col = INDICATOR_COLUMN_MAP.get(indicator.upper(), indicator)
    raw = latest.get(col)
    if raw is None:
        return False
    try:
        left = float(raw)
        if math.isnan(left):
            return False
    except (TypeError, ValueError):
        return False

    right = _resolve_value(node.get("value"), latest)
    if right is None:
        return False

    op = node.get("op", ">")
    ops = {
        "<": left < right, ">": left > right,
        "<=": left <= right, ">=": left >= right,
        "==": abs(left - right) < 1e-9,
        "!=": abs(left - right) >= 1e-9,
    }
    result = ops.get(op, False)
    return not result if node.get("not", False) else result


def _eval_tree(node: dict, latest: pd.Series) -> bool:
    """Recursively evaluate AND/OR/NOT rule tree."""
    negate = node.get("not", False)

    if "operator" in node:
        operator = node["operator"].upper()
        children = node.get("conditions", [])
        if operator == "AND":
            result = all(_eval_tree(c, latest) for c in children)
        elif operator == "OR":
            result = any(_eval_tree(c, latest) for c in children)
        else:
            result = False
    elif "indicator" in node:
        result = _eval_leaf(node, latest)
    else:
        result = False

    return not result if negate else result


def evaluate_custom_rules(df: pd.DataFrame, custom_rules: dict | None) -> dict:
    """
    Evaluate a custom AND/OR/NOT rule tree against the latest row.

    Rule format:
      {
        "operator": "AND",          // AND | OR (group node)
        "not": false,               // negate the whole group
        "signal_if_match": "BUY",  // BUY | SELL | HOLD
        "conditions": [
          {
            "indicator": "RSI",     // leaf condition
            "op": "<",              // < > <= >= == !=
            "value": 30,            // number OR indicator name e.g. "EMA50"
            "not": false
          },
          {
            "operator": "OR",       // nested group
            "conditions": [...]
          }
        ]
      }
    """
    if not custom_rules or not custom_rules.get("conditions"):
        return {}
    latest = df.iloc[-1]
    try:
        matched = _eval_tree(custom_rules, latest)
        signal_match = custom_rules.get("signal_if_match", "BUY")
        signal_no_match = custom_rules.get("signal_if_no_match", "HOLD")
        return {
            "matched": matched,
            "signal": signal_match if matched else signal_no_match,
            "reason": f"Custom rule {'MATCHED → ' + signal_match if matched else 'NOT matched → ' + signal_no_match}",
            "operator": custom_rules.get("operator", "AND"),
            "conditions_count": len(custom_rules.get("conditions", [])),
        }
    except Exception as exc:
        return {
            "matched": False,
            "signal": "HOLD",
            "reason": f"Custom rule error: {exc}",
        }


# ─── Main criteria runner ──────────────────────────────────────────────────────

def custom_criteria(
    df: pd.DataFrame,
    criteria_config: dict | None = None,
    stock_info: dict | None = None,
) -> tuple[dict, dict]:
    """
    Run the full rule-based signal engine on the latest row.

    criteria_config: {indicator_key: {enabled, ...thresholds}}
    stock_info: Yahoo Finance info dict (for fundamental criteria)
    """
    latest = df.iloc[-1]
    signals: dict = {}

    # Build effective config: defaults → overlaid with user config
    config: dict = {k: v.copy() for k, v in DEFAULT_CRITERIA_CONFIG.items()}
    if criteria_config:
        for key, user_cfg in criteria_config.items():
            if key in POST_ANALYSIS_CRITERIA:
                continue
            if key in config:
                config[key].update(user_cfg)
            else:
                config[key] = {"enabled": True, **user_cfg}

    # Evaluate df-based criteria
    for name, evaluator in CRITERIA_EVALUATORS.items():
        if not config.get(name, {}).get("enabled", True):
            continue
        try:
            signals[name] = evaluator(latest, config.get(name, {}))
        except Exception as exc:
            signals[name] = ("HOLD", f"Error: {exc}")

    # Evaluate fundamental criteria (needs stock_info)
    for name, evaluator in FUNDAMENTAL_EVALUATORS.items():
        if not config.get(name, {}).get("enabled", True):
            continue
        try:
            signals[name] = evaluator(latest, config.get(name, {}), stock_info)
        except Exception as exc:
            signals[name] = ("HOLD", f"Error: {exc}")

    buy  = sum(1 for s, _ in signals.values() if s == "BUY")
    sell = sum(1 for s, _ in signals.values() if s == "SELL")
    hold = sum(1 for s, _ in signals.values() if s == "HOLD")

    deferred = [k for k in (criteria_config or {}) if k in POST_ANALYSIS_CRITERIA]

    summary = {
        "signals": {k: {"signal": s, "reason": r} for k, (s, r) in signals.items()},
        "buy_count": buy,
        "sell_count": sell,
        "hold_count": hold,
        "active_criteria": list(signals.keys()),
        "available_criteria": AVAILABLE_CRITERIA,
        "deferred_post_analysis": deferred,
    }
    return signals, summary
