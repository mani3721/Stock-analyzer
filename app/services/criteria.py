import pandas as pd

# All available criteria with their default thresholds
DEFAULT_CRITERIA_CONFIG = {
    "RSI": {
        "enabled": True,
        "oversold": 30,
        "overbought": 70,
    },
    "MACD": {
        "enabled": True,
    },
    "SMA50": {
        "enabled": True,
    },
    "EMA_Cross": {
        "enabled": True,
    },
    "Bollinger": {
        "enabled": True,
    },
    "Volume": {
        "enabled": True,
        "spike_threshold": 1.5,
    },
}

AVAILABLE_CRITERIA = list(DEFAULT_CRITERIA_CONFIG.keys())


def _evaluate_rsi(latest: pd.Series, config: dict) -> tuple[str, str]:
    rsi = latest["RSI"]
    oversold = config.get("oversold", 30)
    overbought = config.get("overbought", 70)
    if rsi < oversold:
        return "BUY", f"RSI={rsi:.1f} — Oversold (below {oversold})"
    elif rsi > overbought:
        return "SELL", f"RSI={rsi:.1f} — Overbought (above {overbought})"
    return "HOLD", f"RSI={rsi:.1f} — Neutral ({oversold}-{overbought})"


def _evaluate_macd(latest: pd.Series, config: dict) -> tuple[str, str]:
    if latest["MACD"] > latest["MACD_Signal"]:
        return "BUY", "MACD bullish crossover"
    return "SELL", "MACD bearish crossover"


def _evaluate_sma50(latest: pd.Series, config: dict) -> tuple[str, str]:
    if latest["Close"] > latest["SMA_50"]:
        return "BUY", f"Price ${latest['Close']:.2f} > SMA50 ${latest['SMA_50']:.2f} — Uptrend"
    return "SELL", f"Price ${latest['Close']:.2f} < SMA50 ${latest['SMA_50']:.2f} — Downtrend"


def _evaluate_ema_cross(latest: pd.Series, config: dict) -> tuple[str, str]:
    if latest["EMA_12"] > latest["EMA_26"]:
        return "BUY", "EMA12 > EMA26 — Golden cross"
    return "SELL", "EMA12 < EMA26 — Death cross"


def _evaluate_bollinger(latest: pd.Series, config: dict) -> tuple[str, str]:
    if latest["Close"] < latest["BB_Lower"]:
        return "BUY", "Price below Lower Band — Oversold"
    elif latest["Close"] > latest["BB_Upper"]:
        return "SELL", "Price above Upper Band — Overbought"
    return "HOLD", "Price within Bollinger Bands"


def _evaluate_volume(latest: pd.Series, config: dict) -> tuple[str, str]:
    spike_threshold = config.get("spike_threshold", 1.5)
    vr = latest["Volume"] / latest["Volume_MA20"]
    if vr > spike_threshold:
        return "BUY", f"Volume spike {vr:.1f}x — Strong buying (>{spike_threshold}x)"
    return "HOLD", f"Volume {vr:.1f}x — Normal"


CRITERIA_EVALUATORS = {
    "RSI": _evaluate_rsi,
    "MACD": _evaluate_macd,
    "SMA50": _evaluate_sma50,
    "EMA_Cross": _evaluate_ema_cross,
    "Bollinger": _evaluate_bollinger,
    "Volume": _evaluate_volume,
}


def custom_criteria(
    df: pd.DataFrame,
    criteria_config: dict | None = None,
) -> tuple[dict, dict]:
    """
    Run rule-based signal engine on the latest row.
    criteria_config: optional dict to select criteria and customize thresholds.
      Example: {"RSI": {"enabled": true, "oversold": 25, "overbought": 75}, "MACD": {"enabled": true}}
      Pass None or empty to use all defaults.
    Returns (signals_dict, summary_dict).
    """
    latest = df.iloc[-1]
    signals = {}

    config = DEFAULT_CRITERIA_CONFIG.copy()
    if criteria_config:
        for key in criteria_config:
            if key in config:
                config[key].update(criteria_config[key])
            elif key in CRITERIA_EVALUATORS:
                config[key] = {"enabled": True, **criteria_config[key]}

    for name, evaluator in CRITERIA_EVALUATORS.items():
        criterion_cfg = config.get(name, {})
        if not criterion_cfg.get("enabled", True):
            continue
        signals[name] = evaluator(latest, criterion_cfg)

    buy_count = sum(1 for s, _ in signals.values() if s == "BUY")
    sell_count = sum(1 for s, _ in signals.values() if s == "SELL")
    hold_count = sum(1 for s, _ in signals.values() if s == "HOLD")

    summary = {
        "signals": {k: {"signal": s, "reason": r} for k, (s, r) in signals.items()},
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "active_criteria": list(signals.keys()),
        "available_criteria": AVAILABLE_CRITERIA,
    }

    return signals, summary
