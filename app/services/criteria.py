import pandas as pd


def custom_criteria(df: pd.DataFrame) -> tuple[dict, dict]:
    """
    Run rule-based signal engine on the latest row.
    Returns (signals_dict, summary_dict).
    Each signal: (signal_str, reason_str)
    """
    latest = df.iloc[-1]
    signals = {}

    rsi = latest["RSI"]
    if rsi < 30:
        signals["RSI"] = ("BUY", f"RSI={rsi:.1f} — Oversold")
    elif rsi > 70:
        signals["RSI"] = ("SELL", f"RSI={rsi:.1f} — Overbought")
    else:
        signals["RSI"] = ("HOLD", f"RSI={rsi:.1f} — Neutral")

    if latest["MACD"] > latest["MACD_Signal"]:
        signals["MACD"] = ("BUY", "MACD bullish crossover")
    else:
        signals["MACD"] = ("SELL", "MACD bearish crossover")

    if latest["Close"] > latest["SMA_50"]:
        signals["SMA50"] = ("BUY", f"Price ${latest['Close']:.2f} > SMA50 ${latest['SMA_50']:.2f} — Uptrend")
    else:
        signals["SMA50"] = ("SELL", f"Price ${latest['Close']:.2f} < SMA50 ${latest['SMA_50']:.2f} — Downtrend")

    if latest["EMA_12"] > latest["EMA_26"]:
        signals["EMA_Cross"] = ("BUY", "EMA12 > EMA26 — Golden cross")
    else:
        signals["EMA_Cross"] = ("SELL", "EMA12 < EMA26 — Death cross")

    if latest["Close"] < latest["BB_Lower"]:
        signals["Bollinger"] = ("BUY", "Price below Lower Band — Oversold")
    elif latest["Close"] > latest["BB_Upper"]:
        signals["Bollinger"] = ("SELL", "Price above Upper Band — Overbought")
    else:
        signals["Bollinger"] = ("HOLD", "Price within Bollinger Bands")

    vr = latest["Volume"] / latest["Volume_MA20"]
    if vr > 1.5:
        signals["Volume"] = ("BUY", f"Volume spike {vr:.1f}x — Strong buying")
    else:
        signals["Volume"] = ("HOLD", f"Volume {vr:.1f}x — Normal")

    buy_count = sum(1 for s, _ in signals.values() if s == "BUY")
    sell_count = sum(1 for s, _ in signals.values() if s == "SELL")
    hold_count = sum(1 for s, _ in signals.values() if s == "HOLD")

    summary = {
        "signals": {k: {"signal": s, "reason": r} for k, (s, r) in signals.items()},
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
    }

    return signals, summary
