import pandas as pd


def compute_trade_levels(df: pd.DataFrame, ai_signal: str) -> dict:
    """
    Compute entry, stop-loss, and target levels based on ATR and current price.
    Uses ATR-based position sizing for risk management.
    """
    latest = df.iloc[-1]
    price = float(latest["Close"])
    atr = float(latest["ATR"])

    if ai_signal == "BUY":
        entry = round(price, 2)
        stop_loss = round(price - 2 * atr, 2)
        target_1 = round(price + 2 * atr, 2)
        target_2 = round(price + 4 * atr, 2)
    elif ai_signal == "SELL":
        entry = round(price, 2)
        stop_loss = round(price + 2 * atr, 2)
        target_1 = round(price - 2 * atr, 2)
        target_2 = round(price - 4 * atr, 2)
    else:
        entry = round(price, 2)
        stop_loss = round(price - 1.5 * atr, 2)
        target_1 = round(price + 1.5 * atr, 2)
        target_2 = round(price + 3 * atr, 2)

    risk_reward = round(abs(target_1 - entry) / max(abs(entry - stop_loss), 0.01), 2)

    return {
        "entry": entry,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "atr": round(atr, 2),
        "risk_reward_ratio": risk_reward,
    }
