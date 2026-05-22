import pandas as pd
import pandas_ta as ta


def compute_indicators(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute all technical indicators and return enriched DataFrame + latest values."""
    df["RSI"] = ta.rsi(df["Close"], length=14)

    macd = ta.macd(df["Close"])
    df["MACD"] = macd["MACD_12_26_9"]
    df["MACD_Signal"] = macd["MACDs_12_26_9"]
    df["MACD_Hist"] = macd["MACDh_12_26_9"]

    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["EMA_12"] = ta.ema(df["Close"], length=12)
    df["EMA_26"] = ta.ema(df["Close"], length=26)

    bb = ta.bbands(df["Close"], length=20)
    df["BB_Upper"] = bb[[c for c in bb.columns if "BBU" in c][0]]
    df["BB_Lower"] = bb[[c for c in bb.columns if "BBL" in c][0]]
    df["BB_Mid"] = bb[[c for c in bb.columns if "BBM" in c][0]]

    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["Volume_MA20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Spike"] = (df["Volume"] > df["Volume_MA20"] * 1.5).astype(int)
    df["Daily_Return"] = df["Close"].pct_change()
    df["Cum_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

    df.dropna(inplace=True)

    latest = df.iloc[-1]
    summary = {
        "rsi": round(float(latest["RSI"]), 2),
        "macd": round(float(latest["MACD"]), 4),
        "macd_signal": round(float(latest["MACD_Signal"]), 4),
        "macd_hist": round(float(latest["MACD_Hist"]), 4),
        "sma_20": round(float(latest["SMA_20"]), 2),
        "sma_50": round(float(latest["SMA_50"]), 2),
        "ema_12": round(float(latest["EMA_12"]), 2),
        "ema_26": round(float(latest["EMA_26"]), 2),
        "bb_upper": round(float(latest["BB_Upper"]), 2),
        "bb_lower": round(float(latest["BB_Lower"]), 2),
        "bb_mid": round(float(latest["BB_Mid"]), 2),
        "atr": round(float(latest["ATR"]), 2),
        "volume_ma20": int(latest["Volume_MA20"]),
        "volume_spike": bool(latest["Volume_Spike"]),
        "cum_return_pct": round(float(latest["Cum_Return"]) * 100, 2),
    }

    return df, summary
