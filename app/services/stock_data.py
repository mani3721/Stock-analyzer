import yfinance as yf
import pandas as pd


def fetch_stock_data(symbol: str, period: str, interval: str) -> tuple[pd.DataFrame, dict]:
    """Download OHLCV history and company info from Yahoo Finance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(f"No price data found for {symbol}")

    info = ticker.info
    company = info.get("longName", symbol)
    price = info.get("currentPrice", float(df["Close"].iloc[-1]))

    summary = {
        "rows": len(df),
        "company": company,
        "price": round(price, 2),
        "latest_open": round(float(df["Open"].iloc[-1]), 2),
        "latest_high": round(float(df["High"].iloc[-1]), 2),
        "latest_low": round(float(df["Low"].iloc[-1]), 2),
        "latest_close": round(float(df["Close"].iloc[-1]), 2),
        "latest_volume": int(df["Volume"].iloc[-1]),
    }

    return df, info, summary
