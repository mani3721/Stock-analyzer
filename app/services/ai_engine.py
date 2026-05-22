import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "ATR", "Volume_Spike", "BB_Upper", "BB_Lower", "Daily_Return",
]


def train_ai_engine(df: pd.DataFrame) -> tuple[str, float, list[float], dict]:
    """
    Train a Random Forest classifier on technical indicators.
    Labels: 5-day forward return bucketed into SELL (<-2%), HOLD (-2% to +2%), BUY (>+2%).
    Returns (signal, confidence, probabilities, summary).
    """
    df_ml = df[FEATURE_COLS].copy()

    future_return = df["Close"].shift(-5) / df["Close"] - 1
    df_ml["Label"] = pd.cut(
        future_return,
        bins=[-np.inf, -0.02, 0.02, np.inf],
        labels=[0, 1, 2],
    ).astype(float)
    df_ml.dropna(inplace=True)

    X = df_ml[FEATURE_COLS].values
    y = df_ml["Label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X_train_s, y_train)

    accuracy = float((clf.predict(X_test_s) == y_test).mean())

    latest_scaled = scaler.transform(df[FEATURE_COLS].iloc[-1:].values)
    prediction = clf.predict(latest_scaled)[0]
    probabilities = clf.predict_proba(latest_scaled)[0]

    signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    signal = signal_map[int(prediction)]
    confidence = float(max(probabilities)) * 100

    summary = {
        "signal": signal,
        "confidence": round(confidence, 1),
        "accuracy": round(accuracy * 100, 1),
        "probabilities": {
            "buy": round(float(probabilities[2]) * 100, 1),
            "hold": round(float(probabilities[1]) * 100, 1),
            "sell": round(float(probabilities[0]) * 100, 1),
        },
    }

    return signal, confidence, probabilities.tolist(), summary
