import json

import requests

from app.config import AI_REQUEST_TIMEOUT


def call_openrouter_ai(prompt: str, api_key: str, model: str = "nvidia/nemotron-3-super-120b-a12b:free") -> dict:
    """Call OpenRouter API for AI-generated explanations."""
    try:
        resp = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://ai-stock-analysis.app",
                "X-Title": "AI Stock Investment Platform",
            },
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "reasoning": {"enabled": True},
                "max_tokens": 1200,
                "temperature": 0.3,
            }),
            timeout=AI_REQUEST_TIMEOUT,
        )
        result = resp.json()
        if "choices" in result and result["choices"]:
            msg = result["choices"][0]["message"]
            content = msg.get("content") or ""
            reasoning = ""
            if msg.get("reasoning_details"):
                for rd in msg["reasoning_details"]:
                    if rd.get("type") == "thinking":
                        reasoning = rd.get("thinking", "")
            return {"content": content, "reasoning": reasoning, "success": True}
        else:
            return {"content": str(result), "reasoning": "", "success": False}
    except Exception as e:
        return {"content": "", "reasoning": "", "success": False, "error": str(e)}


def build_ai_prompt(
    symbol: str, df, ai_signal: str, ai_confidence: float,
    ai_proba: list, criteria_signals: dict, sentiment_score: float,
    sentiment_label: str, trade_levels: dict, raw_sentiment_data: list,
    sector: str = "",
) -> str:
    """Build the analysis prompt to send to the AI model."""
    latest = df.iloc[-1]
    period_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
    buy_c = sum(1 for s, _ in criteria_signals.values() if s == "BUY")
    sell_c = sum(1 for s, _ in criteria_signals.values() if s == "SELL")
    hold_c = sum(1 for s, _ in criteria_signals.values() if s == "HOLD")

    signals_text = ""
    for k, (s, r) in criteria_signals.items():
        signals_text += f"  - {k}: {s} — {r}\n"

    sources_text = ""
    for r in raw_sentiment_data:
        sources_text += f"  - {r['source']}: {r['score']:.3f} ({r['label']})\n"

    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(ai_signal, "⚪")

    return f"""You are a professional stock market analyst AI. Analyze the following data for {symbol} ({sector} sector).

=== STOCK DATA ===
Symbol: {symbol} | Sector: {sector}
Current Price: ${latest['Close']:.2f}
Period Return: {period_return:.1f}%

=== TECHNICAL INDICATORS ===
RSI (14): {latest['RSI']:.1f}  {"→ Oversold (bullish)" if latest['RSI'] < 30 else "→ Overbought (bearish)" if latest['RSI'] > 70 else "→ Neutral"}
MACD: {latest['MACD']:.4f} | Signal: {latest['MACD_Signal']:.4f}  {"→ Bullish crossover" if latest['MACD'] > latest['MACD_Signal'] else "→ Bearish crossover"}
SMA20: ${latest['SMA_20']:.2f} | SMA50: ${latest['SMA_50']:.2f}  {"→ Above both (uptrend)" if latest['Close'] > latest['SMA_50'] else "→ Below SMA50 (downtrend)"}
EMA12: ${latest['EMA_12']:.2f} | EMA26: ${latest['EMA_26']:.2f}  {"→ Golden cross" if latest['EMA_12'] > latest['EMA_26'] else "→ Death cross"}
Bollinger Upper: ${latest['BB_Upper']:.2f} | Lower: ${latest['BB_Lower']:.2f}
ATR (volatility): {latest['ATR']:.2f}

=== RULE-BASED SIGNALS ({buy_c} BUY | {sell_c} SELL | {hold_c} HOLD) ===
{signals_text}
=== NEWS & WEBSITE SENTIMENT ===
Overall Score: {sentiment_score:.3f} → {sentiment_label}
Sources analyzed:
{sources_text}
=== AI MODEL (Random Forest) ===
Prediction: {ai_signal}
Confidence: {ai_confidence:.1f}%
BUY probability: {ai_proba[2] * 100:.1f}%
HOLD probability: {ai_proba[1] * 100:.1f}%
SELL probability: {ai_proba[0] * 100:.1f}%

=== ENTRY / EXIT LEVELS ===
Entry: ${trade_levels['entry']:.2f}
Stop Loss: ${trade_levels['stop_loss']:.2f}
Target 1: ${trade_levels['target_1']:.2f}
Target 2: ${trade_levels['target_2']:.2f}

=== YOUR TASK ===
The final signal is **{ai_signal}**. ONLY explain why {ai_signal} is the right call.
Do NOT explain other signals. Do NOT include WHY BUY, WHY SELL, WHY HOLD sections separately.

Provide your analysis in these EXACT sections:

## {signal_emoji} RECOMMENDATION: {ai_signal}
[One sentence summary of why {ai_signal}]

## 📋 WHY {ai_signal} — Detailed Reasoning
[5-7 bullet points explaining why {ai_signal} is the correct call right now]
[Reference specific numbers from indicators, sentiment, and AI model]

## 📰 SUPPORTING SOURCES
[List each sentiment source with its score and how it supports the {ai_signal} decision]

## 📊 KEY LEVELS TO WATCH
[Entry, stop loss, targets with brief reasoning]

## ⚠️ RISKS TO THIS {ai_signal} CALL
[2-3 specific risks that could invalidate the {ai_signal} recommendation]

Be specific, reference actual numbers. Educational purposes only.
"""


def generate_fallback_explanation(
    ai_signal: str, ai_confidence: float, ai_proba: list,
    criteria_signals: dict, sentiment_score: float, sentiment_label: str,
    trade_levels: dict, raw_sentiment_data: list, df=None,
) -> str:
    """Rule-based explanation when OpenRouter API is unavailable."""
    latest = df.iloc[-1] if df is not None else None
    signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(ai_signal, "⚪")

    supporting = [(k, s, r) for k, (s, r) in criteria_signals.items() if s == ai_signal]
    other = [(k, s, r) for k, (s, r) in criteria_signals.items() if s != ai_signal]

    bullets = ""
    for k, s, r in supporting:
        bullets += f"- **{k}** confirms {ai_signal}: {r}\n"
    if not supporting:
        bullets += f"- Mixed signals across indicators suggest caution\n"

    prob_idx = {"BUY": 2, "SELL": 0, "HOLD": 1}[ai_signal]
    bullets += f"- AI Model assigns {ai_signal} probability: {ai_proba[prob_idx] * 100:.1f}%\n"
    bullets += f"- News sentiment is {sentiment_label} ({sentiment_score:.3f}) across {len(raw_sentiment_data)} sources\n"

    if latest is not None:
        bullets += f"- RSI at {latest['RSI']:.1f} — {'Oversold' if latest['RSI'] < 30 else 'Overbought' if latest['RSI'] > 70 else 'Neutral zone'}\n"
        bullets += f"- MACD {'bullish' if latest['MACD'] > latest['MACD_Signal'] else 'bearish'} crossover (MACD={latest['MACD']:.4f} vs Signal={latest['MACD_Signal']:.4f})\n"

    sources_bullets = ""
    for r in raw_sentiment_data:
        sources_bullets += f"- {r['source']}: score {r['score']:.3f} ({r['label']})\n"

    risk_bullets = ""
    for k, s, r in other:
        risk_bullets += f"- **{k}** shows {s}: {r}\n"
    if not risk_bullets:
        risk_bullets = "- Market-wide volatility can override signals\n"
    risk_bullets += f"- Always use stop loss at ${trade_levels['stop_loss']:.2f}\n"

    return f"""## {signal_emoji} RECOMMENDATION: {ai_signal}
Model confidence {ai_confidence:.1f}% — {ai_signal} is the strongest signal based on combined analysis.

## 📋 WHY {ai_signal} — Detailed Reasoning
{bullets}
## 📰 SUPPORTING SOURCES
{sources_bullets}
## 📊 KEY LEVELS TO WATCH
- Entry: ${trade_levels['entry']:.2f} | Stop: ${trade_levels['stop_loss']:.2f}
- Target 1: ${trade_levels['target_1']:.2f} | Target 2: ${trade_levels['target_2']:.2f}

## ⚠️ RISKS TO THIS {ai_signal} CALL
{risk_bullets}- This is for educational purposes only — DYOR"""


def ai_explanation_engine(
    symbol: str, df, ai_signal: str, ai_confidence: float,
    ai_proba: list, criteria_signals: dict, sentiment_score: float,
    sentiment_label: str, trade_levels: dict, raw_sentiment_data: list,
    api_key: str, model: str = "nvidia/nemotron-3-super-120b-a12b:free",
    sector: str = "",
) -> dict:
    """
    Generate AI explanation — tries OpenRouter first, falls back to rule-based.
    Returns dict with explanation text and metadata.
    """
    ai_explanation = ""
    ai_reasoning = ""
    source = "fallback"

    if api_key and "PASTE" not in api_key:
        prompt = build_ai_prompt(
            symbol, df, ai_signal, ai_confidence, ai_proba,
            criteria_signals, sentiment_score, sentiment_label,
            trade_levels, raw_sentiment_data, sector,
        )
        result = call_openrouter_ai(prompt, api_key, model=model)
        if result["success"] and result["content"]:
            ai_explanation = result["content"]
            ai_reasoning = result.get("reasoning", "")
            source = "openrouter"

    if not ai_explanation:
        ai_explanation = generate_fallback_explanation(
            ai_signal, ai_confidence, ai_proba,
            criteria_signals, sentiment_score, sentiment_label,
            trade_levels, raw_sentiment_data, df,
        )
        source = "fallback"

    return {
        "explanation": ai_explanation,
        "reasoning": ai_reasoning,
        "source": source,
    }
