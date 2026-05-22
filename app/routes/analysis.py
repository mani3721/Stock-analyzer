import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, MAX_CUSTOM_WEBSITES
from app.utils.sse import sse_event
from app.services.stock_data import fetch_stock_data
from app.services.indicators import compute_indicators
from app.services.news_analyzer import analyze_all_sources
from app.services.criteria import custom_criteria, AVAILABLE_CRITERIA
from app.services.ai_engine import train_ai_engine
from app.services.trade_levels import compute_trade_levels
from app.services.explanation import ai_explanation_engine

router = APIRouter()


async def run_analysis(
    symbol: str,
    sector: str,
    timeframe: str,
    interval: str,
    websites: list[str],
    criteria_config: dict | None = None,
    model: str = "",
) -> AsyncGenerator[str, None]:
    """
    Generator that yields one SSE event per step.
    Each step: yields "running" -> executes -> yields "done" with data.
    On error: yields "error" and stops the pipeline.
    """
    state: dict = {}

    try:
        # Step 0 — Fetch stock data
        yield sse_event(0, "running", "Fetching stock data")
        df, stock_info, price_summary = await asyncio.to_thread(
            fetch_stock_data, symbol, timeframe, interval
        )
        state["df"] = df
        state["stock_info"] = stock_info
        yield sse_event(0, "done", "Fetching stock data", price_summary)

        # Step 1 — Compute technical indicators
        yield sse_event(1, "running", "Computing technical indicators")
        df, indicators_summary = await asyncio.to_thread(compute_indicators, df)
        state["df"] = df
        yield sse_event(1, "done", "Computing technical indicators", indicators_summary)

        # Step 2 — Custom criteria / rule-based signals
        yield sse_event(2, "running", "Running custom criteria engine")
        criteria_signals, criteria_summary = await asyncio.to_thread(
            custom_criteria, df, criteria_config
        )
        state["criteria_signals"] = criteria_signals
        yield sse_event(2, "done", "Running custom criteria engine", criteria_summary)

        # Step 3 — AI scoring (Random Forest)
        yield sse_event(3, "running", "Training AI scoring model")
        ai_signal, ai_confidence, ai_proba, ai_summary = await asyncio.to_thread(
            train_ai_engine, df
        )
        state["ai_signal"] = ai_signal
        state["ai_confidence"] = ai_confidence
        state["ai_proba"] = ai_proba
        yield sse_event(3, "done", "Training AI scoring model", ai_summary)

        # Step 4 — Compute trade levels
        yield sse_event(4, "running", "Computing trade levels")
        trade_levels = await asyncio.to_thread(compute_trade_levels, df, ai_signal)
        state["trade_levels"] = trade_levels
        yield sse_event(4, "done", "Computing trade levels", trade_levels)

        # Step 5 — Scan custom websites
        yield sse_event(5, "running", "Scanning custom websites")
        sentiment_score, sentiment_label, raw_sentiment = await asyncio.to_thread(
            analyze_all_sources, symbol, websites
        )
        state["sentiment_score"] = sentiment_score
        state["sentiment_label"] = sentiment_label
        state["raw_sentiment"] = raw_sentiment

        sentiment_summary = {
            "overall_score": sentiment_score,
            "overall_label": sentiment_label,
            "sources_count": len(raw_sentiment),
            "sources": [
                {
                    "source": r["source"],
                    "score": r["score"],
                    "label": r["label"],
                    "url": r["url"],
                    "success": r["success"],
                }
                for r in raw_sentiment
            ],
        }
        yield sse_event(5, "done", "Scanning custom websites", sentiment_summary)

        # Step 6 — AI explanation engine
        ai_model = model or OPENROUTER_MODEL
        yield sse_event(6, "running", f"Generating AI explanation ({ai_model})")
        explanation_result = await asyncio.to_thread(
            ai_explanation_engine,
            symbol, sector, df, ai_signal, ai_confidence, ai_proba,
            criteria_signals, sentiment_score, sentiment_label,
            trade_levels, raw_sentiment, OPENROUTER_API_KEY, ai_model,
        )
        explanation_result["model_used"] = ai_model
        yield sse_event(6, "done", "Generating AI explanation", explanation_result)

        # Final — send complete event with full summary
        summary = {
            "symbol": symbol,
            "sector": sector,
            "price": state.get("df", df).iloc[-1]["Close"],
            "signal": ai_signal,
            "confidence": ai_confidence,
            "probabilities": {
                "buy": round(ai_proba[2] * 100, 1),
                "hold": round(ai_proba[1] * 100, 1),
                "sell": round(ai_proba[0] * 100, 1),
            },
            "trade_levels": trade_levels,
            "sentiment": {
                "score": sentiment_score,
                "label": sentiment_label,
            },
            "criteria": criteria_summary,
            "indicators": indicators_summary,
            "explanation": explanation_result,
        }
        yield sse_event(7, "complete", "Analysis complete", summary)

    except Exception as exc:
        yield sse_event(-1, "error", "Pipeline failed", {"error": str(exc)})


@router.get("/analyze")
async def analyze(
    symbol: str = Query(..., description="Yahoo Finance ticker, e.g. RELIANCE.NS"),
    sector: str = Query("", description="Sector label (informational)"),
    timeframe: str = Query("1y", description="yfinance period: 1mo 3mo 6mo 1y 3y 5y"),
    interval: str = Query("1d", description="yfinance interval: 1d 1wk 1mo"),
    websites: str = Query("", description="Comma-separated domain list"),
    criteria: str = Query("", description='JSON criteria config, e.g. {"RSI":{"enabled":true,"oversold":25},"MACD":{"enabled":false}}'),
    model: str = Query("", description="OpenRouter model name, e.g. nvidia/nemotron-3-super-120b-a12b:free"),
):
    """
    Stream multi-step stock analysis as Server-Sent Events.

    Each SSE message is JSON:
        { step, status: "running"|"done"|"error"|"complete", title, data }

    Parameters:
        - criteria: JSON to select/configure indicators (RSI, MACD, SMA50, EMA_Cross, Bollinger, Volume)
        - model: OpenRouter model name for AI explanation (defaults to env OPENROUTER_MODEL)
    """
    site_list = [s.strip() for s in websites.split(",") if s.strip()][:MAX_CUSTOM_WEBSITES]

    criteria_config = None
    if criteria:
        try:
            criteria_config = json.loads(criteria)
        except json.JSONDecodeError:
            pass

    return StreamingResponse(
        run_analysis(symbol, sector, timeframe, interval, site_list, criteria_config, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/criteria")
async def list_criteria():
    """List all available criteria and their default configuration."""
    from app.services.criteria import DEFAULT_CRITERIA_CONFIG
    return {
        "available_criteria": AVAILABLE_CRITERIA,
        "defaults": DEFAULT_CRITERIA_CONFIG,
    }
