import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.config import OPENROUTER_API_KEY, OPENROUTER_MODEL, MAX_CUSTOM_WEBSITES
from app.utils.sse import sse_event, SSE_STREAM_INIT, SSE_STREAM_CLOSE
from app.services.stock_data import fetch_stock_data
from app.services.indicators import compute_indicators
from app.services.news_analyzer import analyze_all_sources
from app.services.criteria import (
    custom_criteria, AVAILABLE_CRITERIA,
    evaluate_post_analysis_criteria, evaluate_custom_rules,
)
from app.services.ai_engine import train_ai_engine
from app.services.trade_levels import compute_trade_levels
from app.services.explanation import ai_explanation_engine
from app.models.schemas import AnalysisRequest, BatchAnalysisRequest

router = APIRouter()


async def run_analysis(
    symbol: str,
    timeframe: str,
    interval: str,
    websites: list[str],
    criteria_config: dict | None = None,
    model: str = "",
    custom_rules: dict | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generator that yields one SSE event per step.
    Each step: yields "running" -> executes -> yields "done" with data.
    On error: yields "error" and stops the pipeline.
    """
    state: dict = {}

    yield SSE_STREAM_INIT  # disable browser auto-reconnect

    try:
        # Step 0 — Fetch stock data
        yield sse_event(0, "running", "Fetching stock data")
        df, stock_info, price_summary = await asyncio.to_thread(
            fetch_stock_data, symbol, timeframe, interval
        )
        state["df"] = df
        state["stock_info"] = stock_info
        # Extract sector automatically from Yahoo Finance info
        sector = stock_info.get("sector", stock_info.get("industry", ""))
        yield sse_event(0, "done", "Fetching stock data", price_summary)

        # Step 1 — Compute technical indicators
        yield sse_event(1, "running", "Computing technical indicators")
        df, indicators_summary = await asyncio.to_thread(compute_indicators, df)
        state["df"] = df
        yield sse_event(1, "done", "Computing technical indicators", indicators_summary)

        # Step 2 — Custom criteria / rule-based signals
        yield sse_event(2, "running", "Running custom criteria engine")
        criteria_signals, criteria_summary = await asyncio.to_thread(
            custom_criteria, df, criteria_config, stock_info
        )
        state["criteria_signals"] = criteria_signals
        yield sse_event(2, "done", "Running custom criteria engine", criteria_summary)

        # Step 2b — AND/OR/NOT custom rules (visual builder)
        if custom_rules:
            yield sse_event(2, "running", "Evaluating custom rule tree")
            custom_rule_result = await asyncio.to_thread(evaluate_custom_rules, df, custom_rules)
            state["custom_rule_result"] = custom_rule_result
            yield sse_event(2, "done", "Custom rule evaluation complete", custom_rule_result)

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

        # Step 5b — Post-analysis criteria (NEWS_SENTIMENT, BUY_PROB, SELL_PROB, CONFIDENCE)
        post_signals = evaluate_post_analysis_criteria(
            criteria_config,
            ai_signal, ai_confidence, ai_proba,
            sentiment_score, sentiment_label,
        )
        state["post_signals"] = post_signals
        if post_signals:
            # Merge into criteria_signals so the explanation engine sees them
            for k, v in post_signals.items():
                criteria_signals[k] = (v["signal"], v["reason"])
            yield sse_event(
                5,
                "done",
                "Post-analysis criteria evaluated",
                {
                    "post_signals": post_signals,
                    "buy_count": sum(1 for v in post_signals.values() if v["signal"] == "BUY"),
                    "sell_count": sum(1 for v in post_signals.values() if v["signal"] == "SELL"),
                    "hold_count": sum(1 for v in post_signals.values() if v["signal"] == "HOLD"),
                },
            )

        # Step 6 — AI explanation engine
        ai_model = model or OPENROUTER_MODEL
        yield sse_event(6, "running", f"Generating AI explanation ({ai_model})")
        explanation_result = await asyncio.to_thread(
            ai_explanation_engine,
            symbol, df, ai_signal, ai_confidence, ai_proba,
            criteria_signals, sentiment_score, sentiment_label,
            trade_levels, raw_sentiment, OPENROUTER_API_KEY, ai_model, sector,
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
            "custom_rule_result": state.get("custom_rule_result", {}),
            "post_analysis_signals": state.get("post_signals", {}),
            "indicators": indicators_summary,
            "explanation": explanation_result,
        }
        yield sse_event(7, "complete", "Analysis complete", summary)
        yield SSE_STREAM_CLOSE  # signals frontend to close EventSource cleanly

    except Exception as exc:
        yield sse_event(-1, "error", "Pipeline failed", {"error": str(exc)})
        yield SSE_STREAM_CLOSE


@router.get("/analyze")
async def analyze(
    symbol: str = Query(..., description="Yahoo Finance ticker, e.g. RELIANCE.NS"),
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
        run_analysis(symbol, timeframe, interval, site_list, criteria_config, model),
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


async def run_batch_analysis(
    symbols: list[str],
    timeframe: str,
    interval: str,
    websites: list[str],
    criteria_config: dict | None = None,
    model: str = "",
    custom_rules: dict | None = None,
) -> AsyncGenerator[str, None]:
    """Run analysis for multiple symbols in parallel, multiplexing all SSE events
    into a single stream. Each event includes a 'symbol' field so the frontend
    can route it to the correct stock panel."""

    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _run_one(sym: str) -> None:
        try:
            async for event in run_analysis(sym, timeframe, interval, websites, criteria_config, model, custom_rules):
                raw = event.removeprefix("data: ").strip()
                try:
                    payload = json.loads(raw)
                    payload["symbol"] = sym
                    await queue.put(f"data: {json.dumps(payload)}\n\n")
                except Exception:
                    await queue.put(event)
        finally:
            await queue.put(None)  # sentinel: this symbol is done

    tasks = [asyncio.create_task(_run_one(sym)) for sym in symbols]

    finished = 0
    while finished < len(symbols):
        item = await queue.get()
        if item is None:
            finished += 1
        else:
            yield item

    await asyncio.gather(*tasks, return_exceptions=True)


@router.get("/analyze/batch")
async def analyze_batch(
    symbols: str = Query(..., description="Comma-separated Yahoo Finance tickers, e.g. RELIANCE.NS,TCS.NS,INFY.NS"),
    timeframe: str = Query("1y", description="yfinance period: 1mo 3mo 6mo 1y 3y 5y"),
    interval: str = Query("1d", description="yfinance interval: 1d 1wk 1mo"),
    websites: str = Query("", description="Comma-separated domain list"),
    criteria: str = Query("", description="JSON criteria config"),
    model: str = Query("", description="OpenRouter model name"),
    max_symbols: int = Query(10, description="Safety cap on number of symbols (max 20)"),
):
    """
    Stream parallel analysis for multiple NSE stocks as Server-Sent Events.

    Each SSE message includes a 'symbol' field so the frontend can route
    events to the correct stock panel. All stocks are analyzed concurrently.

    Example: GET /analyze/batch?symbols=RELIANCE.NS,TCS.NS,HDFCBANK.NS
    """
    symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
    symbol_list = symbol_list[:min(max_symbols, 20)]

    site_list = [s.strip() for s in websites.split(",") if s.strip()][:MAX_CUSTOM_WEBSITES]

    criteria_config = None
    if criteria:
        try:
            criteria_config = json.loads(criteria)
        except json.JSONDecodeError:
            pass

    return StreamingResponse(
        run_batch_analysis(symbol_list, timeframe, interval, site_list, criteria_config, model),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─── POST endpoints (JSON body — recommended for frontend) ─────────────────────

@router.post("/analyze", summary="Single-symbol analysis via JSON body")
async def analyze_post(req: AnalysisRequest):
    """
    Same as GET /analyze but accepts a clean JSON body instead of query params.
    Use this from the frontend — avoids URL length limits.

    Stream is Server-Sent Events (text/event-stream).
    Frontend should use fetch() + ReadableStream (NOT EventSource) for POST.

    Body example:
        {
            "symbol": "360ONE.NS",
            "timeframe": "1y",
            "interval": "1d",
            "websites": ["moneycontrol.com", "economictimes.com"],
            "criteria": {
                "RSI": {"enabled": true, "oversold": 30},
                "MACD": {"enabled": true},
                "Volume": {"enabled": true, "spike_threshold": 1.5}
            },
            "model": "nvidia/nemotron-3-super-120b-a12b:free"
        }
    """
    site_list = req.websites[:MAX_CUSTOM_WEBSITES]
    return StreamingResponse(
        run_analysis(req.symbol, req.timeframe, req.interval, site_list,
                     req.criteria, req.model, req.custom_rules),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/analyze/batch", summary="Multi-symbol batch analysis via JSON body")
async def analyze_batch_post(req: BatchAnalysisRequest):
    """
    Same as GET /analyze/batch but accepts a clean JSON body instead of query params.
    Use this from the frontend — avoids URL length limits.

    Stream is Server-Sent Events (text/event-stream).
    Frontend should use fetch() + ReadableStream (NOT EventSource) for POST.

    Body example:
        {
            "symbols": ["360ONE.NS", "RELIANCE.NS", "TCS.NS"],
            "timeframe": "1y",
            "interval": "1d",
            "websites": ["moneycontrol.com", "economictimes.com", "livemint.com"],
            "criteria": {
                "RSI": {"enabled": true, "oversold": 30},
                "MACD": {"enabled": true},
                "SMA50": {"enabled": true},
                "EMA_Cross": {"enabled": true},
                "Bollinger": {"enabled": true},
                "Volume": {"enabled": true, "spike_threshold": 1.5}
            },
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "max_symbols": 10
        }
    """
    symbol_list = req.symbols[:min(req.max_symbols, 20)]
    site_list = req.websites[:MAX_CUSTOM_WEBSITES]
    return StreamingResponse(
        run_batch_analysis(symbol_list, req.timeframe, req.interval, site_list,
                           req.criteria, req.model, req.custom_rules),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
