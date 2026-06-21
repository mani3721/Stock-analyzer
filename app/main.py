import asyncio
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.routes.analysis import router as analysis_router
from app.services.nse_fetcher import get_symbols

app = FastAPI(
    title="AI Stock Analysis API",
    description="Multi-step stock analysis streamed via Server-Sent Events",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

app.include_router(analysis_router)


@app.on_event("startup")
async def _warm_cache():
    asyncio.create_task(get_symbols())


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/symbols")
async def symbols_endpoint(
    refresh: bool = Query(False, description="Force a live re-fetch from niftyindices.com"),
):
    """Returns Nifty 500 NSE stock symbols (~502) in Yahoo Finance format (.NS suffix).

    Each entry has a 'label' (e.g. HDFCBANK) and 'value' (e.g. HDFCBANK.NS)
    ready for use in a frontend dropdown.
    Pass ?refresh=true to force an immediate re-fetch.
    """
    try:
        symbols = await get_symbols(force_refresh=refresh)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    options = [{"label": s.replace(".NS", ""), "value": s} for s in symbols]
    return {"count": len(options), "symbols": options}

