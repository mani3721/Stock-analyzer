import asyncio
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.routes.analysis import router as analysis_router
from app.services.nse_fetcher import get_top250

app = FastAPI(
    title="AI Stock Analysis API",
    description="Multi-step stock analysis streamed via Server-Sent Events",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(analysis_router)


@app.on_event("startup")
async def _warm_cache():
    asyncio.create_task(get_top250())


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/top250")
async def top250_symbols(
    refresh: bool = Query(False, description="Force a live re-fetch from niftyindices.com"),
):
    """Returns up to 250 NSE stock symbols in Yahoo Finance format (.NS suffix).

    Results are fetched live from niftyindices.com and cached for 24 hours.
    Pass ?refresh=true to force an immediate re-fetch.
    """
    try:
        symbols = await get_top250(force_refresh=refresh)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    options = [{"label": s.replace(".NS", ""), "value": s} for s in symbols]
    return {"count": len(options), "symbols": options}

