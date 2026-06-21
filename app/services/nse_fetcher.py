import asyncio
import csv
import io
import logging
import time
import urllib.request

logger = logging.getLogger(__name__)

# Single Nifty 500 CSV covers large + mid + small cap (~502 stocks)
_URL = "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Referer": "https://www.niftyindices.com/",
}

_CACHE_TTL = 7 * 24 * 60 * 60  # refresh once per week (indices rebalance semi-annually)

_cache: list[str] = []
_cache_ts: float = 0.0
_lock = asyncio.Lock()


def _fetch_url_sync(url: str) -> list[str]:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(data))
    return [row["Symbol"].strip() for row in reader]


_FETCH_TIMEOUT = 20


async def _fetch_all() -> list[str]:
    loop = asyncio.get_event_loop()
    try:
        symbols = await asyncio.wait_for(
            loop.run_in_executor(None, _fetch_url_sync, _URL),
            timeout=_FETCH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Timeout fetching Nifty 500 list")
        return []
    except Exception as exc:
        logger.warning("Failed to fetch Nifty 500 list: %s", exc)
        return []
    return [s + ".NS" for s in dict.fromkeys(symbols)]


async def get_symbols(force_refresh: bool = False) -> list[str]:
    """Return Nifty 500 NSE symbols as Yahoo Finance tickers (.NS suffix).

    Cached for 7 days (indices rebalance semi-annually).
    Raises RuntimeError only on cold-start with no cache and failed fetch.
    """
    global _cache, _cache_ts

    async with _lock:
        age = time.time() - _cache_ts
        if _cache and not force_refresh and age < _CACHE_TTL:
            return _cache

        try:
            symbols = await _fetch_all()
            if len(symbols) < 400:
                raise ValueError(f"Too few symbols returned: {len(symbols)}")
            _cache = symbols
            _cache_ts = time.time()
            logger.info("NSE symbol cache refreshed (%d symbols)", len(_cache))
        except Exception as exc:
            logger.error("Live fetch failed: %s", exc)
            if not _cache:
                raise RuntimeError(
                    "Failed to fetch NSE symbols and no cached data available."
                ) from exc
            logger.warning("Serving stale cache (%d symbols)", len(_cache))

        return _cache


# backward-compat alias
get_top250 = get_symbols
