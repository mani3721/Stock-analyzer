import asyncio
import csv
import io
import logging
import time
import urllib.request

logger = logging.getLogger(__name__)

_URLS = [
    "https://niftyindices.com/IndexConstituent/ind_nifty200list.csv",
    "https://niftyindices.com/IndexConstituent/ind_niftymidcap150list.csv",
]

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


async def _fetch_all() -> list[str]:
    loop = asyncio.get_event_loop()
    symbols: list[str] = []
    for url in _URLS:
        try:
            batch = await loop.run_in_executor(None, _fetch_url_sync, url)
            symbols.extend(batch)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)

    unique = list(dict.fromkeys(symbols))[:250]
    return [s + ".NS" for s in unique]


async def get_top250(force_refresh: bool = False) -> list[str]:
    """Return top-250 NSE symbols as Yahoo Finance tickers.

    Results are fetched live from niftyindices.com and cached for 24 hours.
    Raises RuntimeError if the live fetch fails and no cached data exists.
    """
    global _cache, _cache_ts

    async with _lock:
        age = time.time() - _cache_ts
        if _cache and not force_refresh and age < _CACHE_TTL:
            return _cache

        try:
            symbols = await _fetch_all()
            if len(symbols) < 200:
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
