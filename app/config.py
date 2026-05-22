import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

DEFAULT_TIMEFRAME = "1y"
DEFAULT_INTERVAL = "1d"
DEFAULT_SECTOR = ""
DEFAULT_WEBSITES = ""

MAX_CUSTOM_WEBSITES = 8
MAX_SEARCH_RESULTS = 3
REQUEST_TIMEOUT = 15
AI_REQUEST_TIMEOUT = 60

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
