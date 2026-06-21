from pydantic import BaseModel, Field


# ─── SSE Event (response model) ───────────────────────────────────────────────

class SSEEvent(BaseModel):
    step: int
    status: str       # "running" | "done" | "error" | "complete"
    title: str
    data: dict = {}


# ─── Criteria / Custom Rules ───────────────────────────────────────────────────

class CriteriaConfig(BaseModel):
    """Per-indicator enable/threshold config passed inside AnalysisRequest."""
    enabled: bool = True
    oversold: float | None = None
    overbought: float | None = None
    spike_threshold: float | None = None

    model_config = {"extra": "allow"}   # allow any extra indicator-specific keys


# ─── Single-symbol POST body ───────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Yahoo Finance ticker, e.g. RELIANCE.NS")
    timeframe: str = Field("1y", description="yfinance period: 1mo 3mo 6mo 1y 3y 5y")
    interval: str = Field("1d", description="yfinance interval: 1d 1wk 1mo")
    websites: list[str] = Field(
        default_factory=list,
        description="List of domains to scan for sentiment, e.g. ['moneycontrol.com']",
    )
    criteria: dict[str, dict] | None = Field(
        None,
        description=(
            "Map of indicator → config. "
            'Example: {"RSI": {"enabled": true, "oversold": 25}, "MACD": {"enabled": true}}'
        ),
    )
    model: str = Field(
        "",
        description="OpenRouter model name, e.g. nvidia/nemotron-3-super-120b-a12b:free",
    )
    custom_rules: dict | None = Field(
        None,
        description=(
            "AND/OR/NOT rule tree from the visual criteria builder. "
            "Supports nested groups. Evaluated as a separate pipeline step. "
            "Example: "
            '{"operator":"OR","signal_if_match":"BUY","conditions":['
            '{"not":true,"indicator":"RSI","op":"<","value":30},'
            '{"indicator":"PRICE","op":">","value":"EMA50"}'
            "]}"
        ),
    )

    model_config = {"json_schema_extra": {
        "example": {
            "symbol": "360ONE.NS",
            "timeframe": "1y",
            "interval": "1d",
            "websites": [
                "moneycontrol.com",
                "economictimes.com",
                "livemint.com",
                "tickertape.in",
                "screener.in",
                "trendlyne.com",
                "ndtvprofit.com",
                "businesstoday.in",
            ],
            "criteria": {
                "RSI":    {"enabled": True, "oversold": 30, "overbought": 70},
                "MACD":   {"enabled": True},
                "EMA20":  {"enabled": True},
                "EMA50":  {"enabled": True},
                "EMA200": {"enabled": True},
                "BOLLINGER": {"enabled": True},
                "VOLUME_SMA": {"enabled": True, "spike_threshold": 1.5},
                "ADX":    {"enabled": True},
                "SUPERTREND": {"enabled": True},
                "VWAP":   {"enabled": True},
                "ICHIMOKU": {"enabled": True},
                "OBV":    {"enabled": True},
                "MFI":    {"enabled": True},
                "STOCHASTIC": {"enabled": True},
                "FIB":    {"enabled": True},
                "PIVOT":  {"enabled": True},
                "DOJI":   {"enabled": True},
                "HAMMER": {"enabled": True},
                "ENGULFING": {"enabled": True},
                "PE":     {"enabled": True, "max": 25},
                "ROE":    {"enabled": True, "min": 15},
                "SHARPE": {"enabled": True, "min": 0.5},
                "MAX_DRAWDOWN": {"enabled": True},
                "BUY_PROB":  {"enabled": True, "min": 50},
                "SELL_PROB": {"enabled": True, "min": 50},
                "CONFIDENCE": {"enabled": True, "min": 60},
                "NEWS_SENTIMENT": {"enabled": True},
            },
            "custom_rules": {
                "operator": "OR",
                "signal_if_match": "BUY",
                "signal_if_no_match": "HOLD",
                "conditions": [
                    {"not": True, "indicator": "RSI", "op": "<", "value": 30},
                    {"indicator": "PRICE", "op": ">", "value": "EMA50"},
                ],
            },
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
        }
    }}


# ─── Batch POST body ───────────────────────────────────────────────────────────

class BatchAnalysisRequest(BaseModel):
    symbols: list[str] = Field(
        ...,
        description="Yahoo Finance tickers, e.g. ['RELIANCE.NS', 'TCS.NS']",
        min_length=1,
    )
    timeframe: str = Field("1y", description="yfinance period: 1mo 3mo 6mo 1y 3y 5y")
    interval: str = Field("1d", description="yfinance interval: 1d 1wk 1mo")
    websites: list[str] = Field(
        default_factory=list,
        description="List of domains to scan for sentiment",
    )
    criteria: dict[str, dict] | None = Field(
        None,
        description="Map of indicator → config (same as single-symbol request)",
    )
    model: str = Field(
        "",
        description="OpenRouter model name",
    )
    custom_rules: dict | None = Field(
        None,
        description="AND/OR/NOT rule tree from the visual criteria builder (same format as single-symbol request)",
    )
    max_symbols: int = Field(
        10,
        ge=1,
        le=20,
        description="Safety cap on number of symbols analysed (max 20)",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "symbols": ["360ONE.NS", "RELIANCE.NS", "TCS.NS"],
            "timeframe": "1y",
            "interval": "1d",
            "websites": [
                "moneycontrol.com",
                "economictimes.com",
                "livemint.com",
                "tickertape.in",
                "screener.in",
                "trendlyne.com",
                "ndtvprofit.com",
                "businesstoday.in",
            ],
            "criteria": {
                "RSI": {"enabled": True, "oversold": 30, "overbought": 70},
                "MACD": {"enabled": True},
                "SMA50": {"enabled": True},
                "EMA_Cross": {"enabled": True},
                "Bollinger": {"enabled": True},
                "Volume": {"enabled": True, "spike_threshold": 1.5},
            },
            "model": "nvidia/nemotron-3-super-120b-a12b:free",
            "max_symbols": 10,
        }
    }}
