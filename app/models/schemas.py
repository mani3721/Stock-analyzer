from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    symbol: str
    sector: str = ""
    timeframe: str = "1y"
    interval: str = "1d"
    websites: str = ""


class SSEEvent(BaseModel):
    step: int
    status: str       # "running" | "done" | "error" | "complete"
    title: str
    data: dict = {}
