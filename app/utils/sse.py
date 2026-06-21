import json


def sse_event(
    step: int,
    status: str,
    title: str,
    data: dict | None = None,
    symbol: str | None = None,
) -> str:
    """
    Format a single Server-Sent Event message.
    Frontend reads: event.data -> JSON with step / status / title / payload
    Optional symbol field is included when streaming batch analysis.
    """
    payload = {
        "step": step,
        "status": status,
        "title": title,
        "data": data or {},
    }
    if symbol is not None:
        payload["symbol"] = symbol
    return f"data: {json.dumps(payload)}\n\n"
