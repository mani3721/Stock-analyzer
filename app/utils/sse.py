import json

# Tell the browser not to auto-reconnect when the stream closes normally.
SSE_STREAM_INIT = "retry: 0\n\n"

# Terminal event sent after the final "complete" payload.
# Frontend should listen for event type "close" and call eventSource.close()
# to prevent the browser firing onerror → "Connection lost".
SSE_STREAM_CLOSE = "event: close\ndata: {}\n\n"


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
