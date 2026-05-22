import json


def sse_event(step: int, status: str, title: str, data: dict | None = None) -> str:
    """
    Format a single Server-Sent Event message.
    Frontend reads: event.data -> JSON with step / status / title / payload
    """
    payload = {
        "step": step,
        "status": status,
        "title": title,
        "data": data or {},
    }
    return f"data: {json.dumps(payload)}\n\n"
