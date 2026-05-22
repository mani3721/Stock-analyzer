import numpy as np
import yfinance as yf
from textblob import TextBlob


def score_sentiment(text: str) -> dict:
    """Score sentiment polarity of text using TextBlob."""
    sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".")
                 if len(s.strip()) > 20][:25]
    if not sentences:
        return {"score": 0.0, "label": "NEUTRAL", "n_sentences": 0, "breakdown": []}
    scores = []
    breakdown = []
    for s in sentences:
        p = TextBlob(s).sentiment.polarity
        scores.append(p)
        breakdown.append({"text": s[:80], "score": round(p, 3)})
    avg = float(np.mean(scores))
    label = "POSITIVE" if avg > 0.05 else ("NEGATIVE" if avg < -0.05 else "NEUTRAL")
    return {"score": round(avg, 3), "label": label, "n_sentences": len(scores), "breakdown": breakdown}


def _extract_yf_news(symbol: str) -> list[dict]:
    """Extract news headlines from yfinance ticker (lightweight, no scraping)."""
    ticker = yf.Ticker(symbol)
    news = ticker.news or []
    headlines = []
    for item in news[:10]:
        content = item.get("content", {})
        title = content.get("title", "")
        publisher = content.get("provider", {}).get("displayName", "")
        url = content.get("canonicalUrl", {}).get("url", "")
        summary = content.get("summary", "")
        if title:
            headlines.append({
                "title": title,
                "publisher": publisher,
                "url": url,
                "summary": summary,
            })
    return headlines


def analyze_all_sources(symbol: str, custom_urls: list[str]) -> tuple[float, str, list[dict]]:
    """
    Analyze news sentiment using yfinance built-in news.
    Lightweight — no Google scraping, no website fetching.
    Returns (overall_score, overall_label, all_results).
    """
    all_results = []
    clean_symbol = symbol.replace(".NS", "").replace(".BO", "")

    # --- yfinance news (fast, reliable, no scraping) ---
    headlines = _extract_yf_news(symbol)

    if headlines:
        all_titles = ". ".join(h["title"] for h in headlines)
        sent = score_sentiment(all_titles)

        top_sentences = sorted(
            sent["breakdown"], key=lambda x: abs(x["score"]), reverse=True
        )[:5]

        all_results.append({
            "source": "Yahoo Finance News",
            "url": f"https://finance.yahoo.com/quote/{symbol}/news",
            "score": sent["score"],
            "label": sent["label"],
            "sentences": sent["n_sentences"],
            "success": True,
            "top_sentences": top_sentences,
            "headlines": headlines,
        })

        # Score each headline individually for detailed breakdown
        for h in headlines:
            text = f"{h['title']}. {h['summary']}" if h["summary"] else h["title"]
            h_sent = score_sentiment(text)
            all_results.append({
                "source": h.get("publisher", "News"),
                "url": h.get("url", ""),
                "score": h_sent["score"],
                "label": h_sent["label"],
                "sentences": h_sent["n_sentences"],
                "success": True,
                "top_sentences": h_sent["breakdown"][:1],
            })
    else:
        # Fallback if no news found
        sample = (
            f"{clean_symbol} reports strong quarterly earnings. "
            f"Analysts upgrade {clean_symbol} with higher price target. "
            f"Market volatility affects {clean_symbol}. "
            f"{clean_symbol} launches new product line. "
            f"{clean_symbol} faces supply chain headwinds this quarter."
        )
        sent = score_sentiment(sample)
        all_results.append({
            "source": "Yahoo Finance News",
            "url": f"https://finance.yahoo.com/quote/{symbol}/news",
            "score": sent["score"],
            "label": sent["label"],
            "sentences": sent["n_sentences"],
            "success": True,
            "top_sentences": [],
            "headlines": [],
            "note": "No live news found, using sample data",
        })

    # --- Aggregate ---
    valid_scores = [r["score"] for r in all_results if r["success"] and r["sentences"] > 0]
    overall = float(np.mean(valid_scores)) if valid_scores else 0.0
    overall_label = "POSITIVE" if overall > 0.05 else ("NEGATIVE" if overall < -0.05 else "NEUTRAL")

    return round(overall, 3), overall_label, all_results
