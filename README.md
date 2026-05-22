# AI Stock Investment Analysis Platform

Multi-step stock analysis backend using FastAPI + Server-Sent Events (SSE).

## Architecture

```
User Input → Yahoo Finance + News Scraper + Custom Criteria → Technical Analysis → AI Scoring → Explanation Engine → SSE Stream
```

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download TextBlob corpora (first time only)
python -m textblob.download_corpora

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OpenRouter API key (optional — fallback works without it)

# 5. Run the server
uvicorn app.main:app --reload --port 8000
```

## API

### `GET /analyze` (SSE Stream)

| Parameter   | Required | Default | Description                              |
|-------------|----------|---------|------------------------------------------|
| `symbol`    | Yes      | —       | Yahoo Finance ticker (e.g. `RELIANCE.NS`)|
| `sector`    | No       | `""`    | Sector label                             |
| `timeframe` | No       | `1y`    | Period: `1mo 3mo 6mo 1y 3y 5y`           |
| `interval`  | No       | `1d`    | Interval: `1d 1wk 1mo`                  |
| `websites`  | No       | `""`    | Comma-separated domains to scrape        |

**Example:**

```
GET http://localhost:8000/analyze?symbol=TCS.NS&sector=Technology&timeframe=1y&websites=moneycontrol.com,livemint.com
```

### SSE Event Format

Each streamed event is JSON:

```json
{
  "step": 0,
  "status": "running | done | error | complete",
  "title": "Fetching stock data",
  "data": { ... }
}
```

### Pipeline Steps

| Step | Title                          |
|------|--------------------------------|
| 0    | Fetching stock data            |
| 1    | Computing technical indicators |
| 2    | Running custom criteria engine |
| 3    | Training AI scoring model      |
| 4    | Computing trade levels         |
| 5    | Scanning custom websites       |
| 6    | Generating AI explanation      |
| 7    | Analysis complete (summary)    |

### `GET /health`

Returns `{ "status": "ok", "timestamp": ... }`

## Project Structure

```
Stock Backend/
├── app/
│   ├── main.py              # FastAPI app + CORS
│   ├── config.py            # Environment settings
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── stock_data.py    # Yahoo Finance data fetch
│   │   ├── indicators.py    # Technical indicators (pandas_ta)
│   │   ├── news_analyzer.py # Web scraping + sentiment
│   │   ├── criteria.py      # Rule-based signal engine
│   │   ├── ai_engine.py     # Random Forest ML model
│   │   ├── trade_levels.py  # Entry/exit level computation
│   │   └── explanation.py   # AI explanation (OpenRouter)
│   ├── routes/
│   │   └── analysis.py      # SSE streaming endpoint
│   └── utils/
│       └── sse.py           # SSE event formatter
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Frontend Connection

```javascript
const eventSource = new EventSource(
  'http://localhost:8000/analyze?symbol=TCS.NS&sector=Technology'
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Step ${data.step}: ${data.status} — ${data.title}`);
  // Update UI step-by-step
};
```

---

*Educational purposes only. Not financial advice.*
