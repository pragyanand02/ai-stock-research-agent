"""
Project 04 — FastAPI REST layer.

GET /brief?ticker=MSFT
GET /health
"""

import logging
import os
import time
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

load_dotenv()

from graph import graph
from resolver import resolve_ticker


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
_research_cache = {}


def get_cached_research(ticker: str):
    now = time.time()

    if ticker in _research_cache:
        result, timestamp = _research_cache[ticker]
        if now - timestamp < CACHE_TTL_SECONDS:
            return result

    result = graph.invoke({"ticker": ticker})
    _research_cache[ticker] = (result, now)
    return result


app = FastAPI(
    title="AI Stock Research & Investment Brief Generator",
    description="Fan-out/fan-in multi-agent stock research: fundamentals, sentiment, and technicals.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/brief")
def get_brief(
    ticker: str = Query(
        ...,
        description="Stock ticker or company name (e.g. MSFT, Apple, RELIANCE.NS)",
    ),
):
    ticker = ticker.strip()

    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="ticker parameter is required.",
        )

    resolved = resolve_ticker(ticker)
    resolved_ticker = resolved.get("ticker")

    if not resolved_ticker:
        raise HTTPException(
            status_code=400,
            detail="Could not resolve the stock ticker.",
        )

    try:
        result = get_cached_research(resolved_ticker)
    except Exception as exc:
        logger.exception(
            "Graph execution failed for %s",
            resolved_ticker,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {str(exc)}",
        ) from exc

    return JSONResponse(
        {
            "ticker": resolved_ticker,
            "input": ticker,
            "resolution_source": resolved.get("source"),
            "company_name": resolved.get("name"),
            "fundamentals_report": result.get("fundamentals_report"),
            "sentiment_report": result.get("sentiment_report"),
            "technical_report": result.get("technical_report"),
            "investment_brief": result.get("investment_brief"),
            "disclaimer": (
                "⚠️ This is AI-generated analysis for educational purposes only. "
                "It is not financial advice. Always do your own research and consult "
                "a licensed financial advisor before making investment decisions."
            ),
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
    )