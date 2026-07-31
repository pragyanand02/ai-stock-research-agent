"""
Project 04 — FastAPI REST layer.

GET /brief?ticker=MSFT
GET /health
"""
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from graph import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    ticker: str = Query(..., description="Stock ticker symbol (e.g. MSFT, RELIANCE.NS)"),
):
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker parameter is required.")

    try:
        result = graph.invoke({"ticker": ticker})
    except Exception as exc:
        logger.exception("Graph execution failed for %s", ticker)
        raise HTTPException(
            status_code=500,
            detail="Pipeline execution failed. Please try again later.",
        ) from exc

    return JSONResponse({
        "ticker": ticker,
        "fundamentals_report": result.get("fundamentals_report"),
        "sentiment_report": result.get("sentiment_report"),
        "technical_report": result.get("technical_report"),
        "investment_brief": result.get("investment_brief"),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8002, reload=True)