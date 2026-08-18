"""
Project 04 — FastAPI REST layer.

GET /brief?ticker=MSFT
GET /compare?tickers=AAPL,MSFT,GOOGL
GET /pdf?ticker=MSFT
GET /health
"""

import logging
import os
import time
from typing import List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf

load_dotenv()

from graph import graph
from resolver import resolve_ticker
from pdf_generator import generate_stock_pdf

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
    description="Fan-out/fan-in multi-agent stock research: fundamentals, sentiment, technicals, and executive briefs.",
    version="2.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


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
        logger.exception("Graph execution failed for %s", resolved_ticker)
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
            "metrics": result.get("metrics"),
            "technical_indicators": result.get("technical_indicators"),
            "news_items": result.get("news_items"),
            "recommendation_summary": result.get("recommendation_summary"),
            "disclaimer": (
                "⚠️ This is AI-generated analysis for educational purposes only. "
                "It is not financial advice. Always do your own research and consult "
                "a licensed financial advisor before making investment decisions."
            ),
        }
    )


@app.get("/compare")
def compare_stocks(
    tickers: str = Query(
        ...,
        description="Comma-separated tickers to compare (e.g. AAPL,MSFT,NVDA)",
    ),
):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail="Please provide at least one ticker.")
    if len(ticker_list) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 tickers allowed for comparison.")

    comparison_results = []
    for sym in ticker_list:
        resolved = resolve_ticker(sym)
        r_ticker = resolved.get("ticker", sym)
        try:
            stock = yf.Ticker(r_ticker)
            info = stock.info or {}
            hist = stock.history(period="1mo")
            current_price = float(hist["Close"].iloc[-1]) if not hist.empty and "Close" in hist else None
            prev_price = float(hist["Close"].iloc[0]) if not hist.empty and len(hist) > 1 else current_price
            month_return = round(((current_price / prev_price) - 1) * 100, 2) if current_price and prev_price else None

            comparison_results.append({
                "ticker": r_ticker,
                "name": info.get("shortName") or info.get("longName") or resolved.get("name", sym),
                "sector": info.get("sector", "N/A"),
                "current_price": current_price,
                "currency": info.get("currency", "USD"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "market_cap": info.get("marketCap"),
                "profit_margins": info.get("profitMargins"),
                "return_on_equity": info.get("returnOnEquity"),
                "1m_return_pct": month_return,
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
            })
        except Exception as exc:
            logger.warning("Error fetching info for comparison ticker %s: %s", r_ticker, exc)
            comparison_results.append({
                "ticker": r_ticker,
                "name": sym,
                "error": str(exc),
            })

    return {"comparison": comparison_results}


@app.get("/pdf")
def get_pdf_report(
    ticker: str = Query(..., description="Stock ticker or company name"),
):
    resolved = resolve_ticker(ticker.strip())
    resolved_ticker = resolved.get("ticker")
    if not resolved_ticker:
        raise HTTPException(status_code=400, detail="Could not resolve stock ticker.")

    try:
        result = get_cached_research(resolved_ticker)
        stock = yf.Ticker(resolved_ticker)
        info = stock.info or {}
        hist = stock.history(period="5d")
        current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        company_name = info.get("longName") or info.get("shortName") or resolved.get("name") or resolved_ticker
        currency_symbol = "$" if info.get("currency") != "INR" else "INR "

        pdf_bytes = generate_stock_pdf(
            ticker=resolved_ticker,
            company_name=company_name,
            current_price=current_price,
            currency_symbol=currency_symbol,
            brief_text=result.get("investment_brief", "No brief available."),
            fundamentals_text=result.get("fundamentals_report", "No report."),
            sentiment_text=result.get("sentiment_report", "No report."),
            technical_text=result.get("technical_report", "No report."),
            metrics=result.get("metrics"),
            technical_indicators=result.get("technical_indicators"),
            recommendation=result.get("recommendation_summary"),
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{resolved_ticker}_report.pdf"'},
        )
    except Exception as exc:
        logger.exception("PDF generation failed for %s", resolved_ticker)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(exc)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
    )
