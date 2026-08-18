"""
Project 04 — State definition (TypedDict).
"""
from typing import TypedDict, Dict, Any, List, Optional


class StockResearchState(TypedDict, total=False):
    ticker: str                             # input — e.g. "MSFT" or "RELIANCE.NS"
    fundamentals_report: str                # Agent 1 output text
    sentiment_report: str                   # Agent 2 output text
    technical_report: str                   # Agent 3 output text
    investment_brief: str                   # Agent 4 output text — final brief
    metrics: Dict[str, Any]                 # Structured financial ratios & fundamentals
    technical_indicators: Dict[str, Any]   # Structured indicator values (RSI, MACD, BB, etc.)
    news_items: List[Dict[str, Any]]        # Structured news articles with links & dates
    recommendation_summary: Dict[str, Any]  # Structured signal, conviction, horizon, etc.