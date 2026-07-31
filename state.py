"""
Project 04 — State definition (TypedDict).
"""
from typing import TypedDict


class StockResearchState(TypedDict, total=False):
    ticker: str                    # input — e.g. "MSFT" or "RELIANCE.NS"
    fundamentals_report: str       # Agent 1 output
    sentiment_report: str          # Agent 2 output
    technical_report: str          # Agent 3 output
    investment_brief: str          # Agent 4 output — final result