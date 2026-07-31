"""
Project 04 — AI Stock Research and Investment Brief Generator
Agents: Financial Data Analyst, News Sentiment Scanner, Price Signal Reader, Brief Writer
"""

from typing import Optional
import json
import logging
import os
from typing import Any

import numpy as np
import requests
import yfinance as yf

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available; technical indicators will use manual fallback.")

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate

from state import StockResearchState
from openai import OpenAI

logger = logging.getLogger(__name__)

_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

_llm_writer = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)
NEWS_API_KEY = ""


def _safe_latest(arr: np.ndarray) -> Optional[float]:
    """Return the latest non-NaN value from a numpy array, or None."""
    clean = arr[~np.isnan(arr)]
    return float(clean[-1]) if len(clean) > 0 else None


# ---------------------------------------------------------------------------
# Fallback indicator calculations (when TA-Lib is not installed)
# ---------------------------------------------------------------------------

def _compute_rsi_manual(prices: np.ndarray, period: int = 14) -> np.ndarray:
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi = np.full(len(prices), np.nan)
    if len(gains) < period:
        return rsi
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    ema = np.full(len(prices), np.nan)
    if len(prices) < period:
        return ema
    # Find the first index at which `period` consecutive finite values exist.
    # This makes the function NaN-aware (e.g. when prices is a MACD line that
    # starts with NaNs from the fast/slow EMA warmup).
    start = -1
    for i in range(len(prices) - period + 1):
        if not np.any(np.isnan(prices[i : i + period])):
            start = i + period - 1
            ema[start] = np.mean(prices[i : i + period])
            break
    if start == -1:
        return ema
    k = 2.0 / (period + 1)
    for i in range(start + 1, len(prices)):
        prev, cur = ema[i - 1], prices[i]
        ema[i] = cur * k + prev * (1 - k) if not (np.isnan(prev) or np.isnan(cur)) else np.nan
    return ema


def _compute_macd_manual(
    prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema_fast = _compute_ema(prices, fast)
    ema_slow = _compute_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _compute_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ---------------------------------------------------------------------------
# Agent 1 — Financial Data Analyst
# ---------------------------------------------------------------------------
_FUNDAMENTALS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a fundamental stock analyst. "
        "Interpret the provided financial metrics and write a concise analysis covering:\n"
        "- Valuation (is the stock cheap, fair, or expensive relative to earnings?)\n"
        "- Growth trajectory (revenue and earnings trend)\n"
        "- Financial health (debt levels, free cash flow, profit margins)\n"
        "- Red flags (any concerning metrics)\n"
        "- Analyst consensus target vs current price\n\n"
        "Be specific and cite the actual numbers. 150–200 words max.",
    ),
    ("human", "Financial metrics for {ticker}:\n\n{metrics}"),
])

_FUNDAMENTAL_FIELDS = [
    "trailingPE", "forwardPE", "trailingEps", "revenueGrowth", "earningsGrowth",
    "debtToEquity", "freeCashflow", "profitMargins", "grossMargins", "operatingMargins",
    "marketCap", "targetMeanPrice", "currentPrice", "returnOnEquity", "returnOnAssets",
    "totalRevenue", "totalDebt",
]


def agent_fundamentals_analyst(state: StockResearchState) -> dict:
    """Agent 1: Fetch fundamentals from Yahoo Finance and interpret them."""
    ticker = state["ticker"]
    info = yf.Ticker(ticker).info
    metrics = {k: info.get(k) for k in _FUNDAMENTAL_FIELDS if info.get(k) is not None}
    chain = _FUNDAMENTALS_PROMPT | _llm
    response = chain.invoke({
        "ticker": ticker,
        "metrics": json.dumps(metrics, indent=2),
    })
    text = response.text if hasattr(response, "text") else str(response.content)

    return {"fundamentals_report": text.strip()}


# ---------------------------------------------------------------------------
# Agent 2 — News Sentiment Scanner
# ---------------------------------------------------------------------------
_SENTIMENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a financial news sentiment analyst. "
        "Review the headlines and classify sentiment for the stock.\n\n"
        "Return:\n"
        "- Overall sentiment: Bullish / Neutral / Bearish\n"
        "- 2–3 key themes from the news\n"
        "- Most significant positive headline (if any)\n"
        "- Most significant negative headline (if any)\n"
        "- Trend direction: improving, stable, or deteriorating\n\n"
        "150 words max. Be specific.",
    ),
    ("human", "Headlines for {ticker} (last 7 days):\n\n{headlines}"),
])


def agent_sentiment_scanner(state: StockResearchState) -> dict:
    """Agent 2: Fetch last 7 days of headlines and classify sentiment."""
    ticker = state["ticker"]
    headlines = _fetch_headlines(ticker)
    chain = _SENTIMENT_PROMPT | _llm
    response = chain.invoke({
        "ticker": ticker,
        "headlines": "\n".join(f"- {h}" for h in headlines) if headlines else "No headlines available.",
    })

    text = response.text if hasattr(response, "text") else str(response.content)

    return {"sentiment_report": text.strip()}


def _fetch_headlines(ticker: str) -> list[str]:
    """Fetch recent news headlines from NewsAPI."""
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not set; using Yahoo Finance news fallback.")
        return _fetch_headlines_yfinance(ticker)
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": ticker,
            "sortBy": "publishedAt",
            "pageSize": 20,
            "language": "en",
            "apiKey": NEWS_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a["title"] for a in articles if a.get("title")]
    except Exception as exc:
        logger.error("NewsAPI error: %s", exc)
        return _fetch_headlines_yfinance(ticker)


def _fetch_headlines_yfinance(ticker: str) -> list[str]:
    """Fallback: get news headlines via yfinance."""
    try:
        news = yf.Ticker(ticker).news or []
        return [item.get("title", "") for item in news[:20] if item.get("title")]
    except Exception as exc:
        logger.error("yfinance news error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Agent 3 — Price Signal Reader (Technical Analysis)
# ---------------------------------------------------------------------------
_TECHNICAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a technical analyst. "
        "Interpret the provided indicator values and derive a technical setup for the stock.\n\n"
        "Cover:\n"
        "- Trend direction (price vs SMA50, SMA200 — golden cross / death cross)\n"
        "- Momentum (RSI — overbought >70 / oversold <30)\n"
        "- MACD signal (bullish crossover / bearish crossover)\n"
        "- Overall technical stance: Bullish / Neutral / Bearish\n\n"
        "150 words max. Be specific with the numbers.",
    ),
    ("human", "Technical indicators for {ticker}:\n\n{indicators}"),
])


def agent_technical_analyst(state: StockResearchState) -> dict:
    """Agent 3: Compute RSI, MACD, SMA50, SMA200 and interpret the technical setup."""
    ticker = state["ticker"]
    hist = yf.Ticker(ticker).history(period="6mo")
    if hist.empty:
        return {"technical_report": f"No price data available for {ticker}."}

    close = hist["Close"].values.astype(float)

    if TALIB_AVAILABLE:
        rsi = talib.RSI(close, timeperiod=14)
        macd_line, signal_line, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        sma50 = talib.SMA(close, timeperiod=50)
        sma200 = talib.SMA(close, timeperiod=200)
    else:
        rsi = _compute_rsi_manual(close, 14)
        macd_line, signal_line, macd_hist = _compute_macd_manual(close)
        sma50 = np.array([np.nan] * len(close))
        sma200 = np.array([np.nan] * len(close))
        if len(close) >= 50:
            for i in range(49, len(close)):
                sma50[i] = np.mean(close[i - 49:i + 1])
        if len(close) >= 200:
            for i in range(199, len(close)):
                sma200[i] = np.mean(close[i - 199:i + 1])

    indicators = {
        "current_price": float(close[-1]),
        "rsi_14": _safe_latest(rsi),
        "macd": _safe_latest(macd_line),
        "macd_signal": _safe_latest(signal_line),
        "macd_histogram": _safe_latest(macd_hist),
        "sma_50": _safe_latest(sma50),
        "sma_200": _safe_latest(sma200),
        "price_vs_sma50": (
            round((float(close[-1]) / _safe_latest(sma50) - 1) * 100, 2)
            if _safe_latest(sma50) else None
        ),
        "price_vs_sma200": (
            round((float(close[-1]) / _safe_latest(sma200) - 1) * 100, 2)
            if _safe_latest(sma200) else None
        ),
    }

    chain = _TECHNICAL_PROMPT | _llm
    response = chain.invoke({
        "ticker": ticker,
        "indicators": json.dumps(indicators, indent=2),
    })
    text = response.text if hasattr(response, "text") else str(response.content)
    return {"technical_report": text.strip()}


# ---------------------------------------------------------------------------
# Agent 4 — Risk and Brief Writer
# ---------------------------------------------------------------------------
_BRIEF_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a decisive investment analyst writing a one-page investment brief. "
        "You must give a clear, opinionated signal. No hedging. No 'it depends'.\n\n"
        "Structure your brief as follows:\n"
        "**Signal:** Buy / Hold / Watch / Avoid\n"
        "**Conviction:** High / Medium / Low\n"
        "**Time Horizon:** Short-term (< 3 months) / Medium-term (3–12 months) / Long-term (> 1 year)\n"
        "**Thesis:** 2–3 sentences summarizing the investment case.\n"
        "**Bull Case:**\n- bullet 1\n- bullet 2\n- bullet 3\n"
        "**Bear Case:**\n- bullet 1\n- bullet 2\n- bullet 3\n"
        "**Bottom Line:** One decisive sentence.\n\n"
        "Be specific. Cite numbers from the reports. Pick a signal and defend it.",
    ),
    (
        "human",
        "Ticker: {ticker}\n\n"
        "Fundamentals Analysis:\n{fundamentals_report}\n\n"
        "News Sentiment:\n{sentiment_report}\n\n"
        "Technical Analysis:\n{technical_report}",
    ),
])


def agent_brief_writer(state: StockResearchState) -> dict:
    """Agent 4: Synthesize all research into an opinionated investment brief."""
    chain = _BRIEF_PROMPT | _llm_writer
    response = chain.invoke({
        "ticker": state["ticker"],
        "fundamentals_report": state.get("fundamentals_report", "Not available."),
        "sentiment_report": state.get("sentiment_report", "Not available."),
        "technical_report": state.get("technical_report", "Not available."),
    })
    text = response.text if hasattr(response, "text") else str(response.content)

    return {"investment_brief": text.strip()}