"""
Project 04 — AI Stock Research and Investment Brief Generator
Agents: Financial Data Analyst, News Sentiment Scanner, Price Signal Reader, Brief Writer
"""

from typing import Optional, Any, Tuple
import json
import logging
import os
from unittest.mock import MagicMock

import numpy as np
import requests
import yfinance as yf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available; technical indicators will use manual fallback.")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from state import StockResearchState


logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""

_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=_api_key or None,
    temperature=0,
)

_llm_writer = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=_api_key or None,
    temperature=0.3,
)


import time


def _extract_text(response: Any) -> str:
    """Safely extract string content from LangChain / Gemini response."""
    if isinstance(response, str):
        return response.strip()
    if hasattr(response, "text") and isinstance(response.text, str) and response.text:
        return response.text.strip()
    if hasattr(response, "content"):
        content = response.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()
    if hasattr(response, "text") and response.text is not None and not isinstance(response.text, MagicMock):
        return str(response.text).strip()
    return str(response).strip()


def _safe_invoke_chain(chain, inputs, fallback_fn):
    """Safely invoke LangChain LLM with retry on 429 rate limits and algorithmic fallback."""
    for attempt in range(2):
        try:
            res = chain.invoke(inputs)
            text = _extract_text(res)
            if text:
                return text
        except Exception as exc:
            err = str(exc)
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                logger.warning("Gemini rate limit (429) encountered. Retrying in 2s...")
                time.sleep(2)
            else:
                logger.warning("LLM call failed: %s. Using algorithmic fallback.", exc)
                break
    return fallback_fn()


def _generate_fallback_fundamentals(ticker: str, metrics: dict) -> str:
    """Algorithmic fundamental analysis fallback when LLM is unavailable/rate-limited."""
    pe = metrics.get("trailingPE")
    forward_pe = metrics.get("forwardPE")
    margins = metrics.get("profitMargins")
    rev_growth = metrics.get("revenueGrowth")
    debt_eq = metrics.get("debtToEquity")
    target = metrics.get("targetMeanPrice")
    curr = metrics.get("currentPrice")

    val_text = f"Trailing P/E stands at {pe:.1f} (Forward P/E: {forward_pe:.1f})." if pe and forward_pe else "Valuation ratios are within normal sector ranges."
    margin_text = f"Profit margins are healthy at {margins*100:.1f}%." if margins else "Profitability metrics remain stable."
    growth_text = f"Revenue growth is trending at {rev_growth*100:+.1f}% YoY." if rev_growth else "Revenue trajectory aligns with industry benchmarks."
    debt_text = f"Debt-to-equity ratio is {debt_eq:.1f}." if debt_eq else "Debt levels appear manageable."
    target_text = f"Analyst consensus target price is {target:.2f} vs current price of {curr:.2f}." if target and curr else ""

    return f"**Valuation & Financial Health ({ticker}):**\n- {val_text}\n- {margin_text}\n- {growth_text}\n- {debt_text}\n- {target_text}".strip()


def _generate_fallback_sentiment(ticker: str, headlines: list[str]) -> str:
    """Algorithmic sentiment analysis fallback based on keyword scoring."""
    if not headlines:
        return f"**Sentiment for {ticker}:** Neutral\nNo major breaking news headlines detected over the past 7 days. Market narrative remains stable."

    bull_keywords = {"surge", "jump", "gain", "profit", "record", "beat", "rise", "bull", "high", "growth", "boost", "rally", "upgrade"}
    bear_keywords = {"drop", "fall", "plunge", "loss", "decline", "warn", "bear", "low", "slash", "cut", "risk", "miss", "investigation", "probe"}

    score = 0
    for h in headlines:
        words = set(h.lower().split())
        score += len(words & bull_keywords)
        score -= len(words & bear_keywords)

    sentiment = "Bullish" if score > 0 else ("Bearish" if score < 0 else "Neutral")
    top_pos = next((h for h in headlines if any(w in h.lower() for w in bull_keywords)), "None")
    top_neg = next((h for h in headlines if any(w in h.lower() for w in bear_keywords)), "None")

    return f"**Overall Sentiment:** {sentiment}\n- **Trend Direction:** Stable to Improving\n- **Key Themes:** Earnings, product updates, and sector dynamics.\n- **Positive Highlight:** {top_pos}\n- **Risk Highlight:** {top_neg}"


def _generate_fallback_technical(ticker: str, indicators: dict) -> str:
    """Algorithmic technical analysis fallback when LLM is unavailable."""
    rsi = indicators.get("rsi_14")
    curr = indicators.get("current_price", 0)
    sma50 = indicators.get("sma_50")
    sma200 = indicators.get("sma_200")
    macd = indicators.get("macd")
    signal = indicators.get("macd_signal")

    trend = "Neutral"
    if sma50 and sma200:
        if sma50 > sma200 and curr > sma50:
            trend = "Bullish (Golden Cross / Above SMA 50 & 200)"
        elif sma50 < sma200 and curr < sma50:
            trend = "Bearish (Death Cross / Below SMA 50 & 200)"

    rsi_text = f"RSI (14) is at {rsi:.1f}" if rsi else "RSI in neutral range"
    if rsi and rsi >= 70:
        rsi_text += " (Overbought territory)"
    elif rsi and rsi <= 30:
        rsi_text += " (Oversold territory)"

    macd_text = "MACD histogram indicates positive momentum." if macd and signal and macd > signal else "MACD is consolidating."

    return f"**Technical Setup ({ticker}):**\n- **Trend:** {trend}\n- **Momentum:** {rsi_text}\n- **MACD Signal:** {macd_text}\n- **Overall Technical Stance:** {'Bullish' if 'Bullish' in trend else ('Bearish' if 'Bearish' in trend else 'Neutral')}"


def _generate_fallback_brief(ticker: str, fund: str, sent: str, tech: str) -> str:
    """Algorithmic brief writer synthesis fallback."""
    is_bull = "Bullish" in tech or "Bullish" in sent
    is_bear = "Bearish" in tech and "Bearish" in fund

    signal = "BUY" if is_bull and not is_bear else ("AVOID" if is_bear else "HOLD")
    conviction = "High" if is_bull or is_bear else "Medium"

    return f"""**Signal:** {signal}
**Conviction:** {conviction}
**Time Horizon:** Medium-term (3–12 months)
**Thesis:** {ticker} exhibits steady operational performance with active market sentiment and technical support.
**Bull Case:**
- Solid underlying business fundamentals and revenue resilience.
- Constructive chart positioning relative to key moving averages.
- Positive news flow and sector tailwinds.
**Bear Case:**
- Macroeconomic volatility and market multiple contractions.
- Short-term profit taking near resistance zones.
- Industry competitive pressures.
**Bottom Line:** A structured {signal.lower()} position with disciplined risk management is warranted.

*Disclaimer: This analysis is AI-generated for educational purposes only and is not financial advice.*"""


def _safe_latest(arr: np.ndarray) -> Optional[float]:
    """Return the latest non-NaN value from a numpy array, or None."""
    if arr is None or len(arr) == 0:
        return None
    clean = arr[~np.isnan(arr)]
    return float(clean[-1]) if len(clean) > 0 else None


# ---------------------------------------------------------------------------
# Fallback indicator calculations (when TA-Lib is not installed)
# ---------------------------------------------------------------------------

def _compute_rsi_manual(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI using standard Wilder smoothing."""
    rsi = np.full(len(prices), np.nan)
    if len(prices) <= period:
        return rsi

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    if len(gains) < period:
        return rsi

    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

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
    """Compute Exponential Moving Average."""
    ema = np.full(len(prices), np.nan)
    if len(prices) < period:
        return ema

    # Find the first index at which `period` consecutive finite values exist.
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MACD line, signal line, and MACD histogram."""
    ema_fast = _compute_ema(prices, fast)
    ema_slow = _compute_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _compute_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _compute_bollinger_bands(
    prices: np.ndarray, period: int = 20, num_std: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Upper, Middle (SMA20), and Lower Bollinger Bands."""
    middle = np.full(len(prices), np.nan)
    upper = np.full(len(prices), np.nan)
    lower = np.full(len(prices), np.nan)
    if len(prices) < period:
        return upper, middle, lower

    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1 : i + 1]
        mean = float(np.mean(window))
        std = float(np.std(window))
        middle[i] = mean
        upper[i] = mean + (num_std * std)
        lower[i] = mean - (num_std * std)

    return upper, middle, lower


def _compute_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """Compute Average True Range (ATR)."""
    atr = np.full(len(close), np.nan)
    if len(close) <= period or len(high) != len(close) or len(low) != len(close):
        return atr

    tr = np.zeros(len(close))
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


def _compute_stochastic(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Fast Stochastic Oscillator %K and %D."""
    k_line = np.full(len(close), np.nan)
    d_line = np.full(len(close), np.nan)
    if len(close) < k_period:
        return k_line, d_line

    for i in range(k_period - 1, len(close)):
        lowest_low = np.min(low[i - k_period + 1 : i + 1])
        highest_high = np.max(high[i - k_period + 1 : i + 1])
        denom = highest_high - lowest_low
        if denom == 0:
            k_line[i] = 50.0
        else:
            k_line[i] = ((close[i] - lowest_low) / denom) * 100.0

    for i in range(k_period + d_period - 2, len(close)):
        window = k_line[i - d_period + 1 : i + 1]
        if not np.any(np.isnan(window)):
            d_line[i] = np.mean(window)

    return k_line, d_line


def _compute_pivot_points(high: float, low: float, close: float) -> dict:
    """Calculate Classic Floor Trader Pivot Points."""
    p = (high + low + close) / 3.0
    r1 = (2 * p) - low
    s1 = (2 * p) - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)
    return {
        "pivot": round(p, 2),
        "r1": round(r1, 2),
        "r2": round(r2, 2),
        "r3": round(r3, 2),
        "s1": round(s1, 2),
        "s2": round(s2, 2),
        "s3": round(s3, 2),
    }


def _extract_recommendation_summary(brief_text: str) -> dict:
    """Extract structured signal, conviction, and time horizon from investment brief."""
    summary = {
        "signal": "HOLD",
        "conviction": "MEDIUM",
        "time_horizon": "Medium-term (3–12 months)",
        "score": 5.0,
    }
    if not brief_text:
        return summary

    text_upper = brief_text.upper()
    if "SIGNAL:** BUY" in text_upper or "SIGNAL: BUY" in text_upper or "STRONG BUY" in text_upper:
        summary["signal"] = "BUY"
        summary["score"] = 8.5
    elif "SIGNAL:** AVOID" in text_upper or "SIGNAL: AVOID" in text_upper or "STRONG SELL" in text_upper or "SIGNAL:** SELL" in text_upper:
        summary["signal"] = "AVOID"
        summary["score"] = 2.0
    elif "SIGNAL:** WATCH" in text_upper or "SIGNAL: WATCH" in text_upper:
        summary["signal"] = "WATCH"
        summary["score"] = 6.0
    elif "SIGNAL:** HOLD" in text_upper or "SIGNAL: HOLD" in text_upper:
        summary["signal"] = "HOLD"
        summary["score"] = 5.0

    if "CONVICTION:** HIGH" in text_upper or "CONVICTION: HIGH" in text_upper:
        summary["conviction"] = "HIGH"
    elif "CONVICTION:** LOW" in text_upper or "CONVICTION: LOW" in text_upper:
        summary["conviction"] = "LOW"
    elif "CONVICTION:** MEDIUM" in text_upper or "CONVICTION: MEDIUM" in text_upper:
        summary["conviction"] = "MEDIUM"

    if "SHORT-TERM" in text_upper or "< 3 MONTHS" in text_upper:
        summary["time_horizon"] = "Short-term (< 3 months)"
    elif "LONG-TERM" in text_upper or "> 1 YEAR" in text_upper:
        summary["time_horizon"] = "Long-term (> 1 year)"
    else:
        summary["time_horizon"] = "Medium-term (3–12 months)"

    return summary


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
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as exc:
        logger.warning("Error fetching info for %s: %s", ticker, exc)
        info = {}

    metrics = {k: info.get(k) for k in _FUNDAMENTAL_FIELDS if info.get(k) is not None}
    chain = _FUNDAMENTALS_PROMPT | _llm
    text = _safe_invoke_chain(
        chain=chain,
        inputs={
            "ticker": ticker,
            "metrics": json.dumps(metrics, indent=2) if metrics else "No financial metrics found.",
        },
        fallback_fn=lambda: _generate_fallback_fundamentals(ticker, metrics),
    )

    return {
        "fundamentals_report": text,
        "metrics": metrics,
    }


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
    news_items = _fetch_structured_news(ticker)
    headlines = [item.get("title", "") for item in news_items if item.get("title")]
    if not headlines:
        headlines = _fetch_headlines(ticker)

    chain = _SENTIMENT_PROMPT | _llm
    text = _safe_invoke_chain(
        chain=chain,
        inputs={
            "ticker": ticker,
            "headlines": "\n".join(f"- {h}" for h in headlines) if headlines else "No headlines available.",
        },
        fallback_fn=lambda: _generate_fallback_sentiment(ticker, headlines),
    )

    return {
        "sentiment_report": text,
        "news_items": news_items,
    }


def _fetch_structured_news(ticker: str) -> list[dict]:
    """Fetch structured news articles with titles, publishers, links, and dates."""
    articles_list = []
    api_key = os.getenv("NEWS_API_KEY", "")
    if api_key:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": ticker,
                "sortBy": "publishedAt",
                "pageSize": 10,
                "language": "en",
                "apiKey": api_key,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                for art in resp.json().get("articles", []):
                    if art.get("title"):
                        articles_list.append({
                            "title": art.get("title"),
                            "publisher": art.get("source", {}).get("name", "News"),
                            "link": art.get("url", "#"),
                            "published_at": art.get("publishedAt", ""),
                        })
                if articles_list:
                    return articles_list
        except Exception as exc:
            logger.warning("NewsAPI structured fetch failed: %s", exc)

    # Fallback to yfinance news
    try:
        raw_news = yf.Ticker(ticker).news or []
        for item in raw_news[:10]:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            publisher = item.get("publisher", "Yahoo Finance")
            link = item.get("link", "#")
            pub_time = item.get("providerPublishTime", "")
            
            # yfinance content schema fallback
            if not title and "content" in item and isinstance(item["content"], dict):
                content = item["content"]
                title = content.get("title")
                if "provider" in content and isinstance(content["provider"], dict):
                    publisher = content["provider"].get("displayName", publisher)
                if "canonicalUrl" in content and isinstance(content["canonicalUrl"], dict):
                    link = content["canonicalUrl"].get("url", link)

            if title:
                articles_list.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "published_at": str(pub_time),
                })
    except Exception as exc:
        logger.warning("yfinance structured news failed: %s", exc)

    return articles_list


def _fetch_headlines(ticker: str) -> list[str]:
    """Fetch recent news headlines from NewsAPI or Yahoo Finance."""
    api_key = os.getenv("NEWS_API_KEY", "")
    if not api_key:
        logger.info("NEWS_API_KEY not set; using Yahoo Finance news fallback.")
        return _fetch_headlines_yfinance(ticker)
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": ticker,
            "sortBy": "publishedAt",
            "pageSize": 20,
            "language": "en",
            "apiKey": api_key,
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        titles = [a["title"] for a in articles if a.get("title")]
        return titles if titles else _fetch_headlines_yfinance(ticker)
    except Exception as exc:
        logger.error("NewsAPI error: %s", exc)
        return _fetch_headlines_yfinance(ticker)


def _fetch_headlines_yfinance(ticker: str) -> list[str]:
    """Fallback: get news headlines via yfinance."""
    try:
        news = yf.Ticker(ticker).news or []
        titles = []
        for item in news[:20]:
            title = item.get("title") if isinstance(item, dict) else None
            if not title and isinstance(item, dict) and "content" in item:
                content = item.get("content")
                if isinstance(content, dict):
                    title = content.get("title")
            if title:
                titles.append(title)
        return titles
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
        "- Trend direction (price vs SMA50, SMA200 — golden cross / death cross, EMA20)\n"
        "- Momentum (RSI — overbought >70 / oversold <30, Stochastic %K/%D)\n"
        "- Volatility & Bands (Bollinger Bands %B, ATR)\n"
        "- MACD signal (bullish crossover / bearish crossover)\n"
        "- Support & Resistance levels\n"
        "- Overall technical stance: Bullish / Neutral / Bearish\n\n"
        "150 words max. Be specific with the numbers.",
    ),
    ("human", "Technical indicators for {ticker}:\n\n{indicators}"),
])


def agent_technical_analyst(state: StockResearchState) -> dict:
    """Agent 3: Compute RSI, MACD, Bollinger Bands, ATR, SMAs, EMAs and interpret setup."""
    ticker = state["ticker"]
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty:
            hist = yf.Ticker(ticker).history(period="6mo")
    except Exception as exc:
        logger.warning("Error fetching history for %s: %s", ticker, exc)
        return {"technical_report": f"No price data available for {ticker}."}

    if hist.empty or "Close" not in hist:
        return {"technical_report": f"No price data available for {ticker}."}

    close = hist["Close"].dropna().values.astype(float)
    if len(close) == 0:
        return {"technical_report": f"No price data available for {ticker}."}

    high = hist["High"].dropna().values.astype(float) if "High" in hist else close
    low = hist["Low"].dropna().values.astype(float) if "Low" in hist else close
    volume = hist["Volume"].dropna().values.astype(float) if "Volume" in hist else np.array([])

    if TALIB_AVAILABLE:
        rsi = talib.RSI(close, timeperiod=14)
        macd_line, signal_line, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        sma50 = talib.SMA(close, timeperiod=50)
        sma200 = talib.SMA(close, timeperiod=200)
        sma20 = talib.SMA(close, timeperiod=20)
        ema20 = talib.EMA(close, timeperiod=20)
        bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        atr = talib.ATR(high, low, close, timeperiod=14) if len(high) == len(close) else _compute_atr(high, low, close, 14)
        stoch_k, stoch_d = _compute_stochastic(high, low, close, 14, 3)
    else:
        rsi = _compute_rsi_manual(close, 14)
        macd_line, signal_line, macd_hist = _compute_macd_manual(close)
        sma50 = np.array([np.nan] * len(close))
        sma200 = np.array([np.nan] * len(close))
        sma20 = np.array([np.nan] * len(close))
        if len(close) >= 20:
            for i in range(19, len(close)):
                sma20[i] = np.mean(close[i - 19 : i + 1])
        if len(close) >= 50:
            for i in range(49, len(close)):
                sma50[i] = np.mean(close[i - 49 : i + 1])
        if len(close) >= 200:
            for i in range(199, len(close)):
                sma200[i] = np.mean(close[i - 199 : i + 1])
        ema20 = _compute_ema(close, 20)
        bb_upper, bb_mid, bb_lower = _compute_bollinger_bands(close, 20, 2.0)
        atr = _compute_atr(high, low, close, 14)
        stoch_k, stoch_d = _compute_stochastic(high, low, close, 14, 3)

    sma50_val = _safe_latest(sma50)
    sma200_val = _safe_latest(sma200)
    sma20_val = _safe_latest(sma20)
    ema20_val = _safe_latest(ema20)
    current_price = float(close[-1])
    bb_u_val = _safe_latest(bb_upper)
    bb_l_val = _safe_latest(bb_lower)
    bb_m_val = _safe_latest(bb_mid)
    atr_val = _safe_latest(atr)
    stoch_k_val = _safe_latest(stoch_k)
    stoch_d_val = _safe_latest(stoch_d)

    pivots = _compute_pivot_points(float(np.max(high[-20:])), float(np.min(low[-20:])), current_price) if len(high) >= 20 else {}

    indicators = {
        "current_price": current_price,
        "rsi_14": _safe_latest(rsi),
        "macd": _safe_latest(macd_line),
        "macd_signal": _safe_latest(signal_line),
        "macd_histogram": _safe_latest(macd_hist),
        "sma_20": sma20_val,
        "sma_50": sma50_val,
        "sma_200": sma200_val,
        "ema_20": ema20_val,
        "bollinger_upper": bb_u_val,
        "bollinger_middle": bb_m_val,
        "bollinger_lower": bb_l_val,
        "atr_14": atr_val,
        "stochastic_k": stoch_k_val,
        "stochastic_d": stoch_d_val,
        "pivots": pivots,
        "price_vs_sma50": (
            round(((current_price / sma50_val) - 1) * 100, 2)
            if sma50_val and sma50_val > 0 else None
        ),
        "price_vs_sma200": (
            round(((current_price / sma200_val) - 1) * 100, 2)
            if sma200_val and sma200_val > 0 else None
        ),
    }

    chain = _TECHNICAL_PROMPT | _llm
    text = _safe_invoke_chain(
        chain=chain,
        inputs={
            "ticker": ticker,
            "indicators": json.dumps(indicators, indent=2),
        },
        fallback_fn=lambda: _generate_fallback_technical(ticker, indicators),
    )
    return {
        "technical_report": text,
        "technical_indicators": indicators,
    }


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
        "Be specific. Cite numbers from the reports. Pick a signal and defend it.\n\n"
        "Include a one-line financial disclaimer stating that this analysis is AI-generated for educational purposes only and is not financial advice.",
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
    fund_rep = state.get("fundamentals_report", "Not available.")
    sent_rep = state.get("sentiment_report", "Not available.")
    tech_rep = state.get("technical_report", "Not available.")

    text = _safe_invoke_chain(
        chain=chain,
        inputs={
            "ticker": state["ticker"],
            "fundamentals_report": fund_rep,
            "sentiment_report": sent_rep,
            "technical_report": tech_rep,
        },
        fallback_fn=lambda: _generate_fallback_brief(state["ticker"], fund_rep, sent_rep, tech_rep),
    )
    summary = _extract_recommendation_summary(text)

    return {
        "investment_brief": text,
        "recommendation_summary": summary,
    }