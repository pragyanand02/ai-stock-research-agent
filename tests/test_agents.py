import os
os.environ["GOOGLE_API_KEY"] = "test-key"

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage

from agents import (
    agent_fundamentals_analyst,
    agent_sentiment_scanner,
    agent_technical_analyst,
    agent_brief_writer,
    _compute_rsi_manual,
    _compute_ema,
    _compute_macd_manual,
    _safe_latest,
)


@patch("agents._llm")
@patch("agents.yf.Ticker")
def test_fundamentals_agent(mock_ticker, mock_llm):
    mock_ticker.return_value.info = {
        "trailingPE": 20,
        "marketCap": 1000000,
        "currentPrice": 100,
    }

    mock_response = AIMessage(content="Fundamental Analysis")
    mock_llm.invoke.return_value = mock_response
    mock_llm.return_value = mock_response

    result = agent_fundamentals_analyst({"ticker": "AAPL"})

    assert "fundamentals_report" in result
    assert result["fundamentals_report"] == "Fundamental Analysis"


@patch("agents._fetch_headlines")
@patch("agents._llm")
def test_sentiment_agent(mock_llm, mock_headlines):
    mock_headlines.return_value = ["Apple launches new AI feature"]

    mock_response = AIMessage(content="Bullish")
    mock_llm.invoke.return_value = mock_response
    mock_llm.return_value = mock_response

    result = agent_sentiment_scanner({"ticker": "AAPL"})

    assert "sentiment_report" in result
    assert result["sentiment_report"] == "Bullish"


@patch("agents._llm")
@patch("agents.yf.Ticker")
def test_technical_agent(mock_ticker, mock_llm):
    mock_ticker.return_value.history.return_value = pd.DataFrame(
        {
            "Close": np.linspace(100, 150, 250)
        }
    )

    mock_response = AIMessage(content="Technical Analysis")
    mock_llm.invoke.return_value = mock_response
    mock_llm.return_value = mock_response

    result = agent_technical_analyst({"ticker": "AAPL"})

    assert "technical_report" in result
    assert result["technical_report"] == "Technical Analysis"


@patch("agents._llm_writer")
def test_brief_writer_agent(mock_llm_writer):
    mock_response = AIMessage(content="**Signal:** Buy\n**Conviction:** High\n**Thesis:** Strong growth.")
    mock_llm_writer.invoke.return_value = mock_response
    mock_llm_writer.return_value = mock_response

    state = {
        "ticker": "AAPL",
        "fundamentals_report": "Strong balance sheet.",
        "sentiment_report": "Positive sentiment.",
        "technical_report": "Bullish trend above SMA 50.",
    }

    result = agent_brief_writer(state)

    assert "investment_brief" in result
    assert "**Signal:** Buy" in result["investment_brief"]


def test_compute_rsi_manual():
    prices = np.linspace(10, 50, 30)
    rsi = _compute_rsi_manual(prices, period=14)
    # Monotonically increasing prices should give RSI of 100
    assert not np.isnan(rsi[-1])
    assert rsi[-1] == 100.0


def test_compute_ema():
    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    ema = _compute_ema(prices, period=3)
    assert not np.isnan(ema[-1])
    assert ema[-1] > 10.0


def test_compute_macd_manual():
    prices = np.linspace(50, 100, 50)
    macd_line, signal_line, hist = _compute_macd_manual(prices)
    assert len(macd_line) == 50
    assert len(signal_line) == 50
    assert len(hist) == 50
    assert not np.isnan(macd_line[-1])


def test_safe_latest():
    arr = np.array([np.nan, 10.5, np.nan, 25.3, np.nan])
    assert _safe_latest(arr) == 25.3

    empty_arr = np.array([np.nan, np.nan])
    assert _safe_latest(empty_arr) is None