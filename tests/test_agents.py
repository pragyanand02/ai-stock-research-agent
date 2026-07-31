import os

os.environ["GOOGLE_API_KEY"] = "test-key"

from unittest.mock import MagicMock, patch

from agents import (
    agent_fundamentals_analyst,
    agent_sentiment_scanner,
    agent_technical_analyst,
)


@patch("agents._llm")
@patch("agents.yf.Ticker")
def test_fundamentals_agent(mock_ticker, mock_llm):
    mock_ticker.return_value.info = {
        "trailingPE": 20,
        "marketCap": 1000000,
        "currentPrice": 100,
    }

    mock_response = MagicMock()
    mock_response.text = "Fundamental Analysis"
    mock_llm.invoke.return_value = mock_response

    result = agent_fundamentals_analyst({"ticker": "AAPL"})

    assert "fundamentals_report" in result


@patch("agents._fetch_headlines")
@patch("agents._llm")
def test_sentiment_agent(mock_llm, mock_headlines):
    mock_headlines.return_value = ["Apple launches new AI feature"]

    mock_response = MagicMock()
    mock_response.text = "Bullish"
    mock_llm.invoke.return_value = mock_response

    result = agent_sentiment_scanner({"ticker": "AAPL"})

    assert "sentiment_report" in result


@patch("agents._llm")
@patch("agents.yf.Ticker")
def test_technical_agent(mock_ticker, mock_llm):
    import pandas as pd
    import numpy as np

    mock_ticker.return_value.history.return_value = pd.DataFrame(
        {
            "Close": np.linspace(100, 150, 250)
        }
    )

    mock_response = MagicMock()
    mock_response.text = "Technical Analysis"
    mock_llm.invoke.return_value = mock_response

    result = agent_technical_analyst({"ticker": "AAPL"})

    assert "technical_report" in result