from unittest.mock import MagicMock, patch

from ticker_resolver import resolve_ticker


@patch("ticker_resolver.yf.Ticker")
def test_resolve_existing_ticker(mock_ticker):
    mock_ticker.return_value.fast_info = {
        "last_price": 100.0
    }

    result = resolve_ticker("AAPL")

    assert result == {
        "ticker": "AAPL",
        "source": "raw",
    }


@patch("ticker_resolver.yf.Search")
@patch("ticker_resolver.yf.Ticker")
def test_resolve_company_name_with_yfinance_search(mock_ticker, mock_search):
    mock_ticker.return_value.fast_info = {}

    mock_search.return_value.quotes = [
        {
            "quoteType": "EQUITY",
            "symbol": "AAPL",
            "shortname": "Apple Inc.",
        }
    ]

    result = resolve_ticker("Apple")

    assert result == {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "source": "search",
    }


@patch("ticker_resolver._llm")
@patch("ticker_resolver.yf.Search")
@patch("ticker_resolver.yf.Ticker")
def test_resolve_with_gemini_fallback(mock_ticker, mock_search, mock_llm):
    mock_ticker.return_value.fast_info = {}
    mock_search.return_value.quotes = []

    mock_response = MagicMock()
    mock_response.text = (
        '{"ticker": "RELIANCE.NS", '
        '"name": "Reliance Industries Limited"}'
    )
    mock_llm.invoke.return_value = mock_response

    result = resolve_ticker("Reliance Industries")

    assert result == {
        "ticker": "RELIANCE.NS",
        "name": "Reliance Industries Limited",
        "source": "gemini",
    }


@patch("ticker_resolver._llm")
@patch("ticker_resolver.yf.Search")
@patch("ticker_resolver.yf.Ticker")
def test_resolve_raw_fallback(mock_ticker, mock_search, mock_llm):
    mock_ticker.return_value.fast_info = {}
    mock_search.return_value.quotes = []

    mock_response = MagicMock()
    mock_response.text = '{"ticker": null, "name": null}'
    mock_llm.invoke.return_value = mock_response

    result = resolve_ticker("UNKNOWN")

    assert result == {
        "ticker": "UNKNOWN",
        "source": "raw_fallback",
    }


@patch("ticker_resolver.yf.Search")
@patch("ticker_resolver.yf.Ticker")
def test_search_ignores_non_equity_results(mock_ticker, mock_search):
    mock_ticker.return_value.fast_info = {}

    mock_search.return_value.quotes = [
        {
            "quoteType": "ETF",
            "symbol": "QQQ",
            "shortname": "Invesco QQQ",
        },
        {
            "quoteType": "EQUITY",
            "symbol": "AAPL",
            "shortname": "Apple Inc.",
        },
    ]

    result = resolve_ticker("Apple")

    assert result == {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "source": "search",
    }


@patch("ticker_resolver._llm")
@patch("ticker_resolver.yf.Search")
@patch("ticker_resolver.yf.Ticker")
def test_gemini_error_uses_raw_fallback(mock_ticker, mock_search, mock_llm):
    mock_ticker.return_value.fast_info = {}
    mock_search.return_value.quotes = []

    mock_llm.invoke.side_effect = Exception("Gemini API error")

    result = resolve_ticker("UNKNOWN")

    assert result == {
        "ticker": "UNKNOWN",
        "source": "raw_fallback",
    }


@patch("ticker_resolver._llm")
@patch("ticker_resolver.yf.Search")
@patch("ticker_resolver.yf.Ticker")
def test_search_error_uses_gemini(mock_ticker, mock_search, mock_llm):
    mock_ticker.return_value.fast_info = {}

    mock_search.side_effect = Exception("Yahoo Finance error")

    mock_response = MagicMock()
    mock_response.text = (
        '{"ticker": "TSLA", '
        '"name": "Tesla, Inc."}'
    )
    mock_llm.invoke.return_value = mock_response

    result = resolve_ticker("Tesla")

    assert result == {
        "ticker": "TSLA",
        "name": "Tesla, Inc.",
        "source": "gemini",
    }