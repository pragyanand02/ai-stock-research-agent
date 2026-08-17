from unittest.mock import MagicMock, patch

from resolver import resolve_ticker


@patch("resolver.yf.Ticker")
def test_resolve_existing_ticker_camel_case(mock_ticker):
    mock_ticker.return_value.fast_info = {
        "lastPrice": 100.0
    }

    result = resolve_ticker("AAPL")

    assert result == {
        "ticker": "AAPL",
        "source": "raw",
    }


@patch("resolver.yf.Ticker")
def test_resolve_existing_ticker_previous_close(mock_ticker):
    mock_ticker.return_value.fast_info = {
        "previousClose": 150.0
    }

    result = resolve_ticker("MSFT")

    assert result == {
        "ticker": "MSFT",
        "source": "raw",
    }


@patch("resolver.yf.Search")
@patch("resolver.yf.Ticker")
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


@patch("resolver._get_llm")
@patch("resolver.yf.Search")
@patch("resolver.yf.Ticker")
def test_resolve_with_gemini_fallback(mock_ticker, mock_search, mock_get_llm):
    mock_ticker.return_value.fast_info = {}
    mock_search.return_value.quotes = []

    mock_response = MagicMock()
    mock_response.text = (
        '{"ticker": "RELIANCE.NS", '
        '"name": "Reliance Industries Limited"}'
    )
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    result = resolve_ticker("Reliance Industries")

    assert result == {
        "ticker": "RELIANCE.NS",
        "name": "Reliance Industries Limited",
        "source": "gemini",
    }


@patch("resolver._get_llm")
@patch("resolver.yf.Search")
@patch("resolver.yf.Ticker")
def test_resolve_with_gemini_markdown_fences(mock_ticker, mock_search, mock_get_llm):
    mock_ticker.return_value.fast_info = {}
    mock_search.return_value.quotes = []

    mock_response = MagicMock()
    mock_response.text = '```json\n{"ticker": "TATAMOTORS.NS", "name": "Tata Motors"}\n```'
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    result = resolve_ticker("Tata Motors")

    assert result == {
        "ticker": "TATAMOTORS.NS",
        "name": "Tata Motors",
        "source": "gemini",
    }


@patch("resolver._get_llm")
@patch("resolver.yf.Search")
@patch("resolver.yf.Ticker")
def test_resolve_raw_fallback(mock_ticker, mock_search, mock_get_llm):
    mock_ticker.return_value.fast_info = {}
    mock_search.return_value.quotes = []

    mock_response = MagicMock()
    mock_response.text = '{"ticker": null, "name": null}'
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    result = resolve_ticker("UNKNOWN")

    assert result == {
        "ticker": "UNKNOWN",
        "source": "raw_fallback",
    }


@patch("resolver.yf.Search")
@patch("resolver.yf.Ticker")
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


@patch("resolver._get_llm")
@patch("resolver.yf.Search")
@patch("resolver.yf.Ticker")
def test_gemini_error_uses_raw_fallback(mock_ticker, mock_search, mock_get_llm):
    mock_ticker.return_value.fast_info = {}
    mock_search.return_value.quotes = []

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("Gemini API error")
    mock_get_llm.return_value = mock_llm

    result = resolve_ticker("UNKNOWN")

    assert result == {
        "ticker": "UNKNOWN",
        "source": "raw_fallback",
    }


def test_empty_input():
    result = resolve_ticker("")
    assert result == {"ticker": "", "source": "empty"}