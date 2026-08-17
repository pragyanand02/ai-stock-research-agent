import json
import logging
import yfinance as yf
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)
_llm = None


def _get_llm():
    global _llm

    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=0,
        )

    return _llm


def resolve_ticker(user_input: str) -> dict:
    """
    Resolve a user-provided stock input to a Yahoo Finance ticker.
    """

    raw_input = user_input.strip().upper()

    # Fast Path: check whether the input is already a valid ticker.
    try:
        ticker = yf.Ticker(raw_input)
        fast_info = ticker.fast_info

        if fast_info and fast_info.get("last_price") is not None:
            return {
                "ticker": raw_input,
                "source": "raw",
            }

    except Exception as exc:
        logger.debug("Raw ticker validation failed: %s", exc)

    # Search Path: search for the company name or ticker.
    try:
        search = yf.Search(user_input, max_results=5)

        for quote in search.quotes:
            if quote.get("quoteType") == "EQUITY":
                symbol = quote.get("symbol")
                name = quote.get("shortname") or quote.get("longname")

                if symbol:
                    return {
                        "ticker": symbol,
                        "name": name,
                        "source": "search",
                    }

    except Exception as exc:
        logger.warning("Yahoo Finance ticker search failed: %s", exc)

    # Gemini Fallback.
    try:
        response = _get_llm().invoke(
            f"""
            Identify the Yahoo Finance stock ticker for this company:
            {user_input}
            Ticker conventions:
            - For NSE India stocks, use the .NS suffix.
            - For BSE India stocks, use the .BO suffix.
            - For US stocks, use no exchange suffix.

            Return ONLY valid JSON in this exact format:
            {{
                "ticker": "AAPL",
                "name": "Apple Inc."
            }}

            If you cannot identify the company, return:
            {{
                "ticker": null,
                "name": null
            }}
            """
        )

        text = response.text if hasattr(response, "text") else str(response.content)
        data = json.loads(text)

        if data.get("ticker"):
            return {
                "ticker": data["ticker"],
                "name": data.get("name"),
                "source": "gemini",
            }

    except Exception as exc:
        logger.warning("Gemini ticker resolution failed: %s", exc)
        pass

    # Raw fallback.
    return {
        "ticker": raw_input,
        "source": "raw_fallback",
    }