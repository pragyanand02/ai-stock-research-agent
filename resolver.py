import json
import logging
import os
import re
import yfinance as yf
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logger = logging.getLogger(__name__)
_llm = None

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


def _get_llm(model: str = None):
    global _llm
    target_model = model or GEMINI_MODEL
    if model is not None or _llm is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        llm_instance = ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=api_key or None,
            temperature=0,
        )
        if model is None:
            _llm = llm_instance
        return llm_instance
    return _llm


def _clean_json_response(text: str) -> str:
    """Clean markdown code fences from LLM output."""
    clean = text.strip()
    match = re.search(r"\{[^{}]*\}", clean)
    if match:
        return match.group(0)
    if clean.startswith("```json"):
        clean = clean[7:]
    elif clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return clean.strip()


def resolve_ticker(user_input: str) -> dict:
    """
    Resolve a user-provided stock input to a Yahoo Finance ticker.
    """
    raw_input = user_input.strip().upper()
    if not raw_input:
        return {"ticker": "", "source": "empty"}

    # Fast Path: check whether the input is already a valid ticker (no spaces).
    if " " not in raw_input:
        try:
            ticker = yf.Ticker(raw_input)
            fast_info = ticker.fast_info

            # yfinance FastInfo uses lastPrice / previousClose
            last_price = None
            if hasattr(fast_info, "get"):
                last_price = (
                    fast_info.get("lastPrice")
                    or fast_info.get("last_price")
                    or fast_info.get("previousClose")
                    or fast_info.get("regularMarketPreviousClose")
                )
            elif hasattr(fast_info, "lastPrice"):
                last_price = getattr(fast_info, "lastPrice", None) or getattr(fast_info, "last_price", None)

            if last_price is not None:
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
    from agents import _extract_text
    prompt_str = f"""
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
    for model_name in [GEMINI_MODEL, "gemini-flash-lite-latest"]:
        try:
            llm = _get_llm(model=model_name)
            response = llm.invoke(prompt_str)

            text = _extract_text(response)
            cleaned_text = _clean_json_response(text)
            data = json.loads(cleaned_text)

            if data.get("ticker"):
                return {
                    "ticker": data["ticker"],
                    "name": data.get("name"),
                    "source": "gemini",
                }
        except Exception as exc:
            logger.warning("Gemini ticker resolution failed with model %s: %s", model_name, exc)

    # Raw fallback.
    return {
        "ticker": raw_input,
        "source": "raw_fallback",
    }