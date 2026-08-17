"""
Project 04 — Streamlit UI for the AI Stock Research & Investment Brief Generator.

Run:  streamlit run app.py
"""
import os
import sys

from dotenv import load_dotenv
import streamlit as st
import yfinance as yf

# Ensure local imports work reliably
sys.path.insert(0, os.path.dirname(__file__))

from graph import graph
from resolver import resolve_ticker

load_dotenv()

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def cached_resolve_ticker(user_input: str) -> dict:
    return resolve_ticker(user_input)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def run_research(ticker: str) -> dict:
    return graph.invoke({"ticker": ticker})


def get_currency_symbol(currency_code: str) -> str:
    if not currency_code:
        return "$"
    symbols = {
        "USD": "$",
        "INR": "₹",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "CAD": "CA$",
        "AUD": "A$",
        "CNY": "¥",
    }
    return symbols.get(currency_code.upper(), f"{currency_code} ")


st.set_page_config(
    page_title="AI Stock Research Brief",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 AI Stock Research & Investment Brief Generator")
st.caption("Powered by LangGraph • Google Gemini • Yahoo Finance • Streamlit")
st.warning(
    "⚠️ This is AI-generated analysis for educational purposes only. "
    "It is not financial advice. Always do your own research and consult "
    "a licensed financial advisor before making investment decisions."
)
st.markdown(
    "Enter a stock ticker or company name. Three research agents run in **parallel** (fundamentals, news sentiment, "
    "technicals), then an AI brief writer synthesizes an opinionated one-page investment brief."
)

with st.sidebar:
    st.header("Stock Lookup")
    ticker_input = st.text_input(
        "Company Name or Ticker Symbol",
        placeholder="e.g. Apple, Tata Motors, MSFT, RELIANCE.NS",
        help="Use .NS for NSE India, .BO for BSE India",
    ).strip()
    time_period = st.selectbox(
        "Chart Time Period",
        ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=1,
    )
    run_button = st.button("🔍 Generate Brief", type="primary", disabled=(not ticker_input))
    st.markdown("---")
    st.markdown(
        "**Examples:** Apple · Tata Motors · MSFT · GOOGL · RELIANCE.NS · INFY.NS"
    )

if run_button and ticker_input:
    with st.spinner(f"Resolving **{ticker_input}** to a ticker symbol..."):
        resolved = cached_resolve_ticker(ticker_input)

    resolved_ticker = resolved.get("ticker", "").strip()
    if resolved.get("source") in ("search", "gemini"):
        company_name = resolved.get("name")
        if company_name:
            st.caption(
                f'🔎 Resolved "{ticker_input}" → **{resolved_ticker}** ({company_name})'
            )
        else:
            st.caption(
                f'🔎 Resolved "{ticker_input}" → **{resolved_ticker}**'
            )

    if not resolved_ticker:
        st.error("❌ Could not resolve the stock ticker.")
        st.stop()

    with st.spinner(f"Running 3 parallel research agents for **{resolved_ticker}**… (30–60 seconds)"):
        stock = yf.Ticker(resolved_ticker)
        try:
            hist = stock.history(period=time_period)
        except Exception as exc:
            st.error(f"❌ Error fetching price data: {exc}")
            st.stop()

        if hist.empty:
            st.error("❌ Invalid ticker or no market data available.")
            st.stop()

        try:
            result = run_research(resolved_ticker)
        except Exception as e:
            err_str = str(e)
            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                st.error("⚠️ Gemini API rate limit exceeded. Please wait 1 minute and try again.")
            else:
                st.error(f"❌ Error during research: {e}")
            st.stop()

    # --- Company Header & Price Chart ---
    try:
        info = stock.info or {}
        company_name = info.get("longName") or info.get("shortName") or resolved.get("name") or resolved_ticker
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        high_52w = info.get("fiftyTwoWeekHigh", "N/A")
        low_52w = info.get("fiftyTwoWeekLow", "N/A")
        currency = info.get("currency", "USD")
        curr_sym = get_currency_symbol(currency)

        current_price = float(hist["Close"].iloc[-1])
        st.subheader(f"🏢 {company_name}")
        st.caption(f"Sector: {sector} | Industry: {industry}")

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("📈 Current Stock Price", f"{curr_sym}{current_price:.2f}")
        with m_col2:
            st.metric("52W High", f"{curr_sym}{high_52w}" if isinstance(high_52w, (int, float)) else f"{high_52w}")
        with m_col3:
            st.metric("52W Low", f"{curr_sym}{low_52w}" if isinstance(low_52w, (int, float)) else f"{low_52w}")

        st.divider()
        st.subheader("📉 Stock Price Trend")
        st.line_chart(hist["Close"])
        st.caption(
            f"Closing price for selected period: {time_period} "
            f"({hist.index.min().date()} → {hist.index.max().date()}) • {len(hist)} data points"
        )
    except Exception as exc:
        st.warning(f"⚠️ Could not load company overview details: {exc}")

    # --- Investment Brief (hero section) ---
    st.divider()
    st.success(f"Research complete for **{resolved_ticker}**!")
    st.toast("Research completed successfully! 🎉")

    st.subheader("📋 Investment Brief")
    st.caption("AI-generated summary synthesized from fundamentals, news sentiment, and technical indicators.")
    
    brief = result.get("investment_brief", "No brief generated.")
    st.markdown(brief)

    st.download_button(
        label="📥 Download Investment Brief",
        data=brief,
        file_name=f"{resolved_ticker}_investment_brief.txt",
        mime="text/plain",
        help="Download the AI-generated investment brief as a text file.",
        use_container_width=True,
    )
    st.caption("Generated by AI Stock Research & Investment Brief Generator")
    st.divider()

    # --- Three specialist reports ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏦 Fundamentals")
        st.caption("Company financial health, valuation, and key metrics")
        st.markdown(result.get("fundamentals_report", "No report available."))

    with col2:
        st.subheader("📰 News Sentiment")
        st.caption("Latest market news and AI sentiment analysis")
        st.markdown(result.get("sentiment_report", "No report available."))

    with col3:
        st.subheader("📊 Technical Analysis")
        st.caption("Technical indicators, trend analysis, and momentum")
        st.markdown(result.get("technical_report", "No report available."))