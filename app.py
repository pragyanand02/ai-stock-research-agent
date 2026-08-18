"""
Project 04 — Modern Streamlit UI for AI Stock Research & Investment Brief Generator.
Features:
- Tab 1: Single Stock Deep Dive with Interactive Plotly Candlestick Chart, Indicators, & PDF Export
- Tab 2: Multi-Stock Side-by-Side Comparison & Relative Return Chart
- Tab 3: Personal Watchlist & Market Quick Scanners
"""
import os
import sys
from typing import Dict, Any, List
import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

# Ensure local imports work reliably
sys.path.insert(0, os.path.dirname(__file__))

from graph import graph
from resolver import resolve_ticker
from charts import create_stock_chart, create_comparison_chart
from pdf_generator import generate_stock_pdf

load_dotenv()

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# Page Config
st.set_page_config(
    page_title="AI Stock Research & Investment Brief",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E88E5; margin-bottom: 0px; }
    .sub-header { color: #888888; font-size: 1.0rem; margin-bottom: 1.5rem; }
    .metric-card {
        background-color: rgba(30, 136, 229, 0.08);
        border: 1px solid rgba(30, 136, 229, 0.2);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .signal-buy {
        background-color: rgba(38, 166, 154, 0.15);
        border-left: 5px solid #26a69a;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .signal-avoid {
        background-color: rgba(239, 83, 80, 0.15);
        border-left: 5px solid #ef5350;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .signal-hold {
        background-color: rgba(255, 179, 0, 0.15);
        border-left: 5px solid #ffb300;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .news-card {
        padding: 10px 14px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def cached_resolve_ticker(user_input: str) -> dict:
    return resolve_ticker(user_input)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def run_research(ticker: str) -> dict:
    return graph.invoke({"ticker": ticker})


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_stock_history(ticker: str, period: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    return stock.history(period=period)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def fetch_stock_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


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


def format_large_number(num: Any, currency_sym: str = "$") -> str:
    if num is None or not isinstance(num, (int, float)):
        return "N/A"
    if abs(num) >= 1e12:
        return f"{currency_sym}{num/1e12:.2f}T"
    elif abs(num) >= 1e9:
        return f"{currency_sym}{num/1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{currency_sym}{num/1e6:.2f}M"
    return f"{currency_sym}{num:,.2f}"


# Initialize Watchlist in session state
if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = ["AAPL", "MSFT", "NVDA", "RELIANCE.NS", "TATAMOTORS.NS"]

# Header
st.markdown('<div class="main-header">📈 AI Stock Research & Investment Brief Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-agent intelligence powered by LangGraph, Google Gemini, and Yahoo Finance</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_single, tab_compare, tab_watchlist = st.tabs([
    "🔍 Single Stock Deep Dive",
    "⚖️ Multi-Stock Comparison",
    "⭐ Watchlist & Market Scans",
])


# ===========================================================================
# TAB 1: SINGLE STOCK DEEP DIVE
# ===========================================================================
with tab_single:
    with st.sidebar:
        st.header("Stock Configuration")
        ticker_input = st.text_input(
            "Company Name or Ticker Symbol",
            value="",
            placeholder="e.g. Apple, NVDA, Tata Motors, RELIANCE.NS",
            help="Supports US symbols (AAPL), Indian symbols (.NS for NSE, .BO for BSE), and company names.",
        ).strip()

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            time_period = st.selectbox(
                "Time Period",
                ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
                index=3,
            )
        with col_p2:
            chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"], index=0)

        st.subheader("Chart Indicators")
        c_sma = st.checkbox("Moving Averages (SMA 50, 200)", value=True)
        c_ema = st.checkbox("EMA 20", value=True)
        c_bb = st.checkbox("Bollinger Bands (20, 2)", value=True)
        c_vol = st.checkbox("Volume Panel", value=True)
        c_rsi = st.checkbox("RSI (14) Panel", value=True)
        c_macd = st.checkbox("MACD Panel", value=True)

        run_button = st.button("🚀 Analyze Stock", type="primary", use_container_width=True)

        st.divider()
        st.caption("Quick Presets:")
        preset_cols = st.columns(3)
        if preset_cols[0].button("NVDA", use_container_width=True):
            ticker_input = "NVDA"
            run_button = True
        if preset_cols[1].button("AAPL", use_container_width=True):
            ticker_input = "AAPL"
            run_button = True
        if preset_cols[2].button("RELIANCE", use_container_width=True):
            ticker_input = "RELIANCE.NS"
            run_button = True

    if not ticker_input and not run_button:
        st.info("💡 Enter a company name or ticker symbol in the sidebar and click **Analyze Stock** to generate an AI research brief.")
        
        # Feature Highlight Showcase
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown("### 🏦 Fundamental Analysis")
            st.write("Deep evaluation of valuation ratios (P/E, PEG, P/B), balance sheet debt, cash flows, and profit margins.")
        with col_f2:
            st.markdown("### 📰 Sentiment Intelligence")
            st.write("Scans market news and social narratives across the last 7 days to classify bullish/bearish momentum.")
        with col_f3:
            st.markdown("### 📊 Technical Setup")
            st.write("Calculates RSI, MACD, Bollinger Bands, ATR, SMA/EMA trends, and key support & resistance pivot points.")

    if run_button or ticker_input:
        if not ticker_input:
            st.warning("Please enter a stock ticker or company name.")
            st.stop()

        with st.spinner(f"Resolving **{ticker_input}** to market ticker..."):
            resolved = cached_resolve_ticker(ticker_input)

        resolved_ticker = resolved.get("ticker", "").strip()
        if not resolved_ticker:
            st.error(f"❌ Could not resolve '{ticker_input}' to a valid stock ticker. Please try a different symbol.")
            st.stop()

        company_resolved_name = resolved.get("name") or resolved_ticker

        with st.spinner(f"Running 3 parallel research agents for **{resolved_ticker}**..."):
            hist = fetch_stock_history(resolved_ticker, time_period)
            if hist is None or hist.empty:
                st.error(f"❌ No historical price data found for ticker '{resolved_ticker}'.")
                st.stop()

            info = fetch_stock_info(resolved_ticker)
            company_name = info.get("longName") or info.get("shortName") or company_resolved_name
            currency = info.get("currency", "USD")
            curr_sym = get_currency_symbol(currency)

            try:
                result = run_research(resolved_ticker)
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    st.error("⚠️ Gemini API rate limit exceeded. Please wait 1 minute and try again.")
                else:
                    st.error(f"❌ Pipeline execution error: {e}")
                st.stop()

        # Extract structured data
        metrics = result.get("metrics") or {}
        indicators = result.get("technical_indicators") or {}
        news_items = result.get("news_items") or []
        summary = result.get("recommendation_summary") or {}
        brief_text = result.get("investment_brief", "No brief generated.")

        current_price = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current_price
        day_change_pct = ((current_price / prev_close) - 1) * 100

        # Company Header Banner
        st.subheader(f"🏢 {company_name} ({resolved_ticker})")
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        st.caption(f"**Sector:** {sector} | **Industry:** {industry} | **Exchange:** {info.get('exchange', 'N/A')} | **Resolution:** {resolved.get('source', 'direct')}")

        # Metrics Bar
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Current Price", f"{curr_sym}{current_price:,.2f}", f"{day_change_pct:+.2f}%")
        with m2:
            st.metric("52W High", f"{curr_sym}{info.get('fiftyTwoWeekHigh', 'N/A')}")
        with m3:
            st.metric("52W Low", f"{curr_sym}{info.get('fiftyTwoWeekLow', 'N/A')}")
        with m4:
            st.metric("Trailing P/E", f"{info.get('trailingPE', 'N/A')}")
        with m5:
            st.metric("Market Cap", format_large_number(info.get('marketCap'), curr_sym))

        st.divider()

        # Recommendation Banner
        signal = summary.get("signal", "HOLD")
        conviction = summary.get("conviction", "MEDIUM")
        horizon = summary.get("time_horizon", "Medium-term")

        signal_class = "signal-buy" if signal == "BUY" else ("signal-avoid" if signal == "AVOID" else "signal-hold")
        st.markdown(
            f"""
            <div class="{signal_class}">
                <h3 style="margin:0; padding:0;">🎯 AI Recommendation: <strong>{signal}</strong> ({conviction} Conviction)</h3>
                <p style="margin:5px 0 0 0;"><strong>Target Horizon:</strong> {horizon} | <strong>Model Score:</strong> {summary.get('score', 5.0)}/10</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Interactive Chart
        st.subheader("📊 Interactive Technical Chart")
        fig = create_stock_chart(
            df=hist,
            ticker=resolved_ticker,
            company_name=company_name,
            chart_type=chart_type,
            show_sma=c_sma,
            show_ema=c_ema,
            show_bb=c_bb,
            show_volume=c_vol,
            show_rsi=c_rsi,
            show_macd=c_macd,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Executive Investment Brief & Export Options
        st.subheader("📋 Executive Investment Brief")
        st.markdown(brief_text)

        # Download Buttons: PDF & Text
        pdf_bytes = generate_stock_pdf(
            ticker=resolved_ticker,
            company_name=company_name,
            current_price=current_price,
            currency_symbol=curr_sym,
            brief_text=brief_text,
            fundamentals_text=result.get("fundamentals_report", ""),
            sentiment_text=result.get("sentiment_report", ""),
            technical_text=result.get("technical_report", ""),
            metrics=metrics,
            technical_indicators=indicators,
            recommendation=summary,
        )

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                label="📥 Download Executive PDF Report",
                data=pdf_bytes,
                file_name=f"{resolved_ticker}_investment_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
        with d_col2:
            st.download_button(
                label="📄 Download Text Summary (.txt)",
                data=brief_text,
                file_name=f"{resolved_ticker}_brief.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.divider()

        # Specialist Reports Columns
        rep_col1, rep_col2, rep_col3 = st.columns(3)

        with rep_col1:
            st.subheader("🏦 Fundamentals")
            st.markdown(result.get("fundamentals_report", "No report available."))
            if metrics:
                with st.expander("🔍 View Raw Financial Ratios"):
                    r_df = pd.DataFrame(
                        [{"Ratio / Metric": k, "Value": str(v)} for k, v in metrics.items()]
                    )
                    st.dataframe(r_df, hide_index=True, use_container_width=True)

        with rep_col2:
            st.subheader("📰 News Sentiment")
            st.markdown(result.get("sentiment_report", "No report available."))
            if news_items:
                with st.expander(f"📰 Recent News Articles ({len(news_items)})"):
                    for item in news_items:
                        title = item.get("title", "Untitled")
                        pub = item.get("publisher", "News")
                        link = item.get("link", "#")
                        st.markdown(
                            f"""
                            <div class="news-card">
                                <a href="{link}" target="_blank" style="text-decoration:none; font-weight:600; color:#42A5F5;">{title}</a>
                                <div style="font-size:0.8rem; color:#888; margin-top:2px;">Source: {pub}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        with rep_col3:
            st.subheader("📊 Technical Analysis")
            st.markdown(result.get("technical_report", "No report available."))
            if indicators:
                with st.expander("📈 Key Indicator Values"):
                    ind_rows = [
                        {"Indicator": "RSI (14)", "Value": f"{indicators.get('rsi_14', 'N/A')}"},
                        {"Indicator": "SMA 50", "Value": f"{indicators.get('sma_50', 'N/A')}"},
                        {"Indicator": "SMA 200", "Value": f"{indicators.get('sma_200', 'N/A')}"},
                        {"Indicator": "EMA 20", "Value": f"{indicators.get('ema_20', 'N/A')}"},
                        {"Indicator": "Bollinger Upper", "Value": f"{indicators.get('bollinger_upper', 'N/A')}"},
                        {"Indicator": "Bollinger Lower", "Value": f"{indicators.get('bollinger_lower', 'N/A')}"},
                        {"Indicator": "ATR (14)", "Value": f"{indicators.get('atr_14', 'N/A')}"},
                    ]
                    st.dataframe(pd.DataFrame(ind_rows), hide_index=True, use_container_width=True)


# ===========================================================================
# TAB 2: MULTI-STOCK COMPARISON
# ===========================================================================
with tab_compare:
    st.subheader("⚖️ Side-by-Side Multi-Stock Comparison")
    st.write("Compare financial metrics, valuation, and historical return across 2 to 5 stocks simultaneously.")

    comp_input = st.text_input(
        "Enter Comma-Separated Symbols or Names",
        value="AAPL, MSFT, GOOGL, NVDA",
        help="Example: AAPL, MSFT, GOOGL or RELIANCE.NS, TCS.NS, INFY.NS",
    )
    comp_period = st.selectbox(
        "Comparison Horizon",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3,
        key="comp_horizon",
    )

    if st.button("📊 Run Comparison", type="primary"):
        symbols = [s.strip() for s in comp_input.split(",") if s.strip()]
        if not symbols:
            st.warning("Please provide at least 2 tickers.")
        else:
            with st.spinner("Fetching comparison data across all tickers..."):
                comp_data = []
                hist_dict = {}

                for sym in symbols:
                    res = cached_resolve_ticker(sym)
                    resolved_sym = res.get("ticker", sym)
                    try:
                        stock = yf.Ticker(resolved_sym)
                        h = stock.history(period=comp_period)
                        if not h.empty and "Close" in h:
                            hist_dict[resolved_sym] = h
                            inf = stock.info or {}
                            c_price = float(h["Close"].iloc[-1])
                            f_price = float(h["Close"].iloc[0])
                            pct_ret = round(((c_price / f_price) - 1) * 100, 2)
                            comp_data.append({
                                "Symbol": resolved_sym,
                                "Name": inf.get("shortName") or res.get("name") or resolved_sym,
                                "Sector": inf.get("sector", "N/A"),
                                "Price": f"{get_currency_symbol(inf.get('currency', 'USD'))}{c_price:.2f}",
                                f"{comp_period} Return": f"{pct_ret:+.2f}%",
                                "Trailing P/E": inf.get("trailingPE", "N/A"),
                                "Forward P/E": inf.get("forwardPE", "N/A"),
                                "Market Cap": format_large_number(inf.get("marketCap"), get_currency_symbol(inf.get("currency", "USD"))),
                                "ROE": f"{inf.get('returnOnEquity', 0)*100:.1f}%" if inf.get("returnOnEquity") else "N/A",
                                "Profit Margin": f"{inf.get('profitMargins', 0)*100:.1f}%" if inf.get("profitMargins") else "N/A",
                            })
                    except Exception as exc:
                        st.warning(f"Could not load data for {resolved_sym}: {exc}")

            if hist_dict:
                st.subheader("📈 Relative Return Comparison")
                fig_comp = create_comparison_chart(hist_dict)
                st.plotly_chart(fig_comp, use_container_width=True)

            if comp_data:
                st.subheader("📋 Metrics Comparison Table")
                st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)


# ===========================================================================
# TAB 3: WATCHLIST & QUICK SCANS
# ===========================================================================
with tab_watchlist:
    st.subheader("⭐ Personal Watchlist & Quick Scans")

    w_col1, w_col2 = st.columns([2, 1])
    with w_col1:
        new_item = st.text_input("Add Ticker to Watchlist", placeholder="e.g. TSLA, INFY.NS, AMZN").strip()
    with w_col2:
        st.write("")
        st.write("")
        if st.button("➕ Add Ticker", use_container_width=True) and new_item:
            res_item = cached_resolve_ticker(new_item)
            ticker_to_add = res_item.get("ticker", new_item).upper()
            if ticker_to_add not in st.session_state["watchlist"]:
                st.session_state["watchlist"].append(ticker_to_add)
                st.toast(f"Added {ticker_to_add} to watchlist!")

    if st.session_state["watchlist"]:
        st.write(f"**Tracking {len(st.session_state['watchlist'])} Stocks:**")
        
        watch_rows = []
        for w_ticker in st.session_state["watchlist"]:
            try:
                stk = yf.Ticker(w_ticker)
                h_w = stk.history(period="5d")
                if not h_w.empty and "Close" in h_w:
                    last_p = float(h_w["Close"].iloc[-1])
                    prev_p = float(h_w["Close"].iloc[-2]) if len(h_w) > 1 else last_p
                    chg_pct = ((last_p / prev_p) - 1) * 100
                    inf_w = stk.info or {}
                    sym_c = get_currency_symbol(inf_w.get("currency", "USD"))
                    watch_rows.append({
                        "Ticker": w_ticker,
                        "Company": inf_w.get("shortName", w_ticker),
                        "Price": f"{sym_c}{last_p:,.2f}",
                        "1-Day Change": f"{chg_pct:+.2f}%",
                        "52W High": f"{sym_c}{inf_w.get('fiftyTwoWeekHigh', 'N/A')}",
                        "52W Low": f"{sym_c}{inf_w.get('fiftyTwoWeekLow', 'N/A')}",
                    })
            except Exception:
                watch_rows.append({"Ticker": w_ticker, "Company": "Error loading", "Price": "N/A"})

        if watch_rows:
            st.dataframe(pd.DataFrame(watch_rows), hide_index=True, use_container_width=True)

        if st.button("🗑️ Clear Watchlist"):
            st.session_state["watchlist"] = []
            st.rerun()

    st.divider()
    st.subheader("🌐 Popular Market Scan Baskets")
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        st.markdown("**🇺🇸 US Tech Leaders**")
        st.caption("AAPL · MSFT · NVDA · GOOGL · AMZN · META · TSLA")
    with b_col2:
        st.markdown("**🇮🇳 India Nifty 50 Leaders**")
        st.caption("RELIANCE.NS · TCS.NS · INFY.NS · HDFCBANK.NS · TATAMOTORS.NS")
    with b_col3:
        st.markdown("**⚡ AI & Semiconductor Leaders**")
        st.caption("NVDA · TSM · ASML · AMD · AVGO · QCOM")
