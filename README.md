# AI Stock Research & Investment Brief Generator (v2.0)
An AI-powered institutional-grade stock research and investment brief generator built using **LangGraph**, **Google Gemini**, **Yahoo Finance**, **Plotly**, and **Streamlit**.

## ✨ Key Features
- 📋 **Executive AI Investment Brief**: Clear, decisive signals (Buy / Hold / Watch / Avoid), conviction ratings, time horizon, bull/bear thesis, and score.
- 📊 **Interactive Plotly Technical Charts**: Candlestick charts, Moving Average overlays (SMA 20/50/200, EMA 20), Bollinger Bands (20, 2), Volume panels, and subplots for RSI (14) and MACD (12, 26, 9).
- 🏦 **Fundamental Analysis**: In-depth evaluation of P/E, PEG, Price-to-Book, EV/EBITDA, ROE, Free Cash Flow, margins, and balance sheet debt.
- 📰 **News & Sentiment Intelligence**: 7-day market headline scanning with sentiment classification and direct source article links.
- 📥 **One-Click PDF Report Export**: Generates professional, multi-page executive PDF reports with metric tables, analyst notes, and charts.
- ⚖️ **Multi-Stock Comparison**: Side-by-side metric comparison and normalized % return comparison charts for 2–5 stocks.
- ⭐ **Personal Watchlist & Market Presets**: Track favorite stocks and quickly scan popular US and Indian market baskets.
- ⚡ **FastAPI REST API**: High-performance backend endpoints (`/brief`, `/compare`, `/pdf`, `/health`) with CORS support and caching.

---

## 🛠️ Tech Stack
- **AI & Orchestration**: LangGraph, LangChain, Google Gemini (`gemini-2.5-flash`)
- **Market Data**: Yahoo Finance (`yfinance`), NewsAPI
- **Visuals & Reporting**: Plotly, FPDF2, Altair
- **Web UI & API**: Streamlit, FastAPI, Uvicorn
- **Testing**: Pytest

---

## 🏗️ Architecture
```text
User Input / Ticker
        │
        ▼ (Fan-Out)
 ┌──────────────────────┬──────────────────────┬──────────────────────┐
 │  Agent 1: Valuation  │  Agent 2: Sentiment  │  Agent 3: Technical  │
 │    & Fundamentals    │     & Market News    │    & Price Signals   │
 └──────────┬───────────┴──────────┬───────────┴──────────┬───────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   ▼ (Fan-In)
                         Agent 4: Brief Writer
                                   │
                                   ▼
                    Final Executive Investment Brief
                   (Plotly Charts + PDF + Streamlit)
```

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pragyanand02/ai-stock-research-agent.git
   cd ai-stock-research-agent
   ```

2. **Create & Activate Virtual Environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables (`.env`):**
   ```env
   GOOGLE_API_KEY=your_gemini_api_key
   NEWS_API_KEY=your_newsapi_key_optional
   GEMINI_MODEL=gemini-2.5-flash
   CACHE_TTL_SECONDS=3600
   ```

---

## 💻 Running the Application

### 1. Launch Streamlit Web UI
```bash
streamlit run app.py
```

### 2. Launch FastAPI REST Server
```bash
uvicorn api:app --host 0.0.0.0 --port 8002 --reload
```

### 3. Run Automated Pytest Suite
```bash
pytest
```

---

## 📡 REST API Endpoints
- `GET /brief?ticker=AAPL` — Full multi-agent research analysis with structured data.
- `GET /compare?tickers=AAPL,MSFT,NVDA` — Side-by-side comparison across multiple stocks.
- `GET /pdf?ticker=AAPL` — Download formatted PDF investment report.
- `GET /health` — Service health check.

---

## ⚖️ Disclaimer
*This project is for educational and informational purposes only. It is not financial advice. Always perform your own due diligence before making investment decisions.*
