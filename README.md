# AI Stock Research & Investment Brief Generator
An AI-powered stock research application built using LangGraph, Google Gemini, Yahoo Finance, and Streamlit.

## Features
- AI-generated investment brief
- Fundamental stock analysis
- Technical analysis with indicators
- AI-powered news sentiment analysis
- Investment recommendation using Google Gemini
- Interactive Streamlit web interface

## Tech Stack
- Python
- LangGraph
- LangChain
- Google Gemini API
- Yahoo Finance (yfinance)
- Streamlit

## Project Architecture
User Input → LangGraph Workflow → Fundamentals → News Sentiment → Technical Analysis → Gemini AI → Final Investment Brief

## Screenshots
### App Interface
![App Interface](Screenshot%202026-07-31%20151320.png)

### Stock Price Trend
![Stock Price Trend](Screenshot%202026-07-31%20151350.png)

### Investment Brief
![Investment Brief](Screenshot%202026-07-31%20151409.png)

### Download Report
![Download Report](Screenshot%202026-07-31%20151436.png)

### Fundamentals, News Sentiment & Technical Analysis
![Fundamentals](Screenshot%202026-07-31%20151449.png)

## Future Improvements
- Export investment brief as PDF
- Interactive stock price charts
- Enhanced financial news integration
- Portfolio watchlist
- Multi-stock comparison

## Installation

Clone the repository:

```bash
git clone https://github.com/pragyanand02/ai-stock-research-agent.git
```

Go to the project folder:

```bash
cd ai-stock-research-agent
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key
NEWS_API_KEY=your_newsapi_key_optional
```

> `NEWS_API_KEY` is optional. If not provided, the application automatically falls back to Yahoo Finance news.

Run the application:

```bash
streamlit run app.py
```

---

## Live Demo

[AI Stock Research Live Demo](https://ai-stock-research-agent-nucf6skvecaofhspzelnc2.streamlit.app)

---

## GitHub Repository

[GitHub Repository](https://github.com/pragyanand02/ai-stock-research-agent)