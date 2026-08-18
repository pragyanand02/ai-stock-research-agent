import numpy as np
import pandas as pd
import pytest

from charts import create_stock_chart, create_comparison_chart
from pdf_generator import generate_stock_pdf
from agents import (
    _compute_bollinger_bands,
    _compute_atr,
    _compute_stochastic,
    _compute_pivot_points,
    _extract_recommendation_summary,
)


def test_compute_bollinger_bands():
    prices = np.linspace(100, 110, 30)
    upper, middle, lower = _compute_bollinger_bands(prices, period=20, num_std=2.0)
    assert len(upper) == 30
    assert len(middle) == 30
    assert len(lower) == 30
    assert not np.isnan(upper[-1])
    assert upper[-1] > middle[-1] > lower[-1]


def test_compute_atr():
    n = 30
    high = np.linspace(105, 115, n)
    low = np.linspace(95, 105, n)
    close = np.linspace(100, 110, n)
    atr = _compute_atr(high, low, close, period=14)
    assert len(atr) == n
    assert not np.isnan(atr[-1])
    assert atr[-1] > 0


def test_compute_stochastic():
    n = 30
    high = np.linspace(105, 115, n)
    low = np.linspace(95, 105, n)
    close = np.linspace(100, 110, n)
    k, d = _compute_stochastic(high, low, close, k_period=14, d_period=3)
    assert len(k) == n
    assert len(d) == n
    assert not np.isnan(k[-1])
    assert not np.isnan(d[-1])
    assert 0 <= k[-1] <= 100


def test_compute_pivot_points():
    pivots = _compute_pivot_points(high=110.0, low=90.0, close=100.0)
    assert "pivot" in pivots
    assert pivots["pivot"] == 100.0
    assert pivots["r1"] == 110.0
    assert pivots["s1"] == 90.0
    assert pivots["r2"] == 120.0
    assert pivots["s2"] == 80.0


def test_extract_recommendation_summary():
    brief_buy = "**Signal:** Buy\n**Conviction:** High\n**Time Horizon:** Long-term (> 1 year)\n**Thesis:** Strong growth."
    summary_buy = _extract_recommendation_summary(brief_buy)
    assert summary_buy["signal"] == "BUY"
    assert summary_buy["conviction"] == "HIGH"
    assert "Long-term" in summary_buy["time_horizon"]
    assert summary_buy["score"] >= 8.0

    brief_avoid = "**Signal:** Avoid\n**Conviction:** Medium\n**Time Horizon:** Short-term (< 3 months)"
    summary_avoid = _extract_recommendation_summary(brief_avoid)
    assert summary_avoid["signal"] == "AVOID"
    assert summary_avoid["score"] <= 3.0


def test_create_stock_chart():
    dates = pd.date_range(start="2025-01-01", periods=60)
    df = pd.DataFrame(
        {
            "Open": np.linspace(100, 150, 60),
            "High": np.linspace(105, 155, 60),
            "Low": np.linspace(95, 145, 60),
            "Close": np.linspace(102, 152, 60),
            "Volume": np.random.randint(1000, 5000, 60),
        },
        index=dates,
    )
    fig = create_stock_chart(df, ticker="TEST", company_name="Test Corp")
    assert fig is not None
    assert len(fig.data) > 0


def test_create_comparison_chart():
    dates = pd.date_range(start="2025-01-01", periods=30)
    df1 = pd.DataFrame({"Close": np.linspace(100, 120, 30)}, index=dates)
    df2 = pd.DataFrame({"Close": np.linspace(200, 210, 30)}, index=dates)
    fig = create_comparison_chart({"AAPL": df1, "MSFT": df2})
    assert fig is not None
    assert len(fig.data) == 2


def test_generate_stock_pdf():
    pdf_bytes = generate_stock_pdf(
        ticker="AAPL",
        company_name="Apple Inc.",
        current_price=175.50,
        currency_symbol="$",
        brief_text="**Signal:** Buy\n**Thesis:** Strong revenue pipeline.",
        fundamentals_text="Solid P/E ratio and low debt.",
        sentiment_text="Bullish news sentiment.",
        technical_text="Above SMA 50 and SMA 200.",
        metrics={"marketCap": 2800000000000, "trailingPE": 28.5},
        recommendation={"signal": "BUY", "conviction": "HIGH", "time_horizon": "Long-term"},
    )
    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert len(pdf_bytes) > 500
    assert bytes(pdf_bytes).startswith(b"%PDF")
