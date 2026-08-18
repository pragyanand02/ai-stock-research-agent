"""
Interactive Financial Charts Module using Plotly.
Supports Candlestick, Overlays (SMA, EMA, Bollinger Bands), Volume, RSI, and MACD subplots.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_stock_chart(
    df: pd.DataFrame,
    ticker: str = "",
    company_name: str = "",
    chart_type: str = "Candlestick",
    show_sma: bool = True,
    show_ema: bool = True,
    show_bb: bool = True,
    show_volume: bool = True,
    show_rsi: bool = True,
    show_macd: bool = True,
) -> go.Figure:
    """
    Generate a professional multi-panel interactive technical chart.
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No chart data available", showarrow=False, font=dict(size=18))
        return fig

    # Ensure necessary columns exist
    df = df.copy()
    close = df["Close"].values
    
    # Calculate indicators if not present
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    
    # Bollinger Bands
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["SMA20"] + (2 * rolling_std)
    df["BB_Lower"] = df["SMA20"] - (2 * rolling_std)
    
    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)
    
    # MACD (12, 26, 9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Determine subplots layout
    rows = 1
    row_heights = [0.6]
    subplot_titles = [f"{company_name or ticker} Price Chart"]
    
    has_vol = show_volume and "Volume" in df and df["Volume"].sum() > 0
    has_rsi = show_rsi
    has_macd = show_macd
    
    if has_vol:
        rows += 1
        row_heights.append(0.15)
        subplot_titles.append("Volume")
    if has_rsi:
        rows += 1
        row_heights.append(0.15)
        subplot_titles.append("RSI (14)")
    if has_macd:
        rows += 1
        row_heights.append(0.15)
        subplot_titles.append("MACD (12, 26, 9)")

    # Normalize row heights
    total_h = sum(row_heights)
    row_heights = [h / total_h for h in row_heights]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    current_row = 1

    # Panel 1: Price (Candlestick or Line)
    if chart_type == "Candlestick" and all(col in df for col in ["Open", "High", "Low", "Close"]):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="OHLC",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            ),
            row=current_row,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="#2962FF", width=2),
            ),
            row=current_row,
            col=1,
        )

    # Overlays: Bollinger Bands
    if show_bb and "BB_Upper" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Upper"],
                name="BB Upper (20,2)",
                line=dict(color="rgba(33, 150, 243, 0.3)", width=1, dash="dash"),
            ),
            row=current_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["BB_Lower"],
                name="BB Lower (20,2)",
                line=dict(color="rgba(33, 150, 243, 0.3)", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(33, 150, 243, 0.05)",
            ),
            row=current_row,
            col=1,
        )

    # Overlays: Moving Averages
    if show_ema and "EMA20" in df:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["EMA20"],
                name="EMA 20",
                line=dict(color="#FF9800", width=1.5),
            ),
            row=current_row,
            col=1,
        )

    if show_sma:
        if "SMA50" in df and df["SMA50"].dropna().shape[0] > 0:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA50"],
                    name="SMA 50",
                    line=dict(color="#00E676", width=1.5),
                ),
                row=current_row,
                col=1,
            )
        if "SMA200" in df and df["SMA200"].dropna().shape[0] > 0:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA200"],
                    name="SMA 200",
                    line=dict(color="#E040FB", width=1.5),
                ),
                row=current_row,
                col=1,
            )

    # Panel 2: Volume
    if has_vol:
        current_row += 1
        vol_colors = [
            "#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df["Close"], df.get("Open", df["Close"]))
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.7,
            ),
            row=current_row,
            col=1,
        )
        vol_sma20 = df["Volume"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vol_sma20,
                name="Vol SMA (20)",
                line=dict(color="#FFB300", width=1.2),
            ),
            row=current_row,
            col=1,
        )

    # Panel 3: RSI
    if has_rsi:
        current_row += 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                name="RSI (14)",
                line=dict(color="#7C4DFF", width=1.5),
            ),
            row=current_row,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", line_width=1, row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", line_width=1, row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="#888888", line_width=0.8, row=current_row, col=1)

    # Panel 4: MACD
    if has_macd:
        current_row += 1
        hist_colors = ["#26a69a" if val >= 0 else "#ef5350" for val in df["MACD_Hist"]]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["MACD_Hist"],
                name="MACD Hist",
                marker_color=hist_colors,
            ),
            row=current_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                name="MACD",
                line=dict(color="#2962FF", width=1.3),
            ),
            row=current_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                name="Signal",
                line=dict(color="#FF6D00", width=1.3),
            ),
            row=current_row,
            col=1,
        )

    # Layout configuration
    total_height = 450 + (120 * (rows - 1))
    fig.update_layout(
        height=total_height,
        margin=dict(l=40, r=40, t=40, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        hovermode="x unified",
        template="plotly_dark",
    )

    return fig


def create_comparison_chart(tickers_history: dict[str, pd.DataFrame]) -> go.Figure:
    """
    Create a normalized percentage return comparison chart for multiple stocks.
    """
    fig = go.Figure()
    if not tickers_history:
        fig.add_annotation(text="No comparison data", showarrow=False)
        return fig

    palette = ["#2962FF", "#00E676", "#FF9100", "#E040FB", "#00E5FF", "#FF5252"]
    color_idx = 0

    for ticker, df in tickers_history.items():
        if df is None or df.empty or "Close" not in df:
            continue
        close_series = df["Close"].dropna()
        if close_series.empty:
            continue
        first_val = float(close_series.iloc[0])
        if first_val == 0:
            continue
        pct_return = ((close_series / first_val) - 1.0) * 100.0

        color = palette[color_idx % len(palette)]
        color_idx += 1

        fig.add_trace(
            go.Scatter(
                x=pct_return.index,
                y=pct_return.values,
                mode="lines",
                name=f"{ticker} ({pct_return.iloc[-1]:+.1f}%)",
                line=dict(color=color, width=2),
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="#888888", line_width=1)
    fig.update_layout(
        title="Relative Performance Comparison (% Return)",
        height=450,
        margin=dict(l=40, r=40, t=50, b=30),
        yaxis_title="Return (%)",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
