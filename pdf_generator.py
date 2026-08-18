"""
PDF Report Generator for AI Stock Research & Investment Brief.
Generates an executive-ready, multi-section PDF document using fpdf2.
"""
from datetime import datetime
from typing import Dict, Any, Optional
import io
import re
from fpdf import FPDF


class StockReportPDF(FPDF):
    def __init__(self, ticker: str, company_name: str):
        super().__init__()
        self.ticker = _clean_text(ticker)
        self.company_name = _clean_text(company_name)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"AI Stock Research Report - {self.ticker} ({self.company_name})", border=0, align="L")
        self.cell(0, 8, datetime.now().strftime("%B %d, %Y"), border=0, align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 200)
        self.line(10, 18, 200, 18)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Educational Research Only | Not Financial Advice", align="C")


def _clean_text(text: str) -> str:
    """Clean markdown artifacts and unsupported characters for basic PDF encoding."""
    if not text:
        return ""
    # Strip emojis or replace unicode characters that fpdf default font doesn't support
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = text.replace('\u2014', '-').replace('\u2013', '-').replace('\u2022', '-')
    text = text.replace('•', '-').replace('–', '-').replace('—', '-')
    text = text.replace('“', '"').replace('”', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('₹', 'INR ').replace('€', 'EUR ').replace('£', 'GBP ').replace('¥', 'JPY ')
    # Encode to latin-1 with replacement for any remaining odd characters
    text = text.encode("latin-1", "replace").decode("latin-1")
    return text.strip()


def generate_stock_pdf(
    ticker: str,
    company_name: str,
    current_price: float,
    currency_symbol: str,
    brief_text: str,
    fundamentals_text: str,
    sentiment_text: str,
    technical_text: str,
    metrics: Optional[Dict[str, Any]] = None,
    technical_indicators: Optional[Dict[str, Any]] = None,
    recommendation: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Generate an executive PDF report and return the binary content.
    """
    pdf = StockReportPDF(ticker=ticker, company_name=company_name)
    pdf.add_page()

    # --- Title Banner ---
    pdf.set_fill_color(33, 150, 243)  # Accent blue
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, f"  {company_name} ({ticker})", fill=True, new_x="LMARGIN", new_y="NEXT")

    # --- Price and Subtitle ---
    pdf.ln(2)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("Helvetica", "B", 12)
    price_str = f"Current Price: {currency_symbol}{current_price:.2f}" if current_price else "Current Price: N/A"
    
    signal = (recommendation or {}).get("signal", "N/A")
    conviction = (recommendation or {}).get("conviction", "N/A")
    horizon = (recommendation or {}).get("time_horizon", "N/A")
    
    pdf.cell(95, 8, price_str, border=0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(95, 8, f"Signal: {signal} | Conviction: {conviction}", border=0, align="R", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # --- Section: Investment Brief ---
    pdf.set_fill_color(240, 244, 248)
    pdf.set_text_color(20, 50, 90)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "  1. Executive Investment Brief", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_text_color(50, 50, 50)
    pdf.set_font("Helvetica", "", 9)
    clean_brief = _clean_text(brief_text)
    # Remove excessive bold markdown from brief
    clean_brief = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_brief)
    pdf.multi_cell(0, 4.5, clean_brief)
    pdf.ln(4)

    # --- Section: Key Financial Metrics Table ---
    if metrics:
        pdf.set_fill_color(240, 244, 248)
        pdf.set_text_color(20, 50, 90)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "  2. Key Financial Fundamentals", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(225, 235, 245)
        pdf.set_text_color(30, 30, 30)
        
        # Headers
        pdf.cell(47, 6, "Metric", 1, align="L", fill=True)
        pdf.cell(47, 6, "Value", 1, align="C", fill=True)
        pdf.cell(47, 6, "Metric", 1, align="L", fill=True)
        pdf.cell(47, 6, "Value", 1, align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

        pdf.set_font("Helvetica", "", 8)
        
        def _fmt(val, is_pct=False, is_cur=False):
            if val is None or val == "N/A":
                return "N/A"
            if isinstance(val, (int, float)):
                if is_pct:
                    return f"{val * 100:.1f}%" if abs(val) < 1 else f"{val:.1f}%"
                if is_cur:
                    if abs(val) >= 1e9:
                        return f"{currency_symbol}{val/1e9:.2f}B"
                    elif abs(val) >= 1e6:
                        return f"{currency_symbol}{val/1e6:.2f}M"
                    return f"{currency_symbol}{val:.2f}"
                return f"{val:.2f}"
            return str(val)

        rows = [
            ("Market Cap", _fmt(metrics.get("marketCap"), is_cur=True), "Trailing P/E", _fmt(metrics.get("trailingPE"))),
            ("Forward P/E", _fmt(metrics.get("forwardPE")), "Trailing EPS", _fmt(metrics.get("trailingEps"), is_cur=True)),
            ("Revenue Growth", _fmt(metrics.get("revenueGrowth"), is_pct=True), "Earnings Growth", _fmt(metrics.get("earningsGrowth"), is_pct=True)),
            ("Profit Margins", _fmt(metrics.get("profitMargins"), is_pct=True), "Return on Equity", _fmt(metrics.get("returnOnEquity"), is_pct=True)),
            ("Debt-to-Equity", _fmt(metrics.get("debtToEquity")), "Target Consensus", _fmt(metrics.get("targetMeanPrice"), is_cur=True)),
        ]

        for m1, v1, m2, v2 in rows:
            pdf.cell(47, 5.5, f" {m1}", 1)
            pdf.cell(47, 5.5, v1, 1, align="C")
            pdf.cell(47, 5.5, f" {m2}", 1)
            pdf.cell(47, 5.5, v2, 1, align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # --- Section: Specialist Breakdown ---
    pdf.set_fill_color(240, 244, 248)
    pdf.set_text_color(20, 50, 90)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "  3. Specialist Research Analysis", fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Fundamentals Analysis Text
    pdf.set_text_color(30, 30, 80)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 5, "Fundamental Analyst Report:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 4, _clean_text(fundamentals_text))
    pdf.ln(2)

    # Technical Analysis Text
    pdf.set_text_color(30, 30, 80)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 5, "Technical Analysis & Setup:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 4, _clean_text(technical_text))
    pdf.ln(2)

    # News Sentiment Text
    pdf.set_text_color(30, 30, 80)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 5, "News Sentiment & Market Narrative:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 4, _clean_text(sentiment_text))
    pdf.ln(4)

    # --- Legal Disclaimer ---
    pdf.set_fill_color(255, 248, 225)
    pdf.set_draw_color(255, 193, 7)
    pdf.set_text_color(120, 80, 0)
    pdf.set_font("Helvetica", "I", 7.5)
    disclaimer = (
        "Disclaimer: This report is automatically generated by AI agents for educational and informational purposes only. "
        "It does not constitute financial, investment, or trading advice. Past performance is not indicative of future results. "
        "Consult a certified financial planner or registered investment advisor before making any investment decision."
    )
    pdf.multi_cell(0, 4, disclaimer, border=1, fill=True)

    return pdf.output()
