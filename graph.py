"""
Project 04 — LangGraph fan-out / fan-in pipeline.

[Ticker] → [Agent 1: Fundamentals]  ↘
         → [Agent 2: Sentiment]      → [Agent 4: Brief Writer] → Investment Brief
         → [Agent 3: Technical]     ↗
"""
from langgraph.graph import StateGraph, START, END

from state import StockResearchState
from agents import (
    agent_fundamentals_analyst,
    agent_sentiment_scanner,
    agent_technical_analyst,
    agent_brief_writer,
)


def build_graph() -> StateGraph:
    builder = StateGraph(StockResearchState)

    builder.add_node("fundamentals_analyst", agent_fundamentals_analyst)
    builder.add_node("sentiment_scanner", agent_sentiment_scanner)
    builder.add_node("technical_analyst", agent_technical_analyst)
    builder.add_node("brief_writer", agent_brief_writer)

    # Fan-out from START to all three independent agents (runs in parallel)
    builder.add_edge(START, "fundamentals_analyst")
    builder.add_edge(START, "sentiment_scanner")
    builder.add_edge(START, "technical_analyst")

    # Fan-in: all three must complete before brief_writer runs
    builder.add_edge("fundamentals_analyst", "brief_writer")
    builder.add_edge("sentiment_scanner", "brief_writer")
    builder.add_edge("technical_analyst", "brief_writer")

    builder.add_edge("brief_writer", END)

    return builder.compile()


# Compiled graph — import this in api.py and app.py
graph = build_graph()