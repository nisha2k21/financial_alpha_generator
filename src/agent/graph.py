"""
src/agent/graph.py
LangGraph StateGraph wiring the 5 alpha-agent nodes into a compiled pipeline.

Flow:
  retrieval ─► technical ─► gemini ─►[conf≥0.40?]─► long_short ─► reporter ─► END
                                       └──[low conf]──────────────────► reporter ─► END

The conditional edge after gemini_analyst_node routes:
  - confidence_score < 0.40  →  reporter         (no trade — insufficient signal)
  - confidence_score ≥ 0.40  →  long_short_engine (compute position sizing)
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from .nodes import (
    gemini_analyst_node,
    long_short_engine_node,
    reporter_node,
    retrieval_node,
    technical_analysis_node,
)
from .state import AlphaAgentState

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.40   # below this we skip position sizing


def _confidence_router(state: AlphaAgentState) -> str:
    """Route after gemini node based on confidence score."""
    conf = state.get("confidence_score", 0.0)
    if conf < LOW_CONFIDENCE_THRESHOLD:
        logger.info(
            "[confidence_router] Low confidence (%.2f) — skipping long/short engine",
            conf,
        )
        return "reporter"
    return "long_short"


def build_alpha_agent():
    """Compile and return the LangGraph StateGraph agent."""
    graph = StateGraph(AlphaAgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("retrieval",  retrieval_node)
    graph.add_node("technical",  technical_analysis_node)
    graph.add_node("gemini",     gemini_analyst_node)
    graph.add_node("long_short", long_short_engine_node)
    graph.add_node("reporter",   reporter_node)

    # ── Set entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("retrieval")

    # ── Linear edges ──────────────────────────────────────────────────────────
    graph.add_edge("retrieval", "technical")
    graph.add_edge("technical", "gemini")
    # gemini → long_short OR reporter (conditional — see below)
    graph.add_edge("long_short", "reporter")
    graph.add_edge("reporter",   END)

    # ── Conditional edge: gemini → long_short | reporter ─────────────────────
    graph.add_conditional_edges(
        "gemini",
        _confidence_router,
        {
            "long_short": "long_short",
            "reporter":   "reporter",
        },
    )

    return graph.compile()


# Pre-build the agent once at module load (reused across calls)
_AGENT = None


def _get_agent():
    global _AGENT
    if _AGENT is None:
        _AGENT = build_alpha_agent()
    return _AGENT


def run_alpha_agent(ticker: str) -> dict[str, Any]:
    """
    Execute the full LangGraph alpha pipeline for one ticker.

    Parameters
    ----------
    ticker : str  Stock symbol, e.g. "AAPL"

    Returns
    -------
    AlphaAgentState dict with all fields populated by each node.
    """
    agent = _get_agent()

    initial_state: AlphaAgentState = {
        "ticker":   ticker.upper(),
        "query":    f"Analyze {ticker.upper()} and generate alpha signal with trade recommendation",
        "messages": [],

        # Retrieval
        "retrieved_news":  [],
        "sentiment_score": 0.0,

        # Technicals
        "technical_indicators": {},

        # Gemini outputs
        "news_summary":      "",
        "technical_verdict": "",
        "alpha_signal":      "NEUTRAL",
        "confidence_score":  0.0,

        # Long/Short engine
        "position_recommendation": "NO_TRADE",
        "position_size_pct":       0.0,
        "entry_price":             0.0,
        "stop_loss_price":         0.0,
        "take_profit_price":       0.0,
        "risk_reward_ratio":       0.0,
        "trade_rationale":         "",

        # Control
        "iterations":   0,
        "is_complete":  False,
    }

    logger.info("[run_alpha_agent] Starting pipeline for %s", ticker)
    final_state = agent.invoke(initial_state)
    logger.info(
        "[run_alpha_agent] Done. signal=%s position=%s conf=%.2f",
        final_state.get("alpha_signal"),
        final_state.get("position_recommendation"),
        final_state.get("confidence_score", 0.0),
    )
    return final_state
