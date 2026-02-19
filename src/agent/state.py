"""
src/agent/state.py
AlphaAgentState — shared state bag that flows through every node in the
LangGraph StateGraph.  Each node reads from this dict and returns a
partial update; LangGraph merges the diffs automatically.

Fields annotated with `operator.add` accumulate across nodes (messages list).
All other fields are replaced by the last writer.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AlphaAgentState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    ticker: str                   # stock symbol being analysed  e.g. "NVDA"
    query:  str                   # free-text query for the agent

    # ── Accumulated message log (grows with operator.add each step) ──────────
    messages: Annotated[list, operator.add]

    # ── Retrieval outputs ─────────────────────────────────────────────────────
    retrieved_news:  list         # list of {text, date, source, sentiment}
    sentiment_score: float        # mean sentiment from retrieved chunks

    # ── Technical analysis outputs ────────────────────────────────────────────
    technical_indicators: dict    # RSI, MA_20, MA_50, MACD, vol_change_pct,
                                  # current_price, technical_verdict, bullish_signals
                                  # atr (Average True Range)

    # ── Gemini analyst outputs ────────────────────────────────────────────────
    news_summary:      str        # 2-3 sentence catalyst summary
    technical_verdict: str        # STRONGLY_BULLISH … STRONGLY_BEARISH
    alpha_signal:      str        # STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL
    confidence_score:  float      # 0.0 – 1.0

    # ── Long / Short engine outputs (new in Prompt 4) ─────────────────────────
    position_recommendation: str  # LONG / SHORT / NO_TRADE
    position_size_pct:  float     # % of portfolio (0 – 10 %)
    entry_price:        float     # suggested entry (≈ current market price)
    stop_loss_price:    float     # 1.5 × ATR away from entry
    take_profit_price:  float     # 2.5 × ATR away from entry
    risk_reward_ratio:  float     # take_profit_dist / stop_loss_dist
    trade_rationale:    str       # full human-readable explanation

    # ── Control ───────────────────────────────────────────────────────────────
    iterations:  int              # safety iteration counter (max 5)
    is_complete: bool
