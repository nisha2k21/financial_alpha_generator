"""
src/alpha_engine.py
Orchestrates the full alpha signal generation pipeline.

Pipeline
--------
1. Fetch news (NewsAPI / sample fallback)
2. Fetch 6-month OHLCV + compute RSI / MA20 / vol_chg_pct
3. Save news and prices to SQLite
4. Embed articles into ChromaDB (with TextBlob sentiment per chunk)
5. Build LangChain LCEL chain (Gemini 1.5 Pro)
6. Query for structured alpha signal
7. Parse LLM response → AlphaSignal dataclass
8. Compute confidence using technical confirmation
9. Persist to `signals` table in SQLite
10. Return AlphaSignal

Exports
-------
AlphaSignal           — Dataclass with all signal fields
run_alpha_pipeline()  — End-to-end pipeline for one ticker
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .database import init_db, save_signal
from .embeddings import embed_and_store, get_collection_count, get_or_create_vectorstore
from .ingestion import (
    fetch_news,
    fetch_stock_data,
    compute_technical_indicators,
    save_news_to_db,
    save_prices_to_db,
    summarise_technicals,
    format_articles_for_rag,
)
from .rag_pipeline import (
    build_quant_chain,
    query_alpha_signal,
    generate_mock_signal,
)

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-1.5-pro"

# ─── Signal Direction Mapping ─────────────────────────────────────────────────

# Maps the five human-readable rating labels to integer strength values
DIRECTION_TO_STRENGTH: dict[str, int] = {
    "strong buy":  5,
    "buy":         4,
    "neutral":     3,
    "sell":        2,
    "strong sell": 1,
}

STRENGTH_TO_DIRECTION: dict[int, str] = {v: k.title() for k, v in DIRECTION_TO_STRENGTH.items()}


# ─── AlphaSignal Dataclass ────────────────────────────────────────────────────

@dataclass
class AlphaSignal:
    """
    Structured alpha signal produced by the end-to-end LangGraph agent pipeline.
    """
    ticker:           str
    signal_strength:  int            # 1 (Strong Sell) → 5 (Strong Buy)
    direction:        str            # "Strong Buy" | "Buy" | "Neutral" | "Sell" | "Strong Sell"
    confidence_score: float          # 0.0 – 1.0
    reasoning:        str
    news_citations:   list[str] = field(default_factory=list)
    generated_at:     str = ""
    ticker_rsi:       float = float("nan")
    ticker_ma20:      float = float("nan")
    vol_change_pct:   float = float("nan")
    model_version:    str = "langgraph-gemini"
    raw_response:     str = ""

    # ── Long/Short Engine fields (v2) ──────────────────────────────────────────
    position:              str   = "NO_TRADE"
    position_size_pct:     float = 0.0
    entry_price:           float = 0.0
    stop_loss_price:       float = 0.0
    take_profit_price:     float = 0.0
    risk_reward_ratio:     float = 0.0
    trade_rationale:       str   = ""



# ─── Direction Parser ─────────────────────────────────────────────────────────

def direction_to_strength(direction_str: str) -> int:
    """
    Convert a direction label to integer strength (1–5).

    Performs case-insensitive fuzzy matching so minor LLM formatting
    variation (e.g. "STRONG BUY", "strong buy") resolves correctly.

    Parameters
    ----------
    direction_str : Raw direction string from LLM output

    Returns
    -------
    Integer in [1, 5]. Defaults to 3 (Neutral) if no match.
    """
    normalized = direction_str.strip().lower()

    # Exact match first
    if normalized in DIRECTION_TO_STRENGTH:
        return DIRECTION_TO_STRENGTH[normalized]

    # Fuzzy: contains match
    for label, strength in DIRECTION_TO_STRENGTH.items():
        if label in normalized:
            return strength

    logger.warning("Unrecognised direction '%s' — defaulting to Neutral (3)", direction_str)
    return 3


# ─── LLM Response Parser ──────────────────────────────────────────────────────

def parse_signal_response(response_text: str) -> dict:
    """
    Parse the Gemini RAG chain's free-text response into structured fields.

    Extracts:
      - RATING      → direction string and signal_strength integer
      - CONFIDENCE  → float 0–100 (converted to 0.0–1.0)
      - REASONING   → full rationale text
      - CATALYSTS   → near-term catalyst text
      - RISKS       → key risks text
      - CITATIONS   → list of cited article headlines

    Robust to minor formatting variation using flexible regexes.

    Parameters
    ----------
    response_text : Raw LLM output string

    Returns
    -------
    Dict with keys:
        direction, signal_strength, raw_confidence, reasoning,
        catalysts, key_risks, news_citations
    """

    def _extract(pattern: str, default: str = "") -> str:
        m = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else default

    # Rating / direction
    raw_rating = _extract(r"RATING:\s*(.+?)(?:\n|$)")
    direction_clean = raw_rating.strip().rstrip(".").strip()
    signal_strength = direction_to_strength(direction_clean)

    # Normalise direction label using our canonical mapping
    direction = STRENGTH_TO_DIRECTION.get(signal_strength, "Neutral")

    # Confidence (0–100 from LLM → 0.0–1.0)
    raw_conf = _extract(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)")
    try:
        raw_confidence = max(0.0, min(100.0, float(raw_conf))) / 100.0
    except (ValueError, TypeError):
        raw_confidence = 0.5

    # Reasoning
    reasoning = _extract(r"REASONING:\s*(.+?)(?=CATALYSTS:|RISKS:|CITATIONS:|$)")
    reasoning = reasoning or "Analysis based on recent news and technical data."

    # Catalysts
    catalysts = _extract(r"CATALYSTS:\s*(.+?)(?=RISKS:|CITATIONS:|$)")

    # Key risks
    key_risks = _extract(r"RISKS:\s*(.+?)(?=CITATIONS:|$)")

    # Citations — comma-separated headline list
    raw_citations = _extract(r"CITATIONS:\s*(.+?)(?=$)")
    news_citations = [c.strip() for c in raw_citations.split(",") if c.strip()]

    # Combine reasoning + catalysts for the full rationale field
    full_reasoning = reasoning
    if catalysts:
        full_reasoning += f"\nKey catalysts: {catalysts}"
    if key_risks:
        full_reasoning += f"\nKey risks: {key_risks}"

    return {
        "direction":       direction,
        "signal_strength": signal_strength,
        "raw_confidence":  raw_confidence,
        "reasoning":       full_reasoning[:800],  # truncate for DB
        "catalysts":       catalysts,
        "key_risks":       key_risks,
        "news_citations":  news_citations,
    }


# ─── Technical Confidence Scoring ────────────────────────────────────────────

def combine_technical_score(
    rsi: Optional[float],
    vol_chg_pct: Optional[float],
    direction: str,
    raw_confidence: float,
) -> float:
    """
    Refine LLM confidence using technical indicator confirmation.

    Adjustment rules
    ----------------
    RSI confirmation
        RSI < 30 + BUY/STRONG BUY  → +0.15  (oversold supports bullish signal)
        RSI > 70 + SELL/STRONG SELL → +0.15  (overbought supports bearish signal)
        RSI < 30 + SELL             → -0.10  (conflicting: oversold but selling)
        RSI > 70 + BUY              → -0.10  (conflicting: overbought but buying)

    Volume confirmation
        Volume > +50% above average → +0.10  (high conviction move)
        Volume > +100% above average→ +0.15  (extremely high conviction)

    Final score is clamped to [0.0, 1.0].

    Parameters
    ----------
    rsi            : RSI(14) value (optional — NaN skips RSI adjustment)
    vol_chg_pct    : Volume change % (optional — None skips volume adjustment)
    direction      : Signal direction label (e.g. "Strong Buy")
    raw_confidence : LLM-reported confidence divided by 100

    Returns
    -------
    Adjusted confidence float in [0.0, 1.0]
    """
    score = raw_confidence
    d_lower = direction.lower()
    is_bullish = "buy" in d_lower      # "Buy" or "Strong Buy"
    is_bearish = "sell" in d_lower     # "Sell" or "Strong Sell"

    # ── RSI confirmation ──
    if rsi is not None and not (isinstance(rsi, float) and rsi != rsi):  # not NaN
        if rsi < 30:
            if is_bullish:
                score += 0.15   # oversold + bullish = strong confirmation
                logger.debug("RSI oversold (%.1f) + bullish: +0.15 confidence", rsi)
            elif is_bearish:
                score -= 0.10   # oversold + bearish = conflicting signal
                logger.debug("RSI oversold (%.1f) + bearish: -0.10 confidence", rsi)
        elif rsi > 70:
            if is_bearish:
                score += 0.15   # overbought + bearish = strong confirmation
                logger.debug("RSI overbought (%.1f) + bearish: +0.15 confidence", rsi)
            elif is_bullish:
                score -= 0.10   # overbought + bullish = conflicting signal
                logger.debug("RSI overbought (%.1f) + bullish: -0.10 confidence", rsi)

    # ── Volume confirmation ──
    if vol_chg_pct is not None and not (isinstance(vol_chg_pct, float) and vol_chg_pct != vol_chg_pct):
        if vol_chg_pct > 100:
            score += 0.15
            logger.debug("Volume surge (%.1f%%): +0.15 confidence", vol_chg_pct)
        elif vol_chg_pct > 50:
            score += 0.10
            logger.debug("High volume (%.1f%%): +0.10 confidence", vol_chg_pct)

    return round(max(0.0, min(1.0, score)), 3)


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def generate_alpha_for_ticker(ticker: str) -> dict:
    """Run the full LangGraph alpha agent for one ticker."""
    from .agent.graph import run_alpha_agent
    final_state = run_alpha_agent(ticker)
    
    # Extract technical values for flat dict return
    ind = final_state.get("technical_indicators", {})
    
    return {
        "ticker":            ticker,
        "alpha_signal":      final_state["alpha_signal"],
        "confidence_score":  final_state["confidence_score"],
        "position":          final_state["position_recommendation"],
        "position_size_pct": final_state["position_size_pct"],
        "entry_price":       final_state["entry_price"],
        "stop_loss":         final_state["stop_loss_price"],
        "take_profit":       final_state["take_profit_price"],
        "risk_reward":       final_state["risk_reward_ratio"],
        "news_summary":      final_state["news_summary"],
        "technical_verdict": ind.get("technical_verdict", "NEUTRAL"),
        "rsi":               ind.get("RSI"),
        "trade_rationale":   final_state["trade_rationale"],
    }


def generate_all_alphas(tickers=None) -> list[dict]:
    """Run agent for all tickers, return list of trade recommendations."""
    if tickers is None:
        tickers = ["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"]
    results = []
    for ticker in tickers:
        try:
            result = generate_alpha_for_ticker(ticker)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            print(f"Error processing {ticker}: {e}")
    return results


def run_alpha_pipeline(
    ticker: str,
    gemini_api_key: str,
    news_api_key: str,
    db_path: str,
    chroma_persist_dir: str,
    force_refresh: bool = False,
    model: str = GEMINI_MODEL,
) -> AlphaSignal:
    """
    Execute the full alpha signal generation pipeline for a single ticker.
    Now utilizes the LangGraph StateGraph agent for analysis & position sizing.
    """
    ticker = ticker.upper()
    generated_at = datetime.now(timezone.utc).isoformat()

    logger.info("═══ Starting LangGraph alpha pipeline: %s ═══", ticker)

    # ── 1. Ingestion (Ensures data exists for the agent) ──────────────────────
    # Fetch news
    articles = fetch_news(ticker, news_api_key, days_back=7)
    # Fetch prices
    stock_df = fetch_stock_data(ticker, period="6mo")
    if not stock_df.empty:
        stock_df = compute_technical_indicators(stock_df)

    # ── 2. Persist to SQLite (Agent nodes read from here) ─────────────────────
    init_db(db_path)
    save_news_to_db(articles, db_path)
    if not stock_df.empty:
        save_prices_to_db(stock_df, ticker, db_path)

    # ── 3. Embed to ChromaDB (Agent node retrieves from here) ─────────────────
    if gemini_api_key and not gemini_api_key.startswith("your_"):
        embed_and_store(
            articles=articles,
            ticker=ticker,
            persist_dir=chroma_persist_dir,
            api_key=gemini_api_key,
        )

    # ── 4. Call LangGraph Agent ───────────────────────────────────────────────
    from .agent.graph import run_alpha_agent
    final_state = run_alpha_agent(ticker)
    
    ind = final_state.get("technical_indicators", {})
    
    # ── 5. Map Agent State to AlphaSignal ─────────────────────────────────────
    return AlphaSignal(
        ticker=ticker,
        signal_strength=_signal_to_strength(final_state["alpha_signal"]),
        direction=final_state["alpha_signal"],
        confidence_score=final_state["confidence_score"],
        reasoning=final_state["news_summary"],
        news_citations=[c.get("source", "") for c in final_state.get("retrieved_news", [])[:5]],
        generated_at=generated_at,
        ticker_rsi=ind.get("RSI", float("nan")),
        ticker_ma20=ind.get("MA_20", float("nan")),
        vol_change_pct=ind.get("volume_change_pct", float("nan")),
        model_version="langgraph-gemini-v2",
        raw_response=final_state["trade_rationale"],
        
        # Extended fields
        position=final_state["position_recommendation"],
        position_size_pct=final_state["position_size_pct"],
        entry_price=final_state["entry_price"],
        stop_loss_price=final_state["stop_loss_price"],
        take_profit_price=final_state["take_profit_price"],
        risk_reward_ratio=final_state["risk_reward_ratio"],
        trade_rationale=final_state["trade_rationale"],
    )


def _signal_to_strength(signal: str) -> int:
    mapping = {
        "STRONG_BUY":  5,
        "BUY":         4,
        "NEUTRAL":     3,
        "SELL":        2,
        "STRONG_SELL": 1,
    }
    return mapping.get(signal.upper().replace(" ", "_"), 3)

