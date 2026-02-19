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
    Structured alpha signal produced by the end-to-end RAG pipeline.

    Fields
    ------
    ticker          : Stock ticker symbol (e.g. "AAPL")
    signal_strength : Integer 1–5
                      1 = Strong Sell, 2 = Sell, 3 = Neutral,
                      4 = Buy,         5 = Strong Buy
    direction       : Human-readable label ("Strong Buy", "Buy", etc.)
    confidence_score: Final confidence in [0.0, 1.0], blending LLM score
                      with technical indicator confirmation
    reasoning       : Full LLM rationale with article citations
    news_citations  : List of cited article headlines extracted from response
    generated_at    : ISO-8601 UTC timestamp of signal generation
    ticker_rsi      : Latest RSI(14) value for the ticker
    ticker_ma20     : Latest 20-day moving average closing price
    vol_change_pct  : Volume change % vs 20-day average
    model_version   : LLM model that generated the signal
    raw_response    : Unprocessed LLM output (for audit)
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
    model_version:    str = GEMINI_MODEL
    raw_response:     str = ""


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

    Steps
    -----
    1.  Fetch last 7 days of news (NewsAPI or sample fallback)
    2.  Fetch 6 months of OHLCV data (yfinance)
    3.  Compute RSI(14), MA20, vol_chg_pct
    4.  Persist articles → ``news`` table
    5.  Persist prices   → ``prices`` table
    6.  Embed articles into ChromaDB (chunk + TextBlob sentiment)
    7.  Build Gemini 1.5 Pro LCEL RAG chain
    8.  Query for structured alpha signal
    9.  Parse response → direction, strength, citations
    10. Adjust confidence with technical confirmation
    11. Persist AlphaSignal → ``signals`` table
    12. Return AlphaSignal object

    Parameters
    ----------
    ticker             : Stock ticker (e.g. "AAPL")
    gemini_api_key     : Google Gemini API key (empty = offline demo)
    news_api_key       : NewsAPI key (empty = sample fallback)
    db_path            : Path to SQLite database file
    chroma_persist_dir : ChromaDB persistence directory
    force_refresh      : Re-embed even if collection already exists
    model              : Gemini model name (default: gemini-1.5-pro)

    Returns
    -------
    AlphaSignal dataclass instance
    """
    ticker = ticker.upper()
    use_live_gemini = bool(gemini_api_key and not gemini_api_key.startswith("your_"))
    generated_at = datetime.now(timezone.utc).isoformat()

    logger.info("═══ Starting alpha pipeline: %s (live_gemini=%s) ═══", ticker, use_live_gemini)

    # ── 1. Fetch news ──────────────────────────────────────────────────────────
    articles = fetch_news(ticker, news_api_key, days_back=7)
    logger.info("News: %d articles for %s", len(articles), ticker)

    # ── 2. Fetch + compute prices ──────────────────────────────────────────────
    stock_df = fetch_stock_data(ticker, period="6mo")
    if not stock_df.empty:
        stock_df = compute_technical_indicators(stock_df)

    # Extract latest indicator values (for AlphaSignal fields + confidence calc)
    def _last(col: str) -> Optional[float]:
        if stock_df.empty or col not in stock_df.columns:
            return None
        val = stock_df[col].dropna()
        return float(val.iloc[-1]) if not val.empty else None

    rsi      = _last("RSI")
    ma20     = _last("MA20")
    vol_chg  = _last("vol_chg_pct")
    tech_summary = summarise_technicals(stock_df, ticker)

    # ── 3. Persist to SQLite ───────────────────────────────────────────────────
    init_db(db_path)
    save_news_to_db(articles, db_path)
    if not stock_df.empty:
        save_prices_to_db(stock_df, ticker, db_path)

    # ── 4-5. Embed + RAG ───────────────────────────────────────────────────────
    if use_live_gemini:
        existing_count = get_collection_count(ticker, chroma_persist_dir, gemini_api_key)

        if existing_count == 0 or force_refresh:
            vectorstore, n_chunks = embed_and_store(
                articles=articles,
                ticker=ticker,
                persist_dir=chroma_persist_dir,
                api_key=gemini_api_key,
            )
            logger.info("Embedded %d chunks for %s", n_chunks, ticker)
        else:
            vectorstore = get_or_create_vectorstore(
                collection_name=f"{ticker.lower()}_news",
                persist_dir=chroma_persist_dir,
                api_key=gemini_api_key,
            )
            logger.info("Reusing %d existing chunks for %s", existing_count, ticker)

        # ── 6. Build chain + query ─────────────────────────────────────────────
        chain_tuple = build_quant_chain(vectorstore, api_key=gemini_api_key, model=model)
        rag_result  = query_alpha_signal(chain_tuple, ticker, tech_summary)

    else:
        logger.info("Offline mode — using mock signal for %s", ticker)
        rag_result = generate_mock_signal(ticker, tech_summary)

    # ── 7. Parse LLM response ──────────────────────────────────────────────────
    raw_answer = rag_result.get("answer", "")
    parsed     = parse_signal_response(raw_answer)

    # ── 8. Confidence adjustment via technicals ────────────────────────────────
    final_confidence = combine_technical_score(
        rsi=rsi,
        vol_chg_pct=vol_chg,
        direction=parsed["direction"],
        raw_confidence=parsed["raw_confidence"],
    )

    # Supplement citations from retrieved source Documents
    source_docs = rag_result.get("sources", [])
    doc_citations = list({
        doc.metadata.get("title", "")
        for doc in source_docs
        if doc.metadata.get("title")
    })
    all_citations = list(dict.fromkeys(parsed["news_citations"] + doc_citations))[:10]

    # ── 9. Persist signal to DB ────────────────────────────────────────────────
    signal_id = save_signal(
        db_path=db_path,
        ticker=ticker,
        signal_strength=parsed["signal_strength"],
        direction=parsed["direction"],
        confidence_score=final_confidence,
        reasoning=parsed["reasoning"],
        news_citations=all_citations,
        rsi=rsi,
        ma20=ma20,
        vol_change_pct=vol_chg,
        generated_at=generated_at,
        model_version=model if use_live_gemini else "mock",
    )

    logger.info(
        "Signal saved: id=%s | %s | %s | strength=%d | confidence=%.2f",
        signal_id[:8], ticker, parsed["direction"],
        parsed["signal_strength"], final_confidence,
    )

    # ── 10. Return AlphaSignal ─────────────────────────────────────────────────
    return AlphaSignal(
        ticker=ticker,
        signal_strength=parsed["signal_strength"],
        direction=parsed["direction"],
        confidence_score=final_confidence,
        reasoning=parsed["reasoning"],
        news_citations=all_citations,
        generated_at=generated_at,
        ticker_rsi=rsi if rsi is not None else float("nan"),
        ticker_ma20=ma20 if ma20 is not None else float("nan"),
        vol_change_pct=vol_chg if vol_chg is not None else float("nan"),
        model_version=model if use_live_gemini else "mock",
        raw_response=raw_answer,
    )
