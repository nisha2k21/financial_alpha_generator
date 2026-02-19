"""
src/agent/nodes.py
Five LangGraph node functions that form the alpha-signal pipeline.

Node execution order (see graph.py for wiring):
  retrieval_node  →  technical_analysis_node  →  gemini_analyst_node
       → [conditional: confidence ≥ 0.40] → long_short_engine_node
       → reporter_node  → END

Each node receives the full AlphaAgentState dict and returns a PARTIAL
dict with only the keys it wants to update — LangGraph merges automatically.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Defaults / constants ──────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
DB_PATH            = os.getenv("DB_PATH", "./data/alpha.db")
GEMINI_MODEL       = "gemini-1.5-pro"
COLLECTION_NAME    = "financial_news"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_chroma_collection():
    """Return (or create) the ChromaDB collection used by embeddings.py."""
    import chromadb
    from chromadb.utils import embedding_functions

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    emb_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=os.getenv("GEMINI_API_KEY", ""),
        model_name="models/embedding-001",
    )
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=emb_fn
        )
    except Exception:
        # Collection may not exist yet — return None, retrieval node handles it
        collection = None
    return collection


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta  = close.diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta.clip(upper=0))
    avg_g  = gain.rolling(period, min_periods=1).mean()
    avg_l  = loss.rolling(period, min_periods=1).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    rsi_s  = 100 - (100 / (1 + rs))
    return float(rsi_s.iloc[-1]) if not rsi_s.empty else float("nan")


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Average True Range — measures price volatility."""
    high  = df["high"]  if "high"  in df.columns else df.get("High",  pd.Series())
    low   = df["low"]   if "low"   in df.columns else df.get("Low",   pd.Series())
    close = df["close"] if "close" in df.columns else df.get("Close", pd.Series())

    if high.empty or low.empty or close.empty:
        # Estimate ATR as 2% of last close price
        last = close.iloc[-1] if not close.empty else 100.0
        return last * 0.02

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period, min_periods=1).mean().iloc[-1])


def _load_prices_df(ticker: str, days: int = 60) -> pd.DataFrame:
    """Load price rows from SQLite for this ticker."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT date, open, high, low, close, volume
            FROM prices
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(ticker, days))
        conn.close()
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as exc:
        logger.warning("Could not load prices for %s: %s", ticker, exc)
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 — RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def retrieval_node(state: dict) -> dict:
    """Query ChromaDB for relevant news chunks for this ticker."""
    ticker = state["ticker"]
    logger.info("[retrieval_node] Fetching news chunks for %s", ticker)

    retrieved: list[dict] = []
    sentiment_score       = 0.0

    try:
        collection = _get_chroma_collection()
        if collection is not None:
            query_text = (
                f"financial news analysis alpha signal {ticker} "
                f"stock market earnings revenue guidance"
            )
            results = collection.query(
                query_texts=[query_text],
                n_results=6,
                include=["documents", "metadatas", "distances"],
            )
            docs      = results.get("documents",  [[]])[0]
            metas     = results.get("metadatas",  [[]])[0]
            distances = results.get("distances",  [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                meta    = meta or {}
                meta_tk = str(meta.get("ticker", "")).upper()
                # Filter: only keep chunks for this ticker OR generic market news
                if meta_tk and meta_tk != ticker.upper():
                    continue
                retrieved.append({
                    "text":      doc,
                    "date":      meta.get("date", ""),
                    "source":    meta.get("source", "unknown"),
                    "sentiment": float(meta.get("sentiment_score", 0.0)),
                    "relevance": round(1 - dist, 3),
                })

            if retrieved:
                sentiment_score = float(
                    np.mean([r["sentiment"] for r in retrieved])
                )
    except Exception as exc:
        logger.warning("[retrieval_node] ChromaDB query failed: %s", exc)

    # Fallback: build dummy context so later nodes still function
    if not retrieved:
        retrieved = [{
            "text": (
                f"No specific news retrieved for {ticker}. "
                "Using technical analysis only."
            ),
            "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            "source": "fallback",
            "sentiment": 0.0,
            "relevance": 0.0,
        }]
        sentiment_score = 0.0

    n = len(retrieved)
    logger.info("[retrieval_node] Got %d chunks, avg sentiment=%.3f", n, sentiment_score)

    return {
        "retrieved_news":  retrieved,
        "sentiment_score": sentiment_score,
        "messages": [
            {"role": "system",
             "content": f"Retrieved {n} news chunks for {ticker} "
                        f"(avg sentiment: {sentiment_score:.3f})"}
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 — TECHNICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def technical_analysis_node(state: dict) -> dict:
    """Pull 60-day OHLCV from SQLite and compute technical indicators."""
    ticker = state["ticker"]
    logger.info("[technical_analysis_node] Computing technicals for %s", ticker)

    df = _load_prices_df(ticker, days=80)   # extra rows for MA-50

    indicators: dict[str, Any] = {}

    if df.empty or len(df) < 20:
        # Not enough data — return neutral defaults
        logger.warning("[technical_analysis_node] Insufficient price data for %s", ticker)
        indicators = {
            "RSI": 50.0,
            "MA_20": 0.0, "MA_50": 0.0,
            "MACD": 0.0, "MACD_signal": 0.0,
            "volume_change_pct": 0.0,
            "price_vs_MA20": 0.0,
            "bollinger_position": 0.5,
            "current_price": 0.0,
            "atr": 0.0,
            "technical_verdict": "NEUTRAL",
            "bullish_signals": 0,
        }
        return {
            "technical_indicators": indicators,
            "technical_verdict":    "NEUTRAL",
            "messages": [
                {"role": "system",
                 "content": f"Technical analysis for {ticker}: insufficient price data, defaulting to NEUTRAL"}
            ],
        }

    close  = df["close"]
    volume = df["volume"]
    current_price = float(close.iloc[-1])

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    rsi_val = _rsi(close, 14)

    # ── Moving averages ───────────────────────────────────────────────────────
    ma20 = float(close.rolling(20, min_periods=1).mean().iloc[-1])
    ma50 = float(close.rolling(50, min_periods=1).mean().iloc[-1]) if len(df) >= 50 else ma20

    # ── MACD (12, 26, 9) ─────────────────────────────────────────────────────
    ema12      = _ema(close, 12)
    ema26      = _ema(close, 26)
    macd_line  = ema12 - ema26
    macd_sig   = _ema(macd_line, 9)
    macd_val   = float(macd_line.iloc[-1])
    macd_signal_val = float(macd_sig.iloc[-1])

    # ── Volume change ─────────────────────────────────────────────────────────
    avg_vol_20    = float(volume.rolling(20, min_periods=1).mean().iloc[-1])
    today_vol     = float(volume.iloc[-1])
    vol_change_pct = (
        (today_vol - avg_vol_20) / avg_vol_20 * 100
        if avg_vol_20 > 0 else 0.0
    )

    # ── Price vs MA20 ─────────────────────────────────────────────────────────
    price_vs_ma20 = (current_price - ma20) / ma20 * 100 if ma20 > 0 else 0.0

    # ── Bollinger Bands position (0 = at lower band, 1 = at upper band) ───────
    rolling_std   = close.rolling(20, min_periods=1).std()
    upper_band    = ma20 + 2 * float(rolling_std.iloc[-1])
    lower_band    = ma20 - 2 * float(rolling_std.iloc[-1])
    band_range    = upper_band - lower_band
    bollinger_pos = (
        (current_price - lower_band) / band_range
        if band_range > 0 else 0.5
    )
    bollinger_pos = max(0.0, min(1.0, bollinger_pos))

    # ── ATR ───────────────────────────────────────────────────────────────────
    atr_val = _atr(df)

    # ── Scoring ───────────────────────────────────────────────────────────────
    score = 0
    if rsi_val < 30:             score += 2   # oversold → bullish
    elif rsi_val > 70:           score -= 2   # overbought → bearish
    if current_price > ma20:     score += 1
    if current_price > ma50:     score += 1
    if macd_val > macd_signal_val: score += 1
    if vol_change_pct > 20:      score += 1   # high volume confirms move

    if   score >= 4:  verdict = "STRONGLY_BULLISH"
    elif score >= 2:  verdict = "BULLISH"
    elif score >= 0:  verdict = "NEUTRAL"
    elif score >= -2: verdict = "BEARISH"
    else:             verdict = "STRONGLY_BEARISH"

    indicators = {
        "RSI":               round(rsi_val, 2),
        "MA_20":             round(ma20, 4),
        "MA_50":             round(ma50, 4),
        "MACD":              round(macd_val, 4),
        "MACD_signal":       round(macd_signal_val, 4),
        "volume_change_pct": round(vol_change_pct, 2),
        "price_vs_MA20":     round(price_vs_ma20, 2),
        "bollinger_position":round(bollinger_pos, 3),
        "current_price":     round(current_price, 4),
        "atr":               round(atr_val, 4),
        "technical_verdict": verdict,
        "bullish_signals":   score,
    }

    logger.info(
        "[technical_analysis_node] %s → RSI=%.1f MA20=%.2f verdict=%s score=%d",
        ticker, rsi_val, ma20, verdict, score,
    )

    return {
        "technical_indicators": indicators,
        "technical_verdict":    verdict,
        "messages": [
            {"role": "system",
             "content": (
                 f"Technical analysis for {ticker}: RSI={rsi_val:.1f}, "
                 f"MA20={ma20:.2f}, MACD={'bullish' if macd_val > macd_signal_val else 'bearish'}, "
                 f"verdict={verdict} (score={score})"
             )}
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 — GEMINI ANALYST
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior quantitative analyst at a hedge fund.
You specialize in generating precise alpha signals from
news sentiment combined with technical analysis.

Your output MUST be valid JSON and follow this EXACT format:
{
  "alpha_signal": "STRONG_BUY|BUY|NEUTRAL|SELL|STRONG_SELL",
  "confidence_score": 0.0-1.0,
  "news_summary": "2-3 sentence summary of key news catalysts",
  "key_bullish_factors": ["factor1", "factor2"],
  "key_bearish_factors": ["factor1", "factor2"],
  "signal_reasoning": "3-4 sentence explanation connecting news to signal"
}

Calibration rules:
- STRONG_BUY:  very positive news + bullish technicals, confidence > 0.75
- BUY:         positive news OR bullish technicals (not both required), confidence 0.55-0.75
- NEUTRAL:     conflicting signals or insufficient data, confidence 0.40-0.55
- SELL:        negative news OR bearish technicals, confidence 0.55-0.75
- STRONG_SELL: very negative news + bearish technicals, confidence > 0.75
"""


def _build_gemini_user_message(state: dict) -> str:
    ticker = state["ticker"]
    ind    = state.get("technical_indicators", {})
    news   = state.get("retrieved_news", [])
    sent   = state.get("sentiment_score", 0.0)

    rsi       = ind.get("RSI", 50)
    verdict   = ind.get("technical_verdict", "NEUTRAL")
    pma20     = ind.get("price_vs_MA20", 0.0)
    macd      = ind.get("MACD", 0.0)
    macd_sig  = ind.get("MACD_signal", 0.0)
    macd_dir  = "bullish crossover" if macd > macd_sig else "bearish crossover"

    news_lines = []
    for chunk in news[:5]:
        date   = chunk.get("date", "")
        source = chunk.get("source", "")
        text   = chunk.get("text", "")[:400]
        news_lines.append(f"[{date} | {source}]: {text}")
    news_context = "\n\n".join(news_lines) if news_lines else "No news available."

    return (
        f"Ticker: {ticker}\n"
        f"Current RSI: {rsi:.1f} | Technical verdict: {verdict}\n"
        f"Price vs 20-day MA: {pma20:+.2f}%\n"
        f"MACD signal: {macd_dir}\n"
        f"Sentiment score (news): {sent:.3f}  (-1 = very negative, +1 = very positive)\n\n"
        f"Recent news context:\n{news_context}\n\n"
        "Generate the alpha signal JSON now. Return ONLY the JSON, no markdown."
    )


def _parse_gemini_json(raw: str) -> dict:
    """Extract JSON from Gemini response, robust to markdown fences."""
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"\s*```$",          "", raw.strip(), flags=re.MULTILINE)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise


def gemini_analyst_node(state: dict) -> dict:
    """Call Gemini 1.5 Pro to synthesise news + technicals → alpha signal JSON."""
    ticker = state["ticker"]
    logger.info("[gemini_analyst_node] Calling Gemini for %s", ticker)

    gemini_key = os.getenv("GEMINI_API_KEY", "")

    # ── Mock response when no API key ─────────────────────────────────────────
    if not gemini_key:
        logger.warning("[gemini_analyst_node] No GEMINI_API_KEY — using mock signal")
        ind     = state.get("technical_indicators", {})
        verdict = ind.get("technical_verdict", "NEUTRAL")
        sent    = state.get("sentiment_score", 0.0)

        # Heuristic fallback signal
        if "STRONGLY_BULLISH" in verdict and sent > 0.1:
            sig, conf = "STRONG_BUY", 0.82
        elif "BULLISH" in verdict or sent > 0.15:
            sig, conf = "BUY", 0.65
        elif "STRONGLY_BEARISH" in verdict and sent < -0.1:
            sig, conf = "STRONG_SELL", 0.78
        elif "BEARISH" in verdict or sent < -0.15:
            sig, conf = "SELL", 0.62
        else:
            sig, conf = "NEUTRAL", 0.48

        parsed = {
            "alpha_signal":       sig,
            "confidence_score":   conf,
            "news_summary":       f"[Demo] Heuristic signal for {ticker} based on {verdict} technicals.",
            "key_bullish_factors":["RSI near oversold" if ind.get("RSI", 50) < 45 else "MA uptrend"],
            "key_bearish_factors":["Low news availability"],
            "signal_reasoning":   f"[Demo] Technical verdict is {verdict} with RSI={ind.get('RSI', 50):.1f}.",
        }
    else:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=gemini_key,
                temperature=0.1,
            )
            user_msg = _build_gemini_user_message(state)
            response = llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ])
            parsed = _parse_gemini_json(response.content)
        except Exception as exc:
            logger.error("[gemini_analyst_node] Gemini call failed: %s", exc)
            parsed = {
                "alpha_signal":       "NEUTRAL",
                "confidence_score":   0.42,
                "news_summary":       f"LLM call failed ({exc}). Defaulting to NEUTRAL.",
                "key_bullish_factors":[],
                "key_bearish_factors":["API error"],
                "signal_reasoning":   "Could not obtain LLM analysis.",
            }

    alpha_signal   = str(parsed.get("alpha_signal",     "NEUTRAL")).upper()
    confidence     = float(parsed.get("confidence_score", 0.42))
    news_summary   = str(parsed.get("news_summary",     "No summary available."))
    reasoning      = str(parsed.get("signal_reasoning", ""))

    # Clamp confidence
    confidence = max(0.0, min(1.0, confidence))

    logger.info(
        "[gemini_analyst_node] %s → %s (conf=%.2f)", ticker, alpha_signal, confidence
    )

    return {
        "alpha_signal":    alpha_signal,
        "confidence_score":confidence,
        "news_summary":    news_summary,
        "messages": [
            {"role": "assistant",
             "content": f"Gemini signal for {ticker}: {alpha_signal} "
                        f"(confidence={confidence:.0%}). {reasoning}"}
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 — LONG / SHORT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def long_short_engine_node(state: dict) -> dict:
    """Convert alpha signal into actionable trade recommendation."""
    ticker       = state["ticker"]
    alpha_signal = state.get("alpha_signal",    "NEUTRAL")
    conf         = state.get("confidence_score", 0.0)
    ind          = state.get("technical_indicators", {})
    news_summary = state.get("news_summary", "")
    verdict      = ind.get("technical_verdict", "NEUTRAL")
    rsi          = ind.get("RSI",               50.0)
    current_price= float(ind.get("current_price", 0.0))
    atr          = float(ind.get("atr",           current_price * 0.02))

    logger.info(
        "[long_short_engine_node] %s signal=%s conf=%.2f price=%.4f atr=%.4f",
        ticker, alpha_signal, conf, current_price, atr,
    )

    # ── 1. Position Direction ────────────────────────────────────────────────
    if alpha_signal in ("STRONG_BUY", "BUY") and conf > 0.55:
        position = "LONG"
    elif alpha_signal in ("STRONG_SELL", "SELL") and conf > 0.55:
        position = "SHORT"
    else:
        position = "NO_TRADE"

    # ── 2. Position Sizing — half-Kelly ──────────────────────────────────────
    if position != "NO_TRADE":
        kelly          = (conf - (1 - conf)) / 1.0      # win_prob - loss_prob
        kelly          = max(0.0, min(kelly, 0.25))     # cap raw Kelly at 25%
        size_pct       = (kelly / 2.0) * 100            # half-Kelly → %
        if "STRONG" in alpha_signal:
            size_pct  *= 1.2
        size_pct       = round(min(size_pct, 10.0), 2)  # hard cap 10%
    else:
        size_pct = 0.0

    # ── 3. Price Levels (ATR-based) ───────────────────────────────────────────
    if current_price <= 0 or atr <= 0:
        # No price data — neutral defaults
        entry_price     = 0.0
        stop_loss       = 0.0
        take_profit     = 0.0
        risk_reward     = 0.0
    elif position == "LONG":
        entry_price  = current_price
        stop_loss    = current_price - 1.5 * atr
        take_profit  = current_price + 2.5 * atr
        risk_reward  = (take_profit  - entry_price) / (entry_price - stop_loss)
    elif position == "SHORT":
        entry_price  = current_price
        stop_loss    = current_price + 1.5 * atr
        take_profit  = current_price - 2.5 * atr
        risk_reward  = (entry_price  - take_profit) / (stop_loss - entry_price)
    else:  # NO_TRADE
        entry_price  = current_price
        stop_loss    = 0.0
        take_profit  = 0.0
        risk_reward  = 0.0

    # ── 4. Trade Rationale text ───────────────────────────────────────────────
    if position == "NO_TRADE":
        sl_str = "N/A"
        tp_str = "N/A"
        sl_pct = 0.0
        tp_pct = 0.0
    else:
        sl_pct = (stop_loss  - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
        tp_pct = (take_profit - entry_price) / entry_price * 100 if entry_price > 0 else 0.0
        sl_str = f"${stop_loss:.2f} ({sl_pct:+.1f}%)"
        tp_str = f"${take_profit:.2f} ({tp_pct:+.1f}%)"

    trade_rationale = (
        f"Position: {position} {ticker} @ ${entry_price:.2f}\n"
        f"Signal: {alpha_signal} (confidence: {conf:.0%})\n"
        f"Size: {size_pct}% of portfolio (half-Kelly sizing)\n"
        f"Stop Loss: {sl_str}\n"
        f"Take Profit: {tp_str}\n"
        f"Risk/Reward: 1:{risk_reward:.1f}\n"
        f"Key drivers: {news_summary}\n"
        f"Technical confirmation: RSI={rsi:.1f}, {verdict}"
    )

    logger.info(
        "[long_short_engine_node] %s → %s %.1f%% SL=%.2f TP=%.2f RR=1:%.1f",
        ticker, position, size_pct, stop_loss, take_profit, risk_reward,
    )

    return {
        "position_recommendation": position,
        "position_size_pct":       size_pct,
        "entry_price":             round(entry_price, 4),
        "stop_loss_price":         round(stop_loss,   4),
        "take_profit_price":       round(take_profit,  4),
        "risk_reward_ratio":       round(risk_reward,  3),
        "trade_rationale":         trade_rationale,
        "messages": [
            {"role": "assistant",
             "content": (
                 f"Trade recommendation for {ticker}: {position} "
                 f"{size_pct}% @ {entry_price:.2f}  SL={stop_loss:.2f}  "
                 f"TP={take_profit:.2f}  R:R=1:{risk_reward:.1f}"
             )}
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 — REPORTER
# ─────────────────────────────────────────────────────────────────────────────

def reporter_node(state: dict) -> dict:
    """Persist final signal + trade details to SQLite, mark pipeline complete."""
    ticker = state["ticker"]
    logger.info("[reporter_node] Saving final signal for %s", ticker)

    try:
        _save_signal_to_db(state)
    except Exception as exc:
        logger.error("[reporter_node] DB save failed for %s: %s", ticker, exc)

    return {
        "is_complete": True,
        "messages": [
            {"role": "system",
             "content": (
                 f"Pipeline complete for {ticker}. "
                 f"Signal={state.get('alpha_signal','?')} "
                 f"Position={state.get('position_recommendation','?')}"
             )}
        ],
    }


def _save_signal_to_db(state: dict) -> None:
    """Write (or update) the signal row in SQLite with extended trade columns."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()

        # Ensure new columns exist (idempotent)
        new_cols = [
            ("position",          "TEXT"),
            ("position_size_pct", "REAL"),
            ("entry_price",       "REAL"),
            ("stop_loss_price",   "REAL"),
            ("take_profit_price", "REAL"),
            ("risk_reward_ratio", "REAL"),
            ("trade_rationale",   "TEXT"),
        ]
        existing = {
            row[1]
            for row in cur.execute("PRAGMA table_info(signals)").fetchall()
        }
        for col_name, col_type in new_cols:
            if col_name not in existing:
                cur.execute(
                    f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}"
                )

        ind         = state.get("technical_indicators", {})
        news_chunks = state.get("retrieved_news", [])
        citations   = [c.get("source", "") for c in news_chunks[:3]]

        cur.execute("""
            INSERT OR REPLACE INTO signals (
                ticker, signal_strength, direction, confidence_score,
                reasoning, news_citations, generated_at,
                ticker_rsi, ticker_ma20, vol_change_pct,
                model_version, raw_response,
                position, position_size_pct, entry_price,
                stop_loss_price, take_profit_price, risk_reward_ratio,
                trade_rationale
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            state.get("ticker",         ""),
            _signal_to_strength(state.get("alpha_signal", "NEUTRAL")),
            state.get("alpha_signal",   "NEUTRAL"),
            state.get("confidence_score", 0.0),
            state.get("news_summary",   ""),
            json.dumps(citations),
            datetime.now(tz=timezone.utc).isoformat(),
            ind.get("RSI",             float("nan")),
            ind.get("MA_20",           float("nan")),
            ind.get("volume_change_pct", float("nan")),
            "langgraph-gemini-1.5-pro",
            json.dumps({"technical_indicators": ind}),
            state.get("position_recommendation", "NO_TRADE"),
            state.get("position_size_pct",    0.0),
            state.get("entry_price",           0.0),
            state.get("stop_loss_price",       0.0),
            state.get("take_profit_price",     0.0),
            state.get("risk_reward_ratio",     0.0),
            state.get("trade_rationale",       ""),
        ))
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.error("_save_signal_to_db failed: %s", exc)
        raise


def _signal_to_strength(signal: str) -> int:
    mapping = {
        "STRONG_BUY":  5,
        "BUY":         4,
        "NEUTRAL":     3,
        "SELL":        2,
        "STRONG_SELL": 1,
    }
    return mapping.get(signal.upper().replace(" ", "_"), 3)
