"""
src/ingestion.py
Fetch financial news and stock price data, then compute technical indicators.

Key functions
-------------
fetch_news(ticker, api_key, ...)          — NewsAPI with sample fallback
fetch_news_for_tickers(tickers, api_key)  — Batch fetch for a list of tickers
fetch_stock_data(ticker, period)          — yfinance OHLCV (default 6 months)
compute_technical_indicators(df)          — RSI(14), MA20, vol_chg_pct
save_news_to_db(articles, db_path)        — Upsert articles into `news` table
save_prices_to_db(df, ticker, db_path)    — Upsert prices into `prices` table
summarise_technicals(df, ticker)          — Human-readable technical summary
format_articles_for_rag(articles)         — Convert to LangChain Documents
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential
from ta.momentum import RSIIndicator

from .database import save_articles_batch, save_prices_batch

logger = logging.getLogger(__name__)

SAMPLE_NEWS_PATH = Path(__file__).parent.parent / "data" / "sample_news.json"

# Default target tickers (free-tier NewsAPI covers well)
DEFAULT_TICKERS = ["AAPL", "GOOGL", "TSLA", "MSFT", "NVDA"]

# Company name → search query mapping
COMPANY_NAMES: dict[str, str] = {
    "AAPL":  "Apple stock",
    "TSLA":  "Tesla stock",
    "NVDA":  "NVIDIA stock",
    "MSFT":  "Microsoft stock",
    "GOOGL": "Google Alphabet stock",
    "AMZN":  "Amazon stock",
    "META":  "Meta Platforms stock",
    "JPM":   "JPMorgan stock",
    "NFLX":  "Netflix stock",
    "AMD":   "AMD semiconductor stock",
}


# ─── NewsAPI ──────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_news(
    ticker: str,
    api_key: str,
    page_size: int = 10,
    days_back: int = 7,
) -> list[dict]:
    """
    Fetch recent financial news articles for a single ticker from NewsAPI.

    Falls back to ``data/sample_news.json`` when:
    - API key is absent or placeholder
    - Network request fails after 3 retries

    Parameters
    ----------
    ticker    : Stock ticker, e.g. "AAPL"
    api_key   : NewsAPI free-tier key
    page_size : Max articles (up to 100 on free tier)
    days_back : Lookback window in days (free tier capped at 30 days back)

    Returns
    -------
    List of article dicts with keys:
        id, ticker, title, source, author, publishedAt, content
    """
    if not api_key or api_key.startswith("your_"):
        logger.warning("No valid NEWS_API_KEY — using sample news for %s", ticker)
        return load_sample_news(ticker)

    try:
        from newsapi import NewsApiClient  # lazy import to avoid breaking offline mode

        newsapi = NewsApiClient(api_key=api_key)
        from_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
        query = COMPANY_NAMES.get(ticker.upper(), f"{ticker} stock")

        response = newsapi.get_everything(
            q=f'"{query}"',
            language="en",
            sort_by="relevancy",
            page_size=page_size,
            from_param=from_date,
        )

        articles = []
        for i, art in enumerate(response.get("articles", [])):
            articles.append({
                "id": f"live_{ticker}_{i:03d}",
                "ticker": ticker.upper(),
                "title": art.get("title", ""),
                "source": art.get("source", {}).get("name", "Unknown"),
                "author": art.get("author", ""),
                "publishedAt": art.get("publishedAt", ""),
                "content": (art.get("content") or art.get("description") or "")[:2000],
            })

        logger.info("Fetched %d articles for %s from NewsAPI", len(articles), ticker)
        return articles if articles else load_sample_news(ticker)

    except Exception as exc:
        logger.error("NewsAPI fetch failed for %s: %s — fallback to sample", ticker, exc)
        return load_sample_news(ticker)


def fetch_news_for_tickers(
    tickers: list[str],
    api_key: str,
    page_size: int = 10,
    days_back: int = 7,
) -> dict[str, list[dict]]:
    """
    Batch-fetch news for a list of tickers.

    Each ticker is fetched independently so rate limits apply per ticker.
    Results are keyed by uppercase ticker.

    Parameters
    ----------
    tickers   : List of ticker symbols, e.g. ["AAPL", "GOOGL", "TSLA"]
    api_key   : NewsAPI key
    page_size : Articles per ticker
    days_back : Lookback window

    Returns
    -------
    Dict mapping ticker → list of article dicts
    """
    results: dict[str, list[dict]] = {}
    for ticker in tickers:
        results[ticker.upper()] = fetch_news(ticker, api_key, page_size, days_back)
        logger.info("Batch fetch: %s → %d articles", ticker, len(results[ticker.upper()]))
    return results


# ─── Sample News Fallback ─────────────────────────────────────────────────────

def load_sample_news(ticker: Optional[str] = None) -> list[dict]:
    """
    Load articles from ``data/sample_news.json``.

    If ticker is given, returns only matching articles.
    Falls back to all articles when no ticker match is found.

    Parameters
    ----------
    ticker : Optional ticker filter

    Returns
    -------
    List of article dicts
    """
    try:
        with open(SAMPLE_NEWS_PATH, "r", encoding="utf-8") as f:
            all_articles = json.load(f)
    except FileNotFoundError:
        logger.error("sample_news.json not found at %s", SAMPLE_NEWS_PATH)
        return []

    if ticker:
        filtered = [a for a in all_articles if a.get("ticker", "").upper() == ticker.upper()]
        return filtered if filtered else all_articles

    return all_articles


# ─── Stock Price Data ─────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_stock_data(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Fetch OHLCV price data for a ticker using yfinance.

    Parameters
    ----------
    ticker : Stock ticker symbol
    period : yfinance period string. Default ``"6mo"`` (~126 trading days).
             Supports: "1mo", "3mo", "6mo", "1y", "2y"

    Returns
    -------
    DataFrame with columns: Open, High, Low, Close, Volume, ticker.
    Index is DatetimeIndex (timezone-naive for SQLite compatibility).
    Returns empty DataFrame on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)

        if df.empty:
            logger.warning("yfinance returned empty data for %s", ticker)
            return pd.DataFrame()

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = df.index.tz_localize(None)  # Strip tz for SQLite compatibility
        df.index.name = "Date"
        df["ticker"] = ticker.upper()

        logger.info("Fetched %d rows of %s price data for %s", len(df), period, ticker)
        return df

    except Exception as exc:
        logger.error("yfinance fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ─── Technical Indicators ─────────────────────────────────────────────────────

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI(14), MA20, and volume_change_pct columns to an OHLCV DataFrame.

    Indicator definitions
    ---------------------
    RSI (Relative Strength Index, 14-period)
        Classic Wilder RSI using the ``ta`` library's RSIIndicator.
        Values: 0–100. <30 = oversold, >70 = overbought.

    MA20 (20-day Simple Moving Average)
        Rolling 20-day mean of Close price.

    vol_chg_pct (Volume Change %)
        Percentage difference between today's volume and the 20-day
        average volume: ``(volume / rolling_20d_avg - 1) * 100``.
        Positive = above-average volume; negative = below-average.

    Parameters
    ----------
    df : OHLCV DataFrame as returned by ``fetch_stock_data()``.
        Must have ``Close`` and ``Volume`` columns.

    Returns
    -------
    DataFrame with three new columns: ``RSI``, ``MA20``, ``vol_chg_pct``.
    Rows with insufficient history for calculation will have NaN values.
    """
    if df.empty:
        return df

    result = df.copy()

    try:
        # RSI(14) — standard 14-period Wilder RSI
        rsi_indicator = RSIIndicator(close=result["Close"], window=14)
        result["RSI"] = rsi_indicator.rsi()
    except Exception as exc:
        logger.warning("RSI calculation failed: %s — using NaN", exc)
        result["RSI"] = float("nan")

    # MA20 — 20-day simple moving average
    result["MA20"] = result["Close"].rolling(window=20, min_periods=1).mean()

    # Volume change % vs 20-day rolling average
    vol_ma20 = result["Volume"].rolling(window=20, min_periods=1).mean()
    result["vol_chg_pct"] = ((result["Volume"] / vol_ma20) - 1) * 100

    logger.info(
        "Computed indicators for %s: RSI=%.1f, MA20=%.2f, vol_chg=%.1f%%",
        df["ticker"].iloc[-1] if "ticker" in df.columns else "?",
        result["RSI"].iloc[-1] if not result["RSI"].isna().all() else float("nan"),
        result["MA20"].iloc[-1],
        result["vol_chg_pct"].iloc[-1],
    )
    return result


# ─── Database Persistence ─────────────────────────────────────────────────────

def save_news_to_db(articles: list[dict], db_path: str) -> int:
    """
    Persist a list of article dicts to the ``news`` SQLite table.

    Sentiment scores should already be attached to each article dict
    (key: ``sentiment_score``) by the embeddings layer before calling this.
    If absent, defaults to 0.0.

    Parameters
    ----------
    articles : List of article dicts
    db_path  : Path to SQLite database

    Returns
    -------
    Number of articles persisted
    """
    return save_articles_batch(db_path, articles)


def save_prices_to_db(df: pd.DataFrame, ticker: str, db_path: str) -> int:
    """
    Persist OHLCV + technical indicator rows to the ``prices`` SQLite table.

    Parameters
    ----------
    df      : DataFrame returned by ``compute_technical_indicators()``
    ticker  : Ticker symbol (redundant since df has it, kept for clarity)
    db_path : Path to SQLite database

    Returns
    -------
    Number of rows persisted
    """
    if df.empty:
        return 0

    records = []
    for date_idx, row in df.iterrows():
        records.append({
            "ticker": ticker.upper(),
            "date": str(date_idx)[:10],  # YYYY-MM-DD
            "open": round(float(row.get("Open", 0) or 0), 4),
            "high": round(float(row.get("High", 0) or 0), 4),
            "low": round(float(row.get("Low", 0) or 0), 4),
            "close": round(float(row.get("Close", 0) or 0), 4),
            "volume": int(row.get("Volume", 0) or 0),
            "rsi": round(float(row["RSI"]), 2) if pd.notna(row.get("RSI")) else None,
            "ma20": round(float(row["MA20"]), 4) if pd.notna(row.get("MA20")) else None,
            "vol_chg_pct": round(float(row["vol_chg_pct"]), 2) if pd.notna(row.get("vol_chg_pct")) else None,
        })

    return save_prices_batch(db_path, records)


# ─── Text Summary for RAG Prompt ──────────────────────────────────────────────

def summarise_technicals(df: pd.DataFrame, ticker: str) -> str:
    """
    Generate a concise technical analysis summary string for the RAG prompt.

    Includes price performance, RSI regime, MA20 context, and volume signal.
    Used as supplemental context injected into the quant analyst prompt.

    Parameters
    ----------
    df     : DataFrame with Close, RSI, MA20, vol_chg_pct columns
    ticker : Ticker symbol

    Returns
    -------
    Formatted multi-line string summarising key technicals
    """
    if df.empty:
        return f"No price data available for {ticker}."

    close = df["Close"]
    current_price = close.iloc[-1]
    start_price = close.iloc[0]
    pct_change = (current_price - start_price) / start_price * 100

    # RSI regime label
    rsi = df["RSI"].iloc[-1] if "RSI" in df.columns and not df["RSI"].isna().all() else None
    if rsi is not None:
        if rsi < 30:
            rsi_regime = f"OVERSOLD ({rsi:.1f})"
        elif rsi > 70:
            rsi_regime = f"OVERBOUGHT ({rsi:.1f})"
        else:
            rsi_regime = f"NEUTRAL ({rsi:.1f})"
    else:
        rsi_regime = "N/A"

    # MA20 relationship
    ma20 = df["MA20"].iloc[-1] if "MA20" in df.columns else None
    if ma20 is not None:
        ma_signal = "above MA20 ✅" if current_price > ma20 else "below MA20 ⚠️"
        ma_text = f"${ma20:.2f} — price is {ma_signal}"
    else:
        ma_text = "N/A"

    # Volume signal
    vol_chg = df["vol_chg_pct"].iloc[-1] if "vol_chg_pct" in df.columns else None
    if vol_chg is not None:
        vol_text = f"{vol_chg:+.1f}% vs 20-day avg"
        if vol_chg > 50:
            vol_text += " 🔥 (high conviction)"
        elif vol_chg < -30:
            vol_text += " (low activity)"
    else:
        vol_text = "N/A"

    # Short-term momentum (5-day vs prior 5-day)
    if len(close) >= 10:
        recent_avg = close.tail(5).mean()
        prior_avg = close.iloc[-10:-5].mean()
        momentum = "BULLISH 📈" if recent_avg > prior_avg else "BEARISH 📉"
    else:
        momentum = "INSUFFICIENT DATA"

    return (
        f"═══ Technical Summary: {ticker} ═══\n"
        f"Period: {len(df)} trading days | Current: ${current_price:.2f} ({pct_change:+.1f}%)\n"
        f"20-Day High: ${df['High'].tail(20).max():.2f} | Low: ${df['Low'].tail(20).min():.2f}\n"
        f"RSI(14): {rsi_regime}\n"
        f"MA20: {ma_text}\n"
        f"Volume: {vol_text}\n"
        f"5-Day Momentum: {momentum}"
    )


# Legacy alias for backward compat with app.py
summarise_stock_data = summarise_technicals


# ─── LangChain Document Formatting ───────────────────────────────────────────

def format_articles_for_rag(articles: list[dict]) -> list[Document]:
    """
    Convert raw article dicts into LangChain ``Document`` objects.

    Each Document's ``page_content`` is formatted for optimal embedding
    quality — including TITLE, TICKER, SOURCE, DATE, and body so semantic
    search works across company name, theme, and keyword queries.

    Sentiment scores (if present) are carried through to chunk metadata.

    Parameters
    ----------
    articles : List of article dicts (from ``fetch_news`` or sample data)

    Returns
    -------
    List of LangChain Document objects ready for chunking and embedding
    """
    docs = []
    for art in articles:
        if not art.get("content") and not art.get("title"):
            continue

        page_content = (
            f"TITLE: {art.get('title', 'N/A')}\n"
            f"TICKER: {art.get('ticker', 'N/A')}\n"
            f"SOURCE: {art.get('source', 'N/A')}\n"
            f"DATE: {art.get('publishedAt', 'N/A')[:10]}\n\n"
            f"{art.get('content', '')}"
        )

        metadata = {
            "article_id":     art.get("id", ""),
            "ticker":         art.get("ticker", ""),
            "title":          art.get("title", ""),
            "source":         art.get("source", ""),
            "author":         art.get("author", ""),
            "published_at":   art.get("publishedAt", "")[:10],
            "sentiment_score": art.get("sentiment_score", 0.0),
        }

        docs.append(Document(page_content=page_content, metadata=metadata))

    logger.info("Formatted %d articles into LangChain Documents", len(docs))
    return docs
