"""
src/database.py
SQLite persistence layer with BigQuery-compatible schema.

Tables
------
news       — Raw news articles with sentiment scores
prices     — OHLCV + technical indicators per ticker per day
signals    — AlphaSignal objects (new schema with signal_strength 1–5)

Legacy tables (preserved for backward compatibility)
----------------------------------------------------
alpha_signals      — Previous signal format
news_ingestion_log — Previous ingestion log
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Schema: news ─────────────────────────────────────────────────────────────

CREATE_NEWS_TABLE = """
CREATE TABLE IF NOT EXISTS news (
    article_id    TEXT    PRIMARY KEY,
    ticker        TEXT    NOT NULL,
    title         TEXT    NOT NULL,
    source        TEXT,
    author        TEXT,
    published_at  TEXT,
    content       TEXT,
    sentiment_score REAL  DEFAULT 0.0,
    ingested_at   TEXT    NOT NULL
);
"""

# ─── Schema: prices ───────────────────────────────────────────────────────────

CREATE_PRICES_TABLE = """
CREATE TABLE IF NOT EXISTS prices (
    price_id      TEXT    PRIMARY KEY,
    ticker        TEXT    NOT NULL,
    date          TEXT    NOT NULL,
    open          REAL,
    high          REAL,
    low           REAL,
    close         REAL    NOT NULL,
    volume        INTEGER,
    rsi           REAL,
    ma20          REAL,
    vol_chg_pct   REAL,
    UNIQUE(ticker, date)
);
"""

# ─── Schema: signals ──────────────────────────────────────────────────────────

CREATE_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS signals (
    signal_id        TEXT    PRIMARY KEY,
    ticker           TEXT    NOT NULL,
    signal_strength  INTEGER NOT NULL CHECK(signal_strength BETWEEN 1 AND 5),
    direction        TEXT    NOT NULL,
    confidence_score REAL    NOT NULL CHECK(confidence_score BETWEEN 0.0 AND 1.0),
    reasoning        TEXT    NOT NULL,
    news_citations   TEXT,
    rsi              REAL,
    ma20             REAL,
    vol_change_pct   REAL,
    generated_at     TEXT    NOT NULL,
    model_version    TEXT
);
"""

# ─── Legacy schemas (backward compat) ────────────────────────────────────────

CREATE_ALPHA_SIGNALS_LEGACY = """
CREATE TABLE IF NOT EXISTS alpha_signals (
    signal_id    TEXT    PRIMARY KEY,
    ticker       TEXT    NOT NULL,
    signal_type  TEXT    NOT NULL,
    score        REAL    NOT NULL,
    rationale    TEXT    NOT NULL,
    key_risks    TEXT,
    sources      TEXT,
    created_at   TEXT    NOT NULL,
    model_version TEXT   NOT NULL
);
"""

CREATE_NEWS_LOG_LEGACY = """
CREATE TABLE IF NOT EXISTS news_ingestion_log (
    log_id        TEXT    PRIMARY KEY,
    ticker        TEXT    NOT NULL,
    article_id    TEXT    NOT NULL,
    article_title TEXT    NOT NULL,
    source        TEXT,
    published_at  TEXT,
    ingested_at   TEXT    NOT NULL,
    chunk_count   INTEGER DEFAULT 0
);
"""

CREATE_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_news_ticker ON news(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_news_published ON news(published_at);",
    "CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, date);",
    "CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_signals_generated ON signals(generated_at);",
    # Legacy
    "CREATE INDEX IF NOT EXISTS idx_alpha_ticker ON alpha_signals(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_ingestion_ticker ON news_ingestion_log(ticker);",
]


# ─── Connection ───────────────────────────────────────────────────────────────

def get_connection(db_path: str) -> sqlite3.Connection:
    """Return a WAL-mode SQLite connection with dict-like row factory."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: str) -> None:
    """
    Initialise the database — create all tables and indices if absent.

    Safe to call multiple times (idempotent). Creates parent directories.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    try:
        for ddl in [
            CREATE_NEWS_TABLE,
            CREATE_PRICES_TABLE,
            CREATE_SIGNALS_TABLE,
            CREATE_ALPHA_SIGNALS_LEGACY,
            CREATE_NEWS_LOG_LEGACY,
        ]:
            conn.execute(ddl)
        for idx_sql in CREATE_INDICES:
            conn.execute(idx_sql)
        conn.commit()
        logger.info("Database initialised at %s", db_path)
    finally:
        conn.close()


# ─── News CRUD ────────────────────────────────────────────────────────────────

def save_article(
    db_path: str,
    *,
    article_id: str,
    ticker: str,
    title: str,
    source: str = "",
    author: str = "",
    published_at: str = "",
    content: str = "",
    sentiment_score: float = 0.0,
) -> None:
    """
    Upsert a single news article into the `news` table.

    Uses INSERT OR REPLACE so re-fetched articles don't create duplicates.

    Parameters
    ----------
    db_path        : Path to the SQLite database file
    article_id     : Unique identifier for the article
    ticker         : Stock ticker this article is tagged to
    title          : Article headline
    source         : Publication source name
    author         : Byline
    published_at   : ISO-8601 publish date/time string
    content        : Full article body text
    sentiment_score: TextBlob polarity score (-1.0 to +1.0)
    """
    ingested_at = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO news
              (article_id, ticker, title, source, author,
               published_at, content, sentiment_score, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (article_id, ticker, title, source, author,
             published_at, content, sentiment_score, ingested_at),
        )
        conn.commit()
    finally:
        conn.close()


def save_articles_batch(db_path: str, articles: list[dict]) -> int:
    """
    Batch-upsert multiple articles in a single transaction.

    Parameters
    ----------
    db_path  : SQLite database path
    articles : List of article dicts (keys match `news` table columns)

    Returns
    -------
    Number of rows inserted/replaced
    """
    if not articles:
        return 0
    ingested_at = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        rows = [
            (
                art.get("id", str(uuid.uuid4())),
                art.get("ticker", ""),
                art.get("title", ""),
                art.get("source", ""),
                art.get("author", ""),
                art.get("publishedAt", ""),
                art.get("content", ""),
                art.get("sentiment_score", 0.0),
                ingested_at,
            )
            for art in articles
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO news
              (article_id, ticker, title, source, author,
               published_at, content, sentiment_score, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        logger.info("Batch saved %d articles to DB", len(rows))
        return len(rows)
    finally:
        conn.close()


def get_articles(
    db_path: str,
    ticker: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """
    Fetch articles from the `news` table, optionally filtered by ticker.

    Parameters
    ----------
    db_path : SQLite database path
    ticker  : Optional ticker filter
    limit   : Maximum rows to return

    Returns
    -------
    List of plain dicts (usable in Pandas/Streamlit directly)
    """
    conn = get_connection(db_path)
    try:
        sql = "SELECT * FROM news"
        params: tuple = (limit,)
        if ticker:
            sql += " WHERE ticker = ?"
            params = (ticker, limit)
        sql += " ORDER BY published_at DESC LIMIT ?"
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─── Prices CRUD ──────────────────────────────────────────────────────────────

def save_prices_batch(db_path: str, records: list[dict]) -> int:
    """
    Batch-upsert OHLCV + technical indicator rows into `prices` table.

    Parameters
    ----------
    db_path  : SQLite database path
    records  : List of dicts with keys:
               ticker, date, open, high, low, close, volume,
               rsi, ma20, vol_chg_pct

    Returns
    -------
    Number of rows upserted
    """
    if not records:
        return 0
    conn = get_connection(db_path)
    try:
        rows = [
            (
                str(uuid.uuid4()),
                r["ticker"],
                r["date"],
                r.get("open"),
                r.get("high"),
                r.get("low"),
                r["close"],
                r.get("volume"),
                r.get("rsi"),
                r.get("ma20"),
                r.get("vol_chg_pct"),
            )
            for r in records
        ]
        conn.executemany(
            """
            INSERT OR REPLACE INTO prices
              (price_id, ticker, date, open, high, low, close,
               volume, rsi, ma20, vol_chg_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        logger.info("Batch saved %d price rows for %s", len(rows),
                    records[0]["ticker"] if records else "?")
        return len(rows)
    finally:
        conn.close()


def get_prices(
    db_path: str,
    ticker: str,
    limit: int = 130,
) -> list[dict]:
    """
    Fetch OHLCV + indicator rows for a given ticker.

    Parameters
    ----------
    db_path : SQLite database path
    ticker  : Ticker symbol
    limit   : Maximum rows (default 130 ≈ 6 months of trading days)

    Returns
    -------
    List of dicts ordered oldest → newest
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM prices WHERE ticker = ? ORDER BY date ASC LIMIT ?",
            (ticker, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─── Signals CRUD ─────────────────────────────────────────────────────────────

def save_signal(
    db_path: str,
    *,
    ticker: str,
    signal_strength: int,
    direction: str,
    confidence_score: float,
    reasoning: str,
    news_citations: Optional[list] = None,
    rsi: Optional[float] = None,
    ma20: Optional[float] = None,
    vol_change_pct: Optional[float] = None,
    generated_at: Optional[str] = None,
    model_version: str = "gemini-1.5-pro",
    # Legacy compat kwargs (silently accepted)
    signal_type: Optional[str] = None,
    score: Optional[float] = None,
    rationale: Optional[str] = None,
    key_risks: Optional[str] = None,
    sources: Optional[list] = None,
) -> str:
    """
    Persist an AlphaSignal to the `signals` table.

    Parameters
    ----------
    ticker           : Stock ticker
    signal_strength  : Integer 1–5 (1=Strong Sell … 5=Strong Buy)
    direction        : Human label e.g. "Strong Buy"
    confidence_score : Float 0.0–1.0
    reasoning        : Full LLM rationale text
    news_citations   : List of cited article titles
    rsi              : Latest RSI value (optional)
    ma20             : Latest MA20 value (optional)
    vol_change_pct   : Latest volume change % (optional)
    generated_at     : ISO-8601 UTC timestamp (auto-generated if None)
    model_version    : LLM model identifier

    Returns
    -------
    Generated UUID signal_id string
    """
    signal_id = str(uuid.uuid4())
    ts = generated_at or datetime.now(timezone.utc).isoformat()
    citations_json = json.dumps(news_citations or [])

    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO signals
              (signal_id, ticker, signal_strength, direction, confidence_score,
               reasoning, news_citations, rsi, ma20, vol_change_pct,
               generated_at, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (signal_id, ticker, signal_strength, direction, confidence_score,
             reasoning, citations_json, rsi, ma20, vol_change_pct, ts, model_version),
        )
        conn.commit()
        logger.info(
            "Saved signal %s | %s | %s | strength=%d | confidence=%.2f",
            signal_id[:8], ticker, direction, signal_strength, confidence_score,
        )
    finally:
        conn.close()

    return signal_id


def get_signals(
    db_path: str,
    ticker: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """
    Fetch alpha signals from the `signals` table.

    Parameters
    ----------
    db_path : SQLite database path
    ticker  : Optional ticker filter
    limit   : Maximum rows to return

    Returns
    -------
    List of plain dicts with `news_citations` already deserialised to a list
    """
    conn = get_connection(db_path)
    try:
        sql = "SELECT * FROM signals"
        params: tuple = (limit,)
        if ticker:
            sql += " WHERE ticker = ?"
            params = (ticker, limit)
        sql += " ORDER BY generated_at DESC LIMIT ?"
        rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            try:
                d["news_citations"] = json.loads(d.get("news_citations") or "[]")
            except (json.JSONDecodeError, TypeError):
                d["news_citations"] = []
            results.append(d)
        return results
    finally:
        conn.close()


# ─── Legacy helpers (kept for app.py + test backward compat) ──────────────────

def log_ingestion(
    db_path: str,
    *,
    ticker: str,
    article_id: str,
    article_title: str,
    source: str = "",
    published_at: str = "",
    chunk_count: int = 0,
) -> None:
    """Write to the legacy `news_ingestion_log` table."""
    log_id = str(uuid.uuid4())
    ingested_at = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO news_ingestion_log
              (log_id, ticker, article_id, article_title, source,
               published_at, ingested_at, chunk_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (log_id, ticker, article_id, article_title, source,
             published_at, ingested_at, chunk_count),
        )
        conn.commit()
    finally:
        conn.close()


def get_news_log(db_path: str, ticker: Optional[str] = None, limit: int = 100) -> list[dict]:
    """Fetch the legacy news ingestion audit log."""
    conn = get_connection(db_path)
    try:
        if ticker:
            rows = conn.execute(
                "SELECT * FROM news_ingestion_log WHERE ticker = ? ORDER BY ingested_at DESC LIMIT ?",
                (ticker, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM news_ingestion_log ORDER BY ingested_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
