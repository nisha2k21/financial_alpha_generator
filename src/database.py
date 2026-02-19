"""
src/database.py
SQLite persistence layer with BigQuery-compatible schema.

Tables
------
news       — Raw news articles with sentiment scores
prices     — OHLCV + technical indicators per ticker per day
signals    — AlphaSignal objects (new schema with signal_strength 1–5)
paper_portfolio — Current cash and total value for virtual trading
paper_trades    — Historical and open virtual trades
alerts          — Price and technical indicators alerts
trade_journals  — Gemini-generated journal entries for closed trades
weekly_reviews  — Gemini-generated weekly performance summaries
sentiment_history — Historical social (Reddit) and news sentiment

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
    model_version    TEXT,
    -- Long/Short Engine fields (v2)
    position              TEXT,
    position_size_pct     REAL,
    entry_price          REAL,
    stop_loss_price      REAL,
    take_profit_price    REAL,
    risk_reward_ratio    REAL,
    trade_rationale      TEXT
);
"""

# ─── Schema: paper_portfolio ──────────────────────────────────────────────────

CREATE_PAPER_PORTFOLIO_TABLE = """
CREATE TABLE IF NOT EXISTS paper_portfolio (
    portfolio_id    TEXT    PRIMARY KEY,
    cash            REAL    DEFAULT 100000.0,
    total_value     REAL    DEFAULT 100000.0,
    last_updated    TEXT    NOT NULL
);
"""

# ─── Schema: paper_trades ─────────────────────────────────────────────────────

CREATE_PAPER_TRADES_TABLE = """
CREATE TABLE IF NOT EXISTS paper_trades (
    trade_id            TEXT    PRIMARY KEY,
    ticker              TEXT    NOT NULL,
    direction           TEXT    NOT NULL, -- LONG, SHORT
    entry_price         REAL    NOT NULL,
    exit_price          REAL,
    quantity            INTEGER NOT NULL,
    position_size_pct   REAL,
    stop_loss           REAL,
    take_profit         REAL,
    entry_date          TEXT    NOT NULL,
    exit_date           TEXT,
    realized_pnl        REAL,
    realized_pnl_pct    REAL,
    outcome             TEXT    DEFAULT 'OPEN', -- WIN, LOSS, OPEN
    exit_reason         TEXT, -- STOPPED_OUT, TARGET_HIT, MANUAL, SIGNAL_REVERSED
    ai_signal           TEXT,
    ai_confidence       REAL
);
"""

# ─── Schema: alerts ───────────────────────────────────────────────────────────

CREATE_ALERTS_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id            TEXT    PRIMARY KEY,
    ticker              TEXT    NOT NULL,
    alert_type          TEXT    NOT NULL, -- PRICE_ABOVE, RSI_OVERBOUGHT, etc.
    trigger_value       REAL    NOT NULL,
    message             TEXT,
    is_active           INTEGER DEFAULT 1,
    created_at          TEXT    NOT NULL,
    triggered_at        TEXT,
    times_triggered     INTEGER DEFAULT 0
);
"""

# ─── Schema: coaching & sentiment ─────────────────────────────────────────────

CREATE_JOURNALS_TABLE = """
CREATE TABLE IF NOT EXISTS trade_journals (
    journal_id    TEXT    PRIMARY KEY,
    trade_id      TEXT    NOT NULL,
    ticker        TEXT    NOT NULL,
    content       TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    FOREIGN KEY(trade_id) REFERENCES paper_trades(trade_id)
);
"""

CREATE_WEEKLY_REVIEWS_TABLE = """
CREATE TABLE IF NOT EXISTS weekly_reviews (
    review_id     TEXT    PRIMARY KEY,
    start_date    TEXT    NOT NULL,
    end_date      TEXT    NOT NULL,
    content       TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);
"""

CREATE_SENTIMENT_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS sentiment_history (
    sentiment_id  TEXT    PRIMARY KEY,
    ticker        TEXT    NOT NULL,
    source        TEXT    NOT NULL, -- REDDIT, NEWS, COMBINED
    score         REAL    NOT NULL,
    mention_count INTEGER DEFAULT 0,
    timestamp     TEXT    NOT NULL
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
    "CREATE INDEX IF NOT EXISTS idx_paper_trades_ticker ON paper_trades(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_paper_trades_status ON paper_trades(outcome);",
    "CREATE INDEX IF NOT EXISTS idx_alerts_ticker ON alerts(ticker);",
    "CREATE INDEX IF NOT EXISTS idx_sentiment_ticker_time ON sentiment_history(ticker, timestamp);",
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
            CREATE_PAPER_PORTFOLIO_TABLE,
            CREATE_PAPER_TRADES_TABLE,
            CREATE_ALERTS_TABLE,
            CREATE_JOURNALS_TABLE,
            CREATE_WEEKLY_REVIEWS_TABLE,
            CREATE_SENTIMENT_HISTORY_TABLE,
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


# ─── Paper Trading CRUD ───────────────────────────────────────────────────────

def get_paper_portfolio(db_path: str) -> dict:
    """Retrieve the paper trading portfolio state."""
    conn = get_connection(db_path)
    try:
        # Check if portfolio exists, if not initialize
        row = conn.execute("SELECT * FROM paper_portfolio LIMIT 1").fetchone()
        if row:
            return dict(row)
        
        # Initialize default portfolio
        initial_ts = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO paper_portfolio (portfolio_id, cash, total_value, last_updated) VALUES (?, ?, ?, ?)",
            ("default", 100000.0, 100000.0, initial_ts)
        )
        conn.commit()
        return {
            "portfolio_id": "default",
            "cash": 100000.0,
            "total_value": 100000.0,
            "last_updated": initial_ts
        }
    finally:
        conn.close()


def save_paper_portfolio(db_path: str, cash: float, total_value: float) -> None:
    """Upsert the paper trading portfolio state."""
    ts = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_portfolio
              (portfolio_id, cash, total_value, last_updated)
            VALUES (?, ?, ?, ?)
            """,
            ("default", cash, total_value, ts)
        )
        conn.commit()
    finally:
        conn.close()


def save_paper_trade(db_path: str, trade: dict) -> str:
    """Persist a new or updated paper trade."""
    trade_id = trade.get("trade_id") or str(uuid.uuid4())
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO paper_trades
              (trade_id, ticker, direction, entry_price, exit_price,
               quantity, position_size_pct, stop_loss, take_profit,
               entry_date, exit_date, realized_pnl, realized_pnl_pct,
               outcome, exit_reason, ai_signal, ai_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id, trade["ticker"], trade["direction"],
                trade["entry_price"], trade.get("exit_price"),
                trade["quantity"], trade.get("position_size_pct"),
                trade.get("stop_loss"), trade.get("take_profit"),
                trade["entry_date"], trade.get("exit_date"),
                trade.get("realized_pnl"), trade.get("realized_pnl_pct"),
                trade.get("outcome", "OPEN"), trade.get("exit_reason"),
                trade.get("ai_signal"), trade.get("ai_confidence")
            )
        )
        conn.commit()
    finally:
        conn.close()
    return trade_id


def get_paper_trades(db_path: str, outcome: Optional[str] = None) -> list[dict]:
    """Fetch paper trades, optionally filtered by outcome (OPEN/WIN/LOSS)."""
    conn = get_connection(db_path)
    try:
        sql = "SELECT * FROM paper_trades"
        params = []
        if outcome:
            sql += " WHERE outcome = ?"
            params.append(outcome)
        sql += " ORDER BY entry_date DESC"
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─── Alerts CRUD ──────────────────────────────────────────────────────────────

def save_alert(db_path: str, alert: dict) -> str:
    """Save a new alert or update an existing one."""
    alert_id = alert.get("alert_id") or str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO alerts
              (alert_id, ticker, alert_type, trigger_value, message,
               is_active, created_at, triggered_at, times_triggered)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert_id, alert["ticker"], alert["alert_type"],
                alert["trigger_value"], alert.get("message"),
                alert.get("is_active", 1), alert.get("created_at", ts),
                alert.get("triggered_at"), alert.get("times_triggered", 0)
            )
        )
        conn.commit()
    finally:
        conn.close()
    return alert_id


def get_alerts(db_path: str, ticker: Optional[str] = None, active_only: bool = False) -> list[dict]:
    """Fetch alerts, optionally filtered by ticker or active status."""
    conn = get_connection(db_path)
    try:
        sql = "SELECT * FROM alerts"
        clauses = []
        params = []
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker.upper())
        if active_only:
            clauses.append("is_active = 1")
        
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        
        sql += " ORDER BY created_at DESC"
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_alert(db_path: str, alert_id: str) -> None:
    """Permanently remove an alert."""
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM alerts WHERE alert_id = ?", (alert_id,))
        conn.commit()
    finally:
        conn.close()


# ─── Sentiment CRUD ───────────────────────────────────────────────────────────

def save_sentiment(db_path: str, ticker: str, source: str, score: float, count: int = 0) -> None:
    """Record a sentiment snapshot."""
    sentiment_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO sentiment_history
              (sentiment_id, ticker, source, score, mention_count, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (sentiment_id, ticker.upper(), source, score, count, ts)
        )
        conn.commit()
    finally:
        conn.close()


def get_sentiment_history(db_path: str, ticker: str, limit: int = 100) -> list[dict]:
    """Fetch historical sentiment scores for a ticker."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM sentiment_history WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?",
            (ticker.upper(), limit)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─── Coaching CRUD ────────────────────────────────────────────────────────────

def save_journal_entry(db_path: str, journal: dict) -> str:
    """Save a Gemini-generated journal entry."""
    journal_id = journal.get("journal_id") or str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO trade_journals
              (journal_id, trade_id, ticker, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (journal_id, journal["trade_id"], journal["ticker"], 
             journal["content"], journal.get("created_at", ts))
        )
        conn.commit()
    finally:
        conn.close()
    return journal_id


def get_journals(db_path: str, ticker: Optional[str] = None) -> list[dict]:
    """Fetch trade journal entries."""
    conn = get_connection(db_path)
    try:
        sql = "SELECT * FROM trade_journals"
        params = []
        if ticker:
            sql += " WHERE ticker = ?"
            params.append(ticker.upper())
        sql += " ORDER BY created_at DESC"
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_weekly_review(db_path: str, review: dict) -> str:
    """Save/update a weekly performance summary."""
    review_id = review.get("review_id") or str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO weekly_reviews
              (review_id, start_date, end_date, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (review_id, review["start_date"], review["end_date"], 
             review["content"], ts)
        )
        conn.commit()
    finally:
        conn.close()
    return review_id


def get_weekly_reviews(db_path: str, limit: int = 10) -> list[dict]:
    """Fetch the latest weekly reviews."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM weekly_reviews ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
