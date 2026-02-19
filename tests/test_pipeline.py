"""
tests/test_pipeline.py
Comprehensive unit tests for the enhanced RAG pipeline.

All tests run WITHOUT requiring API keys by using mocks and sample data.

Test classes
------------
TestSampleNewsLoading       — fetch_news fallback + article schema
TestDocumentChunking        — 500-token chunking + overlap
TestTechnicalIndicators     — RSI, MA20, vol_chg_pct computations
TestTextBlobSentiment       — compute_sentiment polarity range
TestSignalParser            — parse_signal_response (5-level scale)
TestAlphaSignalDataclass    — AlphaSignal field validation + strength mapping
TestTechnicalConfidence     — combine_technical_score adjustment rules
TestDatabase                — news, prices, signals tables (in-memory)
"""

import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

# ────────────────────────────────────────────────────────────────────────────
# Path helpers
# ────────────────────────────────────────────────────────────────────────────

SAMPLE_NEWS = Path(__file__).parent.parent / "data" / "sample_news.json"


# ════════════════════════════════════════════════════════════════════════════
# TestSampleNewsLoading
# ════════════════════════════════════════════════════════════════════════════

class TestSampleNewsLoading:
    """Verify sample news fallback and article schema."""

    def test_load_all_sample_news(self):
        from src.ingestion import load_sample_news
        articles = load_sample_news()
        assert len(articles) > 0, "sample_news.json should have at least one article"

    def test_load_sample_news_by_ticker(self):
        from src.ingestion import load_sample_news
        articles = load_sample_news("AAPL")
        assert any(a.get("ticker") == "AAPL" for a in articles)

    def test_load_sample_news_unknown_ticker_fallback(self):
        """Unknown ticker should return all articles rather than empty list."""
        from src.ingestion import load_sample_news
        articles = load_sample_news("UNKNOWN_ZZZ")
        assert len(articles) > 0, "Should fall back to all articles for unknown ticker"

    def test_article_schema(self):
        from src.ingestion import load_sample_news
        required = {"id", "ticker", "title", "source", "publishedAt", "content"}
        for art in load_sample_news():
            assert required.issubset(art.keys()), f"Missing keys in article: {art.keys()}"

    def test_format_articles_for_rag(self):
        from src.ingestion import load_sample_news, format_articles_for_rag
        from langchain_core.documents import Document
        docs = format_articles_for_rag(load_sample_news("AAPL"))
        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)
        assert all("ticker" in d.metadata for d in docs)
        assert all("published_at" in d.metadata for d in docs)
        assert all("sentiment_score" in d.metadata for d in docs)


# ════════════════════════════════════════════════════════════════════════════
# TestDocumentChunking
# ════════════════════════════════════════════════════════════════════════════

class TestDocumentChunking:
    """500-token (2000-char) chunks with 50-token (200-char) overlap."""

    @pytest.fixture
    def sample_docs(self):
        from src.ingestion import load_sample_news, format_articles_for_rag
        return format_articles_for_rag(load_sample_news())

    def test_chunk_count_reasonable(self, sample_docs):
        from src.embeddings import chunk_documents
        chunks = chunk_documents(sample_docs)
        # Should produce more chunks than docs but not explode
        assert len(chunks) >= len(sample_docs)
        assert len(chunks) < len(sample_docs) * 20

    def test_chunk_metadata_inherited(self, sample_docs):
        from src.embeddings import chunk_documents
        chunks = chunk_documents(sample_docs)
        for chunk in chunks:
            assert "ticker" in chunk.metadata
            assert "source" in chunk.metadata

    def test_chunk_size_respected(self, sample_docs):
        from src.embeddings import chunk_documents, DEFAULT_CHUNK_SIZE
        chunks = chunk_documents(sample_docs, chunk_size=DEFAULT_CHUNK_SIZE)
        oversized = [c for c in chunks if len(c.page_content) > DEFAULT_CHUNK_SIZE * 1.2]
        # Allow small tolerance for splitter behaviour
        assert len(oversized) == 0, f"{len(oversized)} chunks exceed max size"

    def test_no_empty_chunks(self, sample_docs):
        from src.embeddings import chunk_documents
        chunks = chunk_documents(sample_docs)
        assert all(len(c.page_content.strip()) > 0 for c in chunks)

    def test_chunk_index_metadata(self, sample_docs):
        """Each chunk should have chunk_index in metadata."""
        from src.embeddings import chunk_documents
        chunks = chunk_documents(sample_docs)
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["chunk_index"] >= 0

    def test_sentiment_score_attached_to_chunk(self, sample_docs):
        """Every chunk should have a sentiment_score in [-1, 1]."""
        from src.embeddings import chunk_documents
        chunks = chunk_documents(sample_docs)
        for chunk in chunks:
            score = chunk.metadata.get("sentiment_score")
            assert score is not None
            assert -1.0 <= score <= 1.0, f"sentiment_score out of range: {score}"


# ════════════════════════════════════════════════════════════════════════════
# TestTechnicalIndicators
# ════════════════════════════════════════════════════════════════════════════

class TestTechnicalIndicators:
    """RSI, MA20, and vol_chg_pct computations."""

    @pytest.fixture
    def price_df(self):
        """Synthetic 60-row OHLCV DataFrame for indicator tests."""
        n = 60
        # Use pd.date_range with 'D' (calendar days) to guarantee no NaN index issues
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        close = pd.Series([150.0 + i * 0.5 + (i % 3) * 0.25 for i in range(n)], index=dates)
        volume = pd.Series([1_000_000 + 50_000 * (i % 10) for i in range(n)], index=dates)
        df = pd.DataFrame({
            "Open":   close - 0.5,
            "High":   close + 1.0,
            "Low":    close - 1.0,
            "Close":  close,
            "Volume": volume,
            "ticker": "TEST",
        }, index=dates)
        df.index.name = "Date"
        return df

    def test_rsi_column_exists(self, price_df):
        from src.ingestion import compute_technical_indicators
        result = compute_technical_indicators(price_df)
        assert "RSI" in result.columns

    def test_rsi_range(self, price_df):
        """RSI should be in [0, 100] for all non-NaN rows."""
        from src.ingestion import compute_technical_indicators
        result = compute_technical_indicators(price_df)
        valid_rsi = result["RSI"].dropna()
        assert len(valid_rsi) > 0, "RSI should have non-NaN values"
        assert (valid_rsi >= 0).all(), "RSI should not be negative"
        assert (valid_rsi <= 100).all(), "RSI should not exceed 100"

    def test_ma20_column_exists(self, price_df):
        from src.ingestion import compute_technical_indicators
        result = compute_technical_indicators(price_df)
        assert "MA20" in result.columns

    def test_ma20_is_rolling_mean(self, price_df):
        """MA20 at row 25 should equal the mean of rows 5:25 (rolling 20-day window)."""
        from src.ingestion import compute_technical_indicators
        result = compute_technical_indicators(price_df)
        # MA20 at index 25 uses rows 6..25 (20-row rolling window ending at row 25)
        expected = price_df["Close"].iloc[6:26].mean()
        actual = result["MA20"].iloc[25]
        assert not pd.isna(actual), "MA20 should not be NaN at row 25"
        assert abs(actual - expected) < 0.5, f"MA20 mismatch: {actual} vs {expected}"

    def test_vol_chg_pct_column_exists(self, price_df):
        from src.ingestion import compute_technical_indicators
        result = compute_technical_indicators(price_df)
        assert "vol_chg_pct" in result.columns

    def test_vol_chg_pct_positive_on_surge(self, price_df):
        """Setting last row volume to 3x the rolling average gives positive vol_chg_pct."""
        from src.ingestion import compute_technical_indicators
        # First compute the 20-day rolling average on original data
        vol_ma = price_df["Volume"].rolling(20, min_periods=1).mean()
        surge_vol = int(vol_ma.iloc[-1] * 3)
        modified = price_df.copy()
        modified.iloc[-1, modified.columns.get_loc("Volume")] = surge_vol
        result = compute_technical_indicators(modified)
        last_pct = result["vol_chg_pct"].iloc[-1]
        assert not pd.isna(last_pct), "vol_chg_pct should not be NaN"
        assert last_pct > 0, f"Expected positive vol_chg_pct on surge, got {last_pct}"

    def test_empty_df_returns_empty(self):
        from src.ingestion import compute_technical_indicators
        result = compute_technical_indicators(pd.DataFrame())
        assert result.empty

    def test_summarise_technicals_contains_rsi(self, price_df):
        from src.ingestion import compute_technical_indicators, summarise_technicals
        df_with_indicators = compute_technical_indicators(price_df)
        summary = summarise_technicals(df_with_indicators, "TEST")
        assert "RSI" in summary
        assert "MA20" in summary

    def test_fetch_stock_data_default_6mo(self):
        """fetch_stock_data default period should return ~100+ rows."""
        from src.ingestion import fetch_stock_data
        df = fetch_stock_data("AAPL", period="6mo")
        if not df.empty:
            assert len(df) >= 80, "6-month AAPL data should have at least 80 trading days"
            assert "Close" in df.columns
            assert "Volume" in df.columns


# ════════════════════════════════════════════════════════════════════════════
# TestTextBlobSentiment
# ════════════════════════════════════════════════════════════════════════════

class TestTextBlobSentiment:
    """TextBlob polarity scoring."""

    def test_positive_text(self):
        from src.embeddings import compute_sentiment
        score = compute_sentiment("Excellent earnings beat! Stock surges to record high.")
        assert score > 0, f"Expected positive sentiment, got {score}"

    def test_negative_text(self):
        from src.embeddings import compute_sentiment
        score = compute_sentiment("Terrible losses. The company is failing and stock crashes.")
        assert score < 0, f"Expected negative sentiment, got {score}"

    def test_neutral_text(self):
        from src.embeddings import compute_sentiment
        score = compute_sentiment("The company reported quarterly results.")
        assert -0.5 <= score <= 0.5

    def test_empty_text_returns_zero(self):
        from src.embeddings import compute_sentiment
        assert compute_sentiment("") == 0.0
        assert compute_sentiment("   ") == 0.0

    def test_score_in_range(self):
        from src.embeddings import compute_sentiment
        texts = [
            "Apple beats estimates by 15%",
            "Tesla recall hits margins",
            "Federal Reserve holds rates steady",
            "nvidia gpu shortage drives record revenue",
        ]
        for text in texts:
            score = compute_sentiment(text)
            assert -1.0 <= score <= 1.0, f"Score {score} out of range for: {text}"


# ════════════════════════════════════════════════════════════════════════════
# TestSignalParser
# ════════════════════════════════════════════════════════════════════════════

class TestSignalParser:
    """parse_signal_response extracts all 5-level signal fields correctly."""

    def _make_response(self, rating, confidence, reasoning, catalysts, risks, citations):
        return (
            f"RATING: {rating}\n"
            f"CONFIDENCE: {confidence}\n"
            f"REASONING: {reasoning}\n"
            f"CATALYSTS: {catalysts}\n"
            f"RISKS: {risks}\n"
            f"CITATIONS: {citations}"
        )

    def test_parse_strong_buy(self):
        from src.alpha_engine import parse_signal_response
        resp = self._make_response("Strong Buy", 92, "Record AI chip demand.", "Blackwell ramp", "Export controls", "NVDA Q4 Results")
        parsed = parse_signal_response(resp)
        assert parsed["direction"] == "Strong Buy"
        assert parsed["signal_strength"] == 5

    def test_parse_buy(self):
        from src.alpha_engine import parse_signal_response
        resp = self._make_response("Buy", 75, "Positive catalysts.", "AI ad revenue", "Antitrust risk", "Google Q3 Beat")
        parsed = parse_signal_response(resp)
        assert parsed["direction"] == "Buy"
        assert parsed["signal_strength"] == 4

    def test_parse_neutral(self):
        from src.alpha_engine import parse_signal_response
        resp = self._make_response("Neutral", 55, "Mixed signals.", "None", "None", "Report")
        parsed = parse_signal_response(resp)
        assert parsed["signal_strength"] == 3

    def test_parse_sell(self):
        from src.alpha_engine import parse_signal_response
        resp = self._make_response("Sell", 65, "Delivery miss and recall.", "Model 2", "FSD approval", "Tesla Miss")
        parsed = parse_signal_response(resp)
        assert parsed["signal_strength"] == 2

    def test_parse_strong_sell(self):
        from src.alpha_engine import parse_signal_response
        resp = self._make_response("Strong Sell", 88, "Massive fraud allegations.", "None", "All risk", "Earnings Miss")
        parsed = parse_signal_response(resp)
        assert parsed["direction"] == "Strong Sell"
        assert parsed["signal_strength"] == 1

    def test_confidence_range(self):
        from src.alpha_engine import parse_signal_response
        resp = self._make_response("Buy", 80, "Test", "Cat", "Risk", "Citation")
        parsed = parse_signal_response(resp)
        assert 0.0 <= parsed["raw_confidence"] <= 1.0
        assert abs(parsed["raw_confidence"] - 0.80) < 0.01

    def test_citations_extracted_as_list(self):
        from src.alpha_engine import parse_signal_response
        resp = self._make_response("Buy", 75, "Good", "Cat", "Risk", "Article A, Article B, Article C")
        parsed = parse_signal_response(resp)
        assert isinstance(parsed["news_citations"], list)
        assert len(parsed["news_citations"]) == 3

    def test_unknown_rating_defaults_neutral(self):
        from src.alpha_engine import parse_signal_response
        resp = self._make_response("GARBAGE_RATING", 50, "Test", "Cat", "Risk", "Src")
        parsed = parse_signal_response(resp)
        assert parsed["signal_strength"] == 3
        assert parsed["direction"] == "Neutral"

    def test_direction_to_strength_mapping(self):
        from src.alpha_engine import direction_to_strength
        assert direction_to_strength("Strong Buy") == 5
        assert direction_to_strength("Buy") == 4
        assert direction_to_strength("Neutral") == 3
        assert direction_to_strength("Sell") == 2
        assert direction_to_strength("Strong Sell") == 1
        assert direction_to_strength("UNKNOWN") == 3  # default


# ════════════════════════════════════════════════════════════════════════════
# TestAlphaSignalDataclass
# ════════════════════════════════════════════════════════════════════════════

class TestAlphaSignalDataclass:
    """AlphaSignal has all required fields and correct types."""

    def _make_signal(self, **overrides):
        from src.alpha_engine import AlphaSignal
        defaults = dict(
            ticker="AAPL",
            signal_strength=5,
            direction="Strong Buy",
            confidence_score=0.87,
            reasoning="AI iPhone supercycle driving revenue growth.",
            news_citations=["Apple Q4 Beat", "iPhone 16 Demand Surge"],
            generated_at="2024-02-19T10:00:00+00:00",
            ticker_rsi=28.5,
            ticker_ma20=183.22,
            vol_change_pct=62.1,
            model_version="gemini-1.5-pro",
        )
        defaults.update(overrides)
        return AlphaSignal(**defaults)

    def test_all_required_fields_present(self):
        signal = self._make_signal()
        assert signal.ticker == "AAPL"
        assert signal.signal_strength == 5
        assert signal.direction == "Strong Buy"
        assert 0.0 <= signal.confidence_score <= 1.0
        assert isinstance(signal.reasoning, str)
        assert isinstance(signal.news_citations, list)
        assert isinstance(signal.generated_at, str)
        assert isinstance(signal.ticker_rsi, float)
        assert isinstance(signal.ticker_ma20, float)
        assert isinstance(signal.vol_change_pct, float)

    def test_signal_strength_bounds(self):
        for strength in range(1, 6):
            s = self._make_signal(signal_strength=strength)
            assert 1 <= s.signal_strength <= 5

    def test_news_citations_is_list(self):
        s = self._make_signal()
        assert isinstance(s.news_citations, list)

    def test_nan_technical_fields_allowed(self):
        """Technical fields may be NaN when price data is unavailable."""
        s = self._make_signal(ticker_rsi=float("nan"))
        assert math.isnan(s.ticker_rsi)


# ════════════════════════════════════════════════════════════════════════════
# TestTechnicalConfidence
# ════════════════════════════════════════════════════════════════════════════

class TestTechnicalConfidence:
    """combine_technical_score adjustment rules."""

    def test_oversold_rsi_with_buy_increases_confidence(self):
        from src.alpha_engine import combine_technical_score
        base = 0.70
        result = combine_technical_score(rsi=25.0, vol_chg_pct=0.0, direction="Buy", raw_confidence=base)
        assert result > base, f"Oversold RSI + BUY should increase confidence (got {result})"

    def test_overbought_rsi_with_sell_increases_confidence(self):
        from src.alpha_engine import combine_technical_score
        base = 0.70
        result = combine_technical_score(rsi=78.0, vol_chg_pct=0.0, direction="Sell", raw_confidence=base)
        assert result > base

    def test_conflicting_signal_penalises_confidence(self):
        from src.alpha_engine import combine_technical_score
        base = 0.70
        # Oversold RSI but SELL direction — conflicting
        result = combine_technical_score(rsi=25.0, vol_chg_pct=0.0, direction="Sell", raw_confidence=base)
        assert result < base

    def test_high_volume_increases_confidence(self):
        from src.alpha_engine import combine_technical_score
        base = 0.60
        result = combine_technical_score(rsi=50.0, vol_chg_pct=80.0, direction="Buy", raw_confidence=base)
        assert result > base

    def test_confidence_clamped_to_0_1(self):
        from src.alpha_engine import combine_technical_score
        result = combine_technical_score(rsi=20.0, vol_chg_pct=200.0, direction="Strong Buy", raw_confidence=0.95)
        assert 0.0 <= result <= 1.0

    def test_none_rsi_does_not_crash(self):
        from src.alpha_engine import combine_technical_score
        result = combine_technical_score(rsi=None, vol_chg_pct=None, direction="Neutral", raw_confidence=0.5)
        assert result == 0.5


# ════════════════════════════════════════════════════════════════════════════
# TestDatabase
# ════════════════════════════════════════════════════════════════════════════

class TestDatabase:
    """All three new tables: news, prices, signals."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        from src.database import init_db
        db_path = str(tmp_path / "test_alpha.db")
        init_db(db_path)
        return db_path

    # ── Schema ───────────────────────────────────────────────────────────────

    def test_all_new_tables_created(self, temp_db):
        conn = sqlite3.connect(temp_db)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert "news" in tables
        assert "prices" in tables
        assert "signals" in tables

    def test_legacy_tables_preserved(self, temp_db):
        conn = sqlite3.connect(temp_db)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert "alpha_signals" in tables
        assert "news_ingestion_log" in tables

    # ── news table ────────────────────────────────────────────────────────────

    def test_save_and_retrieve_article(self, temp_db):
        from src.database import save_article, get_articles
        save_article(
            temp_db,
            article_id="art001",
            ticker="AAPL",
            title="Apple Q4 Results",
            source="Reuters",
            sentiment_score=0.45,
        )
        articles = get_articles(temp_db, ticker="AAPL")
        assert len(articles) == 1
        assert articles[0]["title"] == "Apple Q4 Results"
        assert articles[0]["sentiment_score"] == pytest.approx(0.45, abs=0.001)

    def test_batch_save_articles(self, temp_db):
        from src.database import save_articles_batch, get_articles
        batch = [
            {"id": f"art{i:03d}", "ticker": "NVDA", "title": f"NVDA News {i}",
             "source": "Bloomberg", "author": "", "publishedAt": "2024-01-01",
             "content": "Some news.", "sentiment_score": 0.2}
            for i in range(5)
        ]
        count = save_articles_batch(temp_db, batch)
        assert count == 5
        articles = get_articles(temp_db, ticker="NVDA")
        assert len(articles) == 5

    # ── prices table ──────────────────────────────────────────────────────────

    def test_save_and_retrieve_prices(self, temp_db):
        from src.database import save_prices_batch, get_prices
        records = [
            {"ticker": "MSFT", "date": f"2024-01-{i+1:02d}",
             "open": 400.0, "high": 405.0, "low": 398.0, "close": 402.0 + i,
             "volume": 25_000_000, "rsi": 55.0, "ma20": 398.0, "vol_chg_pct": 5.0}
            for i in range(10)
        ]
        count = save_prices_batch(temp_db, records)
        assert count == 10
        prices = get_prices(temp_db, "MSFT")
        assert len(prices) == 10
        assert "rsi" in prices[0]
        assert "ma20" in prices[0]
        assert "vol_chg_pct" in prices[0]

    # ── signals table ─────────────────────────────────────────────────────────

    def test_save_and_retrieve_signal(self, temp_db):
        from src.database import save_signal, get_signals
        signal_id = save_signal(
            temp_db,
            ticker="NVDA",
            signal_strength=5,
            direction="Strong Buy",
            confidence_score=0.92,
            reasoning="Blackwell demand surge confirmed by multiple catalysts.",
            news_citations=["NVDA Q4 Record Revenue", "Blackwell Backlog $80B"],
            rsi=28.5,
            ma20=780.0,
            vol_change_pct=65.0,
        )
        assert signal_id  # non-empty UUID
        signals = get_signals(temp_db, ticker="NVDA")
        assert len(signals) == 1
        s = signals[0]
        assert s["signal_strength"] == 5
        assert s["direction"] == "Strong Buy"
        assert s["confidence_score"] == pytest.approx(0.92, abs=0.001)
        assert isinstance(s["news_citations"], list)
        assert len(s["news_citations"]) == 2

    def test_signal_strength_range(self, temp_db):
        """signal_strength outside 1-5 should raise IntegrityError."""
        from src.database import get_connection
        conn = get_connection(temp_db)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO signals (signal_id, ticker, signal_strength, direction, "
                "confidence_score, reasoning, generated_at) VALUES (?,?,?,?,?,?,?)",
                ("bad_id", "AAPL", 6, "Invalid", 0.5, "Test", "2024-01-01"),
            )
            conn.commit()

    def test_get_signals_filter_by_ticker(self, temp_db):
        from src.database import save_signal, get_signals
        for ticker, strength in [("AAPL", 4), ("TSLA", 2), ("AAPL", 5)]:
            save_signal(
                temp_db, ticker=ticker, signal_strength=strength,
                direction="Buy", confidence_score=0.7, reasoning="Test signal",
                generated_at="2024-01-01T00:00:00+00:00",
            )
        aapl = get_signals(temp_db, ticker="AAPL")
        tsla = get_signals(temp_db, ticker="TSLA")
        assert len(aapl) == 2
        assert len(tsla) == 1
