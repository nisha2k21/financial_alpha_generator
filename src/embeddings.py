"""
src/embeddings.py
Document chunking, TextBlob sentiment scoring, and ChromaDB vector storage.

Key functions
-------------
compute_sentiment(text)                    — TextBlob polarity score (-1 to +1)
chunk_documents(docs, chunk_size, overlap) — Split docs into 500-token chunks
embed_and_store(articles, ticker, ...)     — All-in-one: sentiment→chunk→embed
get_or_create_vectorstore(...)             — Load or create ChromaDB collection
embed_documents(docs, ticker, ...)         — Embed pre-formatted Documents
similarity_search(query, ticker, ...)      — Retrieve top-k relevant chunks
get_collection_count(ticker, ...)          — Count docs in a collection
"""

import logging
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ─── Chunking settings ────────────────────────────────────────────────────────

# 500 tokens ≈ 2000 characters (assuming ~4 chars/token for English text)
CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
CHARS_PER_TOKEN = 4

DEFAULT_CHUNK_SIZE = CHUNK_TOKENS * CHARS_PER_TOKEN        # 2000 chars
DEFAULT_CHUNK_OVERLAP = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN  # 200 chars

EMBEDDING_MODEL = "models/embedding-001"


# ─── Sentiment Scoring ────────────────────────────────────────────────────────

def compute_sentiment(text: str) -> float:
    """
    Compute a sentiment polarity score for a text string using TextBlob.

    TextBlob's ``sentiment.polarity`` is a float in [-1.0, +1.0]:
        -1.0 = maximally negative
         0.0 = neutral
        +1.0 = maximally positive

    Applied per chunk (not per article) for more granular sentiment signal.
    Falls back to 0.0 if TextBlob is unavailable or the text is empty.

    Parameters
    ----------
    text : Input text string

    Returns
    -------
    Float polarity score in range [-1.0, +1.0]
    """
    if not text or not text.strip():
        return 0.0
    try:
        from textblob import TextBlob
        return round(float(TextBlob(text).sentiment.polarity), 4)
    except Exception as exc:
        logger.debug("TextBlob sentiment failed: %s — returning 0.0", exc)
        return 0.0


# ─── Document Chunking ────────────────────────────────────────────────────────

def chunk_documents(
    docs: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split LangChain Documents into smaller chunks with sentiment scoring.

    Each resulting chunk gets:
    - All original metadata (ticker, date, source, article_id, etc.)
    - ``chunk_index`` : position of this chunk within the parent article
    - ``sentiment_score`` : TextBlob polarity for this specific chunk

    Chunk parameters
    ----------------
    chunk_size    : Target characters per chunk (default 2000 ≈ 500 tokens)
    chunk_overlap : Overlap between adjacent chunks (default 200 ≈ 50 tokens)

    Parameters
    ----------
    docs          : List of LangChain Document objects
    chunk_size    : Max characters per chunk
    chunk_overlap : Characters shared between consecutive chunks

    Returns
    -------
    List of chunked Document objects with updated metadata
    """
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: list[Document] = []
    for doc in docs:
        raw_chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(raw_chunks):
            # Add chunk-level metadata
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(raw_chunks)
            # Override article-level sentiment with per-chunk sentiment
            chunk.metadata["sentiment_score"] = compute_sentiment(chunk.page_content)
            all_chunks.append(chunk)

    logger.info(
        "Chunked %d docs → %d chunks (size≈%d tokens, overlap≈%d tokens)",
        len(docs), len(all_chunks), chunk_size // CHARS_PER_TOKEN,
        chunk_overlap // CHARS_PER_TOKEN,
    )
    return all_chunks


# ─── ChromaDB Helpers ─────────────────────────────────────────────────────────

def _make_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    """Instantiate the Google Generative AI embeddings model."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )


def get_or_create_vectorstore(
    collection_name: str,
    persist_dir: str,
    api_key: str,
) -> Chroma:
    """
    Load an existing ChromaDB collection or create a new empty one.

    Parameters
    ----------
    collection_name : ChromaDB collection identifier (e.g. ``"aapl_news"``)
    persist_dir     : Filesystem path for ChromaDB persistence
    api_key         : Google Gemini API key for embedding model

    Returns
    -------
    Chroma vector store object
    """
    embeddings = _make_embeddings(api_key)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    logger.debug("Loaded vectorstore: collection=%s dir=%s", collection_name, persist_dir)
    return vectorstore


def embed_documents(
    docs: list[Document],
    ticker: str,
    persist_dir: str,
    api_key: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[Chroma, int]:
    """
    Chunk, score sentiment, and embed a list of Documents into ChromaDB.

    Parameters
    ----------
    docs          : Pre-formatted LangChain Documents (from ``format_articles_for_rag``)
    ticker        : Stock ticker — used as ChromaDB collection namespace
    persist_dir   : ChromaDB persistence directory path
    api_key       : Google Gemini API key
    chunk_size    : Characters per chunk (~500 tokens default)
    chunk_overlap : Characters of overlap between chunks (~50 tokens)

    Returns
    -------
    (vectorstore, n_chunks)  — The Chroma store and count of embedded chunks
    """
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    if not chunks:
        logger.warning("No chunks to embed for %s", ticker)
        vs = get_or_create_vectorstore(
            collection_name=f"{ticker.lower()}_news",
            persist_dir=persist_dir,
            api_key=api_key,
        )
        return vs, 0

    embeddings = _make_embeddings(api_key)
    collection_name = f"{ticker.lower()}_news"

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    logger.info(
        "Embedded %d chunks for %s into collection '%s'",
        len(chunks), ticker, collection_name,
    )
    return vectorstore, len(chunks)


def embed_and_store(
    articles: list[dict],
    ticker: str,
    persist_dir: str,
    api_key: str,
) -> tuple[Chroma, int]:
    """
    All-in-one pipeline: raw articles → chunked → sentiment scored → embedded.

    This is the primary entry point for the embeddings layer when starting
    from raw article dicts (e.g. from ``fetch_news``).

    Steps
    -----
    1. Format articles as LangChain Documents
    2. Chunk each document into ~500-token pieces with 50-token overlap
    3. Score each chunk with TextBlob sentiment
    4. Embed and persist to ChromaDB

    Parameters
    ----------
    articles    : Raw article dicts from ``fetch_news``
    ticker      : Stock ticker symbol
    persist_dir : ChromaDB persistence directory
    api_key     : Google Gemini API key

    Returns
    -------
    (vectorstore, n_chunks)
    """
    from .ingestion import format_articles_for_rag

    docs = format_articles_for_rag(articles)
    return embed_documents(docs, ticker, persist_dir, api_key)


# ─── Similarity Search ────────────────────────────────────────────────────────

def similarity_search(
    query: str,
    ticker: str,
    persist_dir: str,
    api_key: str,
    k: int = 5,
) -> list[Document]:
    """
    Retrieve the top-k most semantically similar chunks for a query.

    Parameters
    ----------
    query       : Natural language search query
    ticker      : Ticker whose collection to search
    persist_dir : ChromaDB persistence directory
    api_key     : Google Gemini API key
    k           : Number of top results to return (default 5)

    Returns
    -------
    List of k most relevant Document chunks with metadata
    """
    vs = get_or_create_vectorstore(
        collection_name=f"{ticker.lower()}_news",
        persist_dir=persist_dir,
        api_key=api_key,
    )
    results = vs.similarity_search(query, k=k)
    logger.info("Similarity search for '%s…' returned %d chunks", query[:40], len(results))
    return results


# ─── Collection Metadata ──────────────────────────────────────────────────────

def get_collection_count(
    ticker: str,
    persist_dir: str,
    api_key: str,
) -> int:
    """
    Return the number of embedded chunks in a ticker's ChromaDB collection.

    Parameters
    ----------
    ticker      : Stock ticker symbol
    persist_dir : ChromaDB persistence directory
    api_key     : Google Gemini API key

    Returns
    -------
    Integer count of documents in the collection (0 if not found)
    """
    try:
        vs = get_or_create_vectorstore(
            collection_name=f"{ticker.lower()}_news",
            persist_dir=persist_dir,
            api_key=api_key,
        )
        count = vs._collection.count()
        logger.debug("Collection %s_news has %d chunks", ticker.lower(), count)
        return count
    except Exception as exc:
        logger.warning("Could not count collection for %s: %s", ticker, exc)
        return 0
