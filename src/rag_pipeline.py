"""
src/rag_pipeline.py
LangChain LCEL RAG chain using Gemini 1.5 Pro and ChromaDB.

The chain positions Gemini as a quantitative analyst and asks for a
structured alpha signal with one of five ratings:
  Strong Buy / Buy / Neutral / Sell / Strong Sell

Key functions
-------------
build_quant_chain(vectorstore, api_key, model) — Build the LCEL RAG chain
query_alpha_signal(chain_tuple, ticker, tech)  — Query with structured prompt
generate_alpha_query(ticker, tech_summary)     — Build the analyst question
generate_mock_signal(ticker, tech_summary)     — Offline fallback response
"""

import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

# ─── Model Configuration ──────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-1.5-pro"
RETRIEVER_K = 5   # top-k chunks retrieved per query


# ─── Quant Analyst Prompt ─────────────────────────────────────────────────────

QUANT_ANALYST_PROMPT = """\
You are a senior quantitative analyst at a top-tier hedge fund. Your job is to \
generate precise, evidence-based alpha signals from financial news and market data.

Below is relevant context from recent financial news articles:

{context}

---

Based on the news above AND the following price/technical data, answer the question \
with rigor, citing specific facts from the articles where possible.

{question}

Guidelines:
- Lead with the signal rating on its own line
- Cite specific article headlines, data points, or quotes as evidence
- Distinguish between short-term catalysts (< 30 days) and structural trends
- Acknowledge conflicting signals if present
- Keep reasoning to 3–5 concise sentences

Answer:"""

ALPHA_PROMPT = ChatPromptTemplate.from_template(QUANT_ANALYST_PROMPT)


# ─── Helper: format retrieved docs ────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    """
    Concatenate retrieved document chunks into a single context block.

    Each chunk is separated by a rule and prefixed with its source metadata.

    Parameters
    ----------
    docs : List of retrieved LangChain Documents

    Returns
    -------
    Formatted string for injection into the prompt context field
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = (
            f"[Article {i}] "
            f"Ticker: {meta.get('ticker', '?')} | "
            f"Source: {meta.get('source', '?')} | "
            f"Date: {meta.get('published_at', '?')} | "
            f"Sentiment: {meta.get('sentiment_score', 0.0):+.2f}"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ─── Chain Builder ────────────────────────────────────────────────────────────

def build_quant_chain(
    vectorstore,
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    k: int = RETRIEVER_K,
) -> tuple:
    """
    Build a LangChain LCEL RAG chain using Gemini 1.5 Pro.

    Architecture
    ------------
    ChromaDB retriever (top-5) → context formatter →
    ChatPromptTemplate → Gemini 1.5 Pro → StrOutputParser

    The chain is returned as a tuple ``(answer_chain, retriever)`` so callers
    can run the chain *and* fetch source documents independently.

    Parameters
    ----------
    vectorstore : Chroma vector store (from ``embeddings.py``)
    api_key     : Google Gemini API key
    model       : Gemini model name (default: ``gemini-1.5-pro``)
    temperature : Low temperature for factual, reproducible outputs
    k           : Number of chunks retrieved per query (default 5)

    Returns
    -------
    ``(answer_chain, retriever)`` tuple:
        - ``answer_chain`` : LCEL Runnable, call with ``{"question": str}``
        - ``retriever``    : ChromaDB retriever for fetching source docs
    """
    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # LCEL chain: retrieve context + pass question → prompt → LLM → parse
    answer_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | ALPHA_PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info("Built quant RAG chain: model=%s, k=%d, temp=%.1f", model, k, temperature)
    return answer_chain, retriever


# ─── Alpha Signal Query ───────────────────────────────────────────────────────

def generate_alpha_query(ticker: str, tech_summary: str) -> str:
    """
    Compose the structured quant analyst question for the RAG chain.

    The question explicitly asks for the 5-level rating scale and
    requires citation-backed reasoning.

    Parameters
    ----------
    ticker       : Stock ticker symbol (e.g. "AAPL")
    tech_summary : String output from ``summarise_technicals()``

    Returns
    -------
    Formatted question string to pass as the chain's ``question`` input
    """
    return (
        f"Based on recent news and price action, what is the alpha signal for {ticker}?\n\n"
        f"Technical data:\n{tech_summary}\n\n"
        f"Rate the signal EXACTLY as one of:\n"
        f"  Strong Buy / Buy / Neutral / Sell / Strong Sell\n\n"
        f"Structure your answer as:\n"
        f"RATING: [one of the five options above]\n"
        f"CONFIDENCE: [0-100]\n"
        f"REASONING: [3-5 sentences with specific article citations]\n"
        f"CATALYSTS: [1-2 near-term catalysts identified in the news]\n"
        f"RISKS: [1-2 key risks]\n"
        f"CITATIONS: [comma-separated article headlines used as evidence]"
    )


def query_alpha_signal(
    chain_tuple: tuple,
    ticker: str,
    tech_summary: str,
) -> dict:
    """
    Execute the RAG chain and return a structured result dict.

    Parameters
    ----------
    chain_tuple  : ``(answer_chain, retriever)`` from ``build_quant_chain``
    ticker       : Stock ticker
    tech_summary : Technical indicator summary string

    Returns
    -------
    Dict with keys:
        - ``answer``   : Full LLM response text
        - ``sources``  : List of retrieved source Document objects
        - ``question`` : The question that was asked
    """
    answer_chain, retriever = chain_tuple
    question = generate_alpha_query(ticker, tech_summary)

    try:
        answer = answer_chain.invoke(question)
        sources = retriever.invoke(question)
        logger.info("RAG query for %s: %d chars, %d sources", ticker, len(answer), len(sources))
        return {"answer": answer, "sources": sources, "question": question}
    except Exception as exc:
        logger.error("RAG chain query failed for %s: %s", ticker, exc)
        return {
            "answer": f"RATING: Neutral\nCONFIDENCE: 50\nREASONING: Error generating response: {exc}\n"
                      f"CATALYSTS: N/A\nRISKS: N/A\nCITATIONS:",
            "sources": [],
            "question": question,
        }


# ─── Legacy alias (kept for app.py backward compat) ──────────────────────────

def build_rag_chain(vectorstore, api_key: str, model_name: str = DEFAULT_MODEL, **kwargs):
    """Alias for build_quant_chain — retained for backward compatibility."""
    return build_quant_chain(vectorstore, api_key, model=model_name, **kwargs)


def query_rag(chain_tuple: tuple, question: str) -> dict:
    """
    Generic RAG query (used by the app.py Q&A tab).

    Parameters
    ----------
    chain_tuple : ``(answer_chain, retriever)`` from ``build_quant_chain``
    question    : Free-form natural language question

    Returns
    -------
    Dict with keys: answer, sources, question
    """
    answer_chain, retriever = chain_tuple
    try:
        answer = answer_chain.invoke(question)
        sources = retriever.invoke(question)
        return {"answer": answer, "sources": sources, "question": question}
    except Exception as exc:
        logger.error("RAG Q&A query failed: %s", exc)
        return {
            "answer": f"Error: {exc}",
            "sources": [],
            "question": question,
        }


# ─── Offline Mock Signals ─────────────────────────────────────────────────────

# Pre-built mock signals for demo mode (no API key required)
_MOCK_SIGNALS: dict[str, tuple] = {
    "AAPL":  ("Strong Buy",  88, "Apple Intelligence upgrade supercycle driving record services margin. "
                                "Strong developer adoption of Swift AI APIs signals multi-quarter tailwind. "
                                "iPhone 16 Pro sell-through tracking 12% ahead of iPhone 15.",
                                "Holiday upgrade cycle, Vision Pro price cut rumors",
                                "China market share erosion from Huawei; FX headwinds on international ASP",
                                "Apple Q1 Results Beat, iPhone 16 Pro Demand Exceeds Expectations"),
    "GOOGL": ("Neutral",     65, "Search market share intact at 89.5% despite DOJ antitrust case creating "
                                "binary regulatory risk. Gemini Ultra monetisation still early-stage. "
                                "Conflicting signals: AI ad revenue up but cloud margins compressed.",
                                "Gemini 2.0 Flash enterprise rollout",
                                "DOJ ruling requiring Chrome/Search divestiture — 20% EPS downside risk",
                                "Google Maintains Search Dominance Despite AI Competition"),
    "TSLA":  ("Sell",        72, "Cybertruck recall #3 within 8 months denting brand equity. "
                                "Q4 delivery miss of 6% vs consensus on Model Y inventory build. "
                                "FSD v13 NHTSA scrutiny adding regulatory overhang.",
                                "Sub-$30K Model 2 reveal (Q3 catalyst)",
                                "Any FSD approval or robotaxi launch would sharply reverse bearish signal",
                                "Tesla Reports Third Cybertruck Recall, Q4 Deliveries Miss"),
    "MSFT":  ("Strong Buy",  91, "Azure AI Copilot at $10B ARR run rate growing 157% YoY. "
                                "GitHub Copilot Enterprise 72% margin product at 2M enterprise seats. "
                                "OpenAI exclusivity through 2030 creates durable AI platform moat.",
                                "Copilot+ PC refresh cycle (45M units target)",
                                "Regulatory action on OpenAI investment; EU AI Act compliance costs",
                                "Microsoft Azure AI Revenue Surpasses $10 Billion Run Rate"),
    "NVDA":  ("Strong Buy",  96, "Blackwell GPU backlog of $80B+ extends visibility through Q2 2026. "
                                "Sovereign AI demand from UAE, India, Japan adding 15% incremental revenue. "
                                "Gross margin expansion to 78.4% vs consensus 76.1% on software attach.",
                                "Blackwell Ultra GB300 production ramp (Q2 2025)",
                                "China export control expansion; customer concentration (4 cos = 46% rev)",
                                "NVIDIA Blackwell Revenue Exceeds $8B in First Quarter"),
}


def generate_mock_signal(ticker: str, tech_summary: str) -> dict:
    """
    Return a pre-built mock alpha signal response for offline / demo mode.

    Used when no GEMINI_API_KEY is configured. Each mock signal is
    constructed to match the exact structured output format the real chain
    produces, so the parser in ``alpha_engine.py`` works identically.

    Parameters
    ----------
    ticker       : Stock ticker symbol
    tech_summary : Technical data string (included for completeness)

    Returns
    -------
    Dict with keys: answer (str), sources (list), question (str)
    """
    data = _MOCK_SIGNALS.get(
        ticker.upper(),
        ("Neutral", 50,
         f"Insufficient news data for {ticker}. Signal based on technical indicators only.",
         "Price breakout above MA20", "Low liquidity in sample dataset",
         "Sample financial news article"),
    )
    rating, confidence, reasoning, catalysts, risks, citation = data

    answer = (
        f"RATING: {rating}\n"
        f"CONFIDENCE: {confidence}\n"
        f"REASONING: {reasoning}\n"
        f"CATALYSTS: {catalysts}\n"
        f"RISKS: {risks}\n"
        f"CITATIONS: {citation}"
    )
    return {
        "answer": answer,
        "sources": [],
        "question": f"Alpha signal for {ticker}",
    }
