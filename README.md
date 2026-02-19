# 📈 AI Financial Alpha Generator

> **RAG-powered alpha signal engine** — Gemini 1.5 Pro · LangChain · ChromaDB · Streamlit

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini-1.5%20Pro-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat&logo=chainlink&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=flat)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🧠 What Is This?

Traditional alpha generation relies on hand-crafted rules and lagging indicators. This project
replaces that with a **Retrieval-Augmented Generation (RAG)** architecture: financial news is
ingested, embedded into a vector store, and queried by Gemini 1.5 Pro — which acts as a
**senior quantitative analyst** — to generate structured, evidence-based trading signals.

The result: scalable, explainable, citation-backed alpha signals that blend LLM reasoning
with quantitative technical confirmation (RSI, MA20, volume analysis).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     AI Financial Alpha Generator                            │
│                                                                             │
│  ┌──────────┐   ┌───────────┐   ┌────────────┐   ┌────────────────────┐   │
│  │ NewsAPI  │   │ yfinance  │   │  TextBlob  │   │  ta (RSI/MA20/Vol) │   │
│  └────┬─────┘   └─────┬─────┘   └─────┬──────┘   └────────┬───────────┘   │
│       │               │               │                    │               │
│       ▼               ▼               ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              INGESTION LAYER  (ingestion.py)                        │   │
│  │  Fetch news → format as LangChain Documents → compute indicators    │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │           EMBEDDINGS LAYER  (embeddings.py)                         │   │
│  │  Chunk: 500 tokens / 50-token overlap                               │   │
│  │  Sentiment: TextBlob polarity scored per chunk                      │   │
│  │  Embed: Google models/embedding-001 → ChromaDB                      │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              RAG PIPELINE  (rag_pipeline.py)                        │   │
│  │  Retriever: ChromaDB top-5 similarity search                        │   │
│  │  LLM: Gemini 1.5 Pro (temp=0.1, quant analyst system prompt)       │   │
│  │  Output: Strong Buy / Buy / Neutral / Sell / Strong Sell            │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │            ALPHA ENGINE  (alpha_engine.py)                          │   │
│  │  Parse response → AlphaSignal dataclass                             │   │
│  │  Adjust confidence: RSI + volume technical confirmation             │   │
│  │  Persist to SQLite (BigQuery-compatible schema)                     │   │
│  └──────────────────────────┬──────────────────────────────────────────┘   │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │          STREAMLIT DASHBOARD  (app.py)  — 4 Pages                   │   │
│  │  1. Alpha Dashboard  2. AI Research  3. History  4. How It Works    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/financial-alpha-generator.git
cd financial-alpha-generator
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # Download TextBlob English corpus
```

### 3. Configure API keys

```bash
cp .env.template .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
NEWS_API_KEY=your_newsapi_key_here
```

> **Demo mode:** Leave the keys blank — the app works with sample data and pre-built mock
> signals for AAPL, GOOGL, TSLA, MSFT, NVDA.

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📸 Screenshots

| Alpha Dashboard | AI Research Assistant |
|---|---|
| *(screenshot placeholder)* | *(screenshot placeholder)* |

| Signal History | How It Works |
|---|---|
| *(screenshot placeholder)* | *(screenshot placeholder)* |

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Gemini 1.5 Pro | Alpha signal generation with citation-backed reasoning |
| **RAG Framework** | LangChain LCEL | Retrieval chain, prompt templates, output parsers |
| **Vector Store** | ChromaDB | Local embedding storage & semantic similarity search |
| **Embeddings** | Google models/embedding-001 | 768-dim document embeddings |
| **Price Data** | yfinance | 6-month OHLCV historical data |
| **News** | NewsAPI | Real-time financial news ingestion |
| **Technical Analysis** | `ta` library | RSI(14), MA20, Volume Change % |
| **Sentiment** | TextBlob | Per-chunk polarity scoring (−1 to +1) |
| **Database** | SQLite (BigQuery schema) | Signal persistence & audit trail |
| **Frontend** | Streamlit + Plotly | Interactive dashboard |
| **Testing** | pytest | 52 unit tests, no API keys required |

---

## 📁 Project Structure

```
financial_alpha_generator/
├── app.py                     # Streamlit 4-page dashboard
├── requirements.txt           # Pinned dependencies
├── .env.template              # API key template
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── alpha_engine.py        # AlphaSignal dataclass + orchestration
│   ├── database.py            # SQLite (news, prices, signals tables)
│   ├── embeddings.py          # Chunking + TextBlob sentiment + ChromaDB
│   ├── ingestion.py           # NewsAPI + yfinance + RSI/MA20/vol
│   └── rag_pipeline.py        # Gemini 1.5 Pro LCEL chain
│
├── tests/
│   └── test_pipeline.py       # 52 unit tests (8 test classes)
│
└── data/
    └── sample_news.json       # Fallback news for demo mode
```

---

## 📊 Signal Schema

```python
@dataclass
class AlphaSignal:
    ticker:           str     # "AAPL"
    signal_strength:  int     # 1=Strong Sell ... 5=Strong Buy
    direction:        str     # "Strong Buy" | "Buy" | "Neutral" | "Sell" | "Strong Sell"
    confidence_score: float   # 0.0–1.0 (LLM + technical confirmation)
    reasoning:        str     # Full rationale with article citations
    news_citations:   list    # Article headlines used as evidence
    ticker_rsi:       float   # RSI(14) at signal time
    ticker_ma20:      float   # 20-day moving average
    vol_change_pct:   float   # Volume % vs 20-day average
    generated_at:     str     # ISO-8601 UTC timestamp
    model_version:    str     # "gemini-1.5-pro" | "mock"
```

---

## ☁️ Scaling to BigQuery

The SQLite schema mirrors BigQuery column types. To push to production:

```bash
# Export signals
streamlit run app.py  # then use Signal History → Export CSV

# Load to BigQuery
bq load \
  --source_format=CSV \
  --autodetect \
  my-project:alpha_signals.signals \
  alpha_signals_2024-01-01.csv
```

For streaming ingestion, replace `database.py` with a `google-cloud-bigquery` client using
the same column definitions. See `database.py` for the BigQuery-compatible CREATE TABLE DDL.

---

## 🧪 Running Tests

```bash
pytest tests/test_pipeline.py -v
# 52 passed in ~6 seconds (no API keys required)
```

Test coverage: sample news loading, document chunking, RSI/MA20 computation,
TextBlob sentiment, signal parser, AlphaSignal dataclass, technical confidence scoring,
database CRUD for all 3 tables.

---

## 💡 Why This Project?

Traditional quantitative alpha generation relies on:
- **Static rules** (e.g., RSI < 30 = buy) that don't adapt to market narratives
- **Lagging indicators** that miss fast-moving news-driven catalysts
- **Opaque signals** without explainable reasoning

This project demonstrates how **RAG fundamentally changes alpha generation**:

1. **Evidence-based reasoning** — Gemini cites specific news headlines, not just historical price patterns
2. **Adaptable context** — the vector store updates with fresh news; no retraining needed
3. **Hybrid scoring** — LLM confidence is refined by RSI and volume confirmation, reducing hallucination risk
4. **Audit trail** — every signal stored with its source articles, model version, and timestamp

---

## 📄 License

MIT © 2024 Nisha Kumari
