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

Traditional alpha generation relies on hand-crafted rules and lagging indicators. This project replaces that with a **Retrieval-Augmented Generation (RAG)** architecture: financial news is ingested, embedded into a vector store, and queried by Gemini 1.5 Pro — which acts as a **senior quantitative analyst** — to generate structured, evidence-based trading signals.

The result is a suite of scalable, explainable, and citation-backed alpha signals that blend LLM reasoning with quantitative technical confirmation (RSI, MA20, volume analysis).

---

## 💡 Why This Project? & How It Differs

Most trading bots use either pure Technical Analysis (TA) or basic Sentiment Analysis. This project differs by:

1.  **Explainable AI (XAI)**: Instead of a "black box" prediction, the agent provides full reasoning, citing specific news articles as evidence for its bias.
2.  **Contextual Awareness**: It doesn't just see a "high RSI"; it understands *why* the price moved (e.g., an earnings beat vs. a speculative rumor) by analyzing news first.
3.  **Hybrid Confidence**: A signal is only marked as "Strong" if both the AI narrative (qualitative) and the technical indicators (quantitative) align.
4.  **No Re-training Required**: Unlike traditional ML, this uses RAG. Add new news, and the system immediately "knows" the current market state without expensive retraining.

---

## 🚀 Functions & Key Features

-   **📊 Alpha Dashboard**: Live signal cards showing strength, direction (Long/Short), and technical context (RSI, MA, Vol).
-   **🧠 AI Research Assistant**: A streaming Q&A interface to query your private news database using natural language.
-   **💰 Paper Trading**: Execute virtual trades based on AI signals to track performance in a risk-free environment.
-   **🗄️ Signal History**: A full audit log of every signal generated, stored in a BigQuery-ready SQLite database.
-   **❓ How It Works**: Transparent documentation of the ingestion, embedding, and inference pipeline.

---

## 🏗️ Architecture & How It Works

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
│  │          STREAMLIT DASHBOARD  (app.py)  — 5 Pages                   │   │
│  │  1. Alpha Dashboard  2. AI Research  3. Paper Trading               │   │
│  │  4. Signal History  5. How It Works                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Insights You Gain

-   **Deep Correlation**: Understand how specific news events (e.g., a regulatory filing or CEO tweet) correlate with technical breakouts.
-   **Narrative Sentiment**: Identify when market sentiment is shifting before it fully reflects in the lagging price indicators.
-   **Risk Identification**: The AI specifically looks for "Downside Risks" in reports that standard TA might miss (e.g., supply chain issues or litigation).

---

## 🎯 Why & When to Use This?

-   **Why**: To remove emotional bias from trading and back your decisions with high-fidelity LLM reasoning and real-time news data.
-   **When to use**:
    -   **Daily Pre-market Prep**: Scan your watchlists to see which tickers have the strongest AI-backed momentum.
    -   **Trade Auditing**: Before entering a technical setup, use the Research Assistant to cross-reference news catalysts or "Red Flags."
    -   **Strategy Refinement**: Use Paper Trading to validate the AI's "Win Rate" over time before risking real capital.

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
python -m textblob.download_corpora
```

### 3. Configure API keys
Edit `.env` (or set in Streamlit Secrets):
```env
GEMINI_API_KEY=your_key
NEWS_API_KEY=your_key
```

### 4. Run Locally
```bash
streamlit run app.py
```

---

## 🧪 Testing
```bash
pytest tests/test_pipeline.py -v
# 52 unit tests for schema, technicals, and RAG logic
```

---

## 📄 License
MIT © 2024 Nisha Kumari
