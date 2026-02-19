# 📈 AI Financial Alpha Generator

> **RAG-powered alpha signal engine** — Gemini 1.5 Pro · LangChain · ChromaDB · Streamlit

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Gemini](https://img.shields.io/badge/Gemini-1.5%20Pro-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=flat&logo=chainlink&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange?style=flat)](https://trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🔍 What is this?
The **AI Financial Alpha Generator** is a next-generation decision-support tool for traders and analysts. It leverages **Gemini 1.5 Pro** and **Retrieval-Augmented Generation (RAG)** to transform raw financial news into actionable, citation-backed alpha signals. 

Unlike traditional platforms that rely solely on lagging indicators, this engine reads the news through the eyes of a senior quantitative analyst, cross-references it with technical data, and provides structured trade recommendations with explicit reasoning.

---

## ⚙️ How it Works
The application follows a 4-stage intelligent pipeline:

1.  **Ingestion Layer**: Real-time news is fetched via NewsAPI and financial data via yfinance. The system calculates core technical indicators (RSI, MA20, Volume Change).
2.  **Semantic Processing**: Raw text is chunked and embedded into a **ChromaDB** vector store using Google's 768-dimensional embedding models. Each chunk is also scored for sentiment using TextBlob.
3.  **RAG Pipeline**: When you analyze a ticker, Gemini 1.5 Pro performs a semantic search across the news embeddings. It retrieves the most relevant catalysts, risks, and narratives to form a basis for its analysis.
4.  **Signal Generation**: The LLM output is parsed into a structured `AlphaSignal`. This signal is then "tempered" by the technical engine—if the AI is bullish but RSI indicates an overbought state, the confidence score is automatically adjusted to prevent "hallucinated" buys.

---

## 🏗️ Core Functionality
The application is organized into 5 intuitive modules:

1.  **📊 Alpha Dashboard**: View live signals (Strong Buy to Strong Sell), confidence percentages, and citation-backed reasoning. Execute one-click virtual paper trades with automated stop-loss and take-profit targets.
2.  **🧠 AI Research Assistant**: A dedicated RAG interface where you can ask complex questions (e.g., *"How do NVDA's Q3 guidance and recent AI chip delays impact the short-term signal?"*) and get cited answers.
3.  **💰 Paper Trading**: Manage your virtual $100k portfolio. Track open positions, unrealized PnL, and win rates in real-time.
4.  **🗄️ Signal History**: A permanent SQLite audit trail of every signal ever generated, allowing you to export data and review performance over time.
5.  **❓ How It Works**: An interactive map of the pipeline architecture and scaling guides.

---

## 💎 How it Differs from Other Tools
*   **Explainable AI (XAI)**: Most black-box AI tools give you a "Buy" signal without context. This tool gives you the exact news headlines it used to reach that conclusion.
*   **Narrative + Technical Hybrid**: It doesn't just look at charts. It understands the "Story" (Earnings, M&A, Product Launches) and uses charts only to confirm entries.
*   **Zero Retraining Needed**: By using RAG, the "brain" stays fresh. As soon as a news article is ingested, the AI knows about it—no fine-tuning required.
*   **Local Privacy**: Uses ChromaDB for local vector storage, ensuring your research history stays on your machine.

---

## 💡 Key Insights You Get
*   **Catalyst Identification**: Quickly find the *reason* behind a price move without reading 50 articles.
*   **Sentiment Divergence**: Spot when the news is purely hype but technicals (like declining volume) suggest a reversal.
*   **Risk-Adjusted Parameters**: Get precise, volatility-aware Entry, Stop-Loss, and Take-Profit levels for every signal.

---

## 🎯 Why & When to Use This?
**Use this when:**
*   You are overwhelmed by financial news and need a summary of what actually matters.
*   You want to verify your "gut feel" with a senior AI analyst that ignores emotional bias.
*   You need to back-audit your trading decisions using the Signal History.

**Why use it?**
To reduce cognitive load. Instead of spending 2 hours researching a ticker, you get a institutional-grade summary and technical confirmation in under 30 seconds.

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/financial-alpha-generator.git
cd financial-alpha-generator
pip install -r requirements.txt
python -m textblob.download_corpora
```

### 2. Configuration
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
NEWS_API_KEY=your_newsapi_key_here
```

### 3. Run
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Gemini 1.5 Pro | Deep reasoning & alpha generation |
| **Vector Store** | ChromaDB | Semantic search & news retrieval |
| **Data** | yfinance & NewsAPI | Market data & real-time narratives |
| **State** | LangChain LCEL | Pipeline orchestration |
| **DB** | SQLite | Persistent audit trail |

---

## 📄 License
MIT © 2024 Nisha Kumari
