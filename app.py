"""
app.py — AI Financial Alpha Generator
4-Page Streamlit Dashboard

Pages
-----
1. Alpha Dashboard   — Signal cards, confidence bars, price chart
2. AI Research       — Streaming RAG Q&A with source citations
3. Signal History    — SQLite table view with filters & CSV export
4. How It Works      — Pipeline diagram + BigQuery scaling guide
"""

import logging
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# ── Path / Env Setup ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.alpha_engine import run_alpha_pipeline, AlphaSignal
from src.database import init_db, get_signals, get_articles, get_prices
from src.ingestion import (
    fetch_news, fetch_stock_data,
    compute_technical_indicators, summarise_technicals,
    load_sample_news,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Read from st.secrets (Streamlit Community Cloud) or .env (local dev)
def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.getenv(key, default))  # type: ignore[attr-defined]
    except Exception:
        return os.getenv(key, default)

GEMINI_KEY = _secret("GEMINI_API_KEY")
NEWS_KEY   = _secret("NEWS_API_KEY")

# On cloud use /tmp (writable ephemeral volume), locally use project dir
_IS_CLOUD  = os.getenv("HOME", "").startswith("/home/") or os.path.exists("/mount/src")
_DATA_ROOT = Path("/tmp") if _IS_CLOUD else ROOT_DIR
DB_PATH    = os.getenv("SQLITE_DB_PATH",  str(_DATA_ROOT / "data" / "alpha_signals.db"))
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", str(_DATA_ROOT / "chroma_db"))

GEMINI_OK  = bool(GEMINI_KEY and not GEMINI_KEY.startswith("your_"))
NEWS_OK    = bool(NEWS_KEY   and not NEWS_KEY.startswith("your_"))

ALL_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "JPM", "NFLX", "AMD"]

# Signal direction → color / badge styling
DIR_COLOR: dict[str, str] = {
    "Strong Buy":  "#10b981",
    "Buy":         "#34d399",
    "Neutral":     "#94a3b8",
    "Sell":        "#f87171",
    "Strong Sell": "#ef4444",
}
DIR_EMOJI: dict[str, str] = {
    "Strong Buy":  "🟢",
    "Buy":         "🟩",
    "Neutral":     "⬜",
    "Sell":        "🟥",
    "Strong Sell": "🔴",
}
DIR_BADGE: dict[str, str] = {
    "Strong Buy":  "signal-buy",
    "Buy":         "signal-buy",
    "Neutral":     "signal-neutral",
    "Sell":        "signal-sell",
    "Strong Sell": "signal-sell",
}

# ── Initialise DB ─────────────────────────────────────────────────────────────
init_db(DB_PATH)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Financial Alpha Generator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary:   #070d1a;
    --bg-card:      #0f1929;
    --bg-card2:     #151f2e;
    --accent-blue:  #3b82f6;
    --accent-green: #10b981;
    --accent-red:   #ef4444;
    --accent-gold:  #f59e0b;
    --text-primary: #f1f5f9;
    --text-muted:   #64748b;
    --border:       #1e293b;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: var(--bg-primary) !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1628 0%, #060c18 100%) !important;
    border-right: 1px solid var(--border);
}

/* Signal Cards */
.alpha-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    transition: transform 0.2s, box-shadow 0.2s;
    position: relative;
    overflow: hidden;
}
.alpha-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.alpha-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.alpha-card.buy::before   { background: linear-gradient(90deg, #10b981, #34d399); }
.alpha-card.sell::before  { background: linear-gradient(90deg, #ef4444, #f87171); }
.alpha-card.neutral::before { background: linear-gradient(90deg, #475569, #64748b); }

/* Summary Metric Card */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 2rem; font-weight: 800; line-height: 1.1; }
.metric-label { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.07em; }

/* Signal badge */
.signal-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
}
.signal-buy     { background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.4); }
.signal-sell    { background: rgba(239,68,68,0.12);  color: #ef4444; border: 1px solid rgba(239,68,68,0.4);  }
.signal-neutral { background: rgba(148,163,184,0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }

/* Confidence bar */
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    height: 7px;
    overflow: hidden;
    margin-top: 6px;
}
.conf-bar { height: 7px; border-radius: 6px; transition: width 0.8s ease; }

/* News card */
.news-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-blue);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.55rem;
}

/* RAG output */
.rag-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-green);
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
    white-space: pre-wrap;
    font-size: 0.9rem;
    color: var(--text-primary);
    line-height: 1.7;
}

/* Citation chip */
.cite-chip {
    display: inline-block;
    background: rgba(59,130,246,0.1);
    border: 1px solid rgba(59,130,246,0.3);
    color: #93c5fd;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.73rem;
    margin: 2px 3px;
}

/* Section header */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1.1rem;
    letter-spacing: 0.02em;
}

/* Pipeline step box */
.pipeline-step {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    min-height: 90px;
}
.pipeline-step .step-icon { font-size: 1.8rem; }
.pipeline-step .step-label { font-size: 0.75rem; color: var(--text-muted); margin-top: 4px; }
.pipeline-step .step-name  { font-weight: 700; color: var(--text-primary); font-size: 0.9rem; }

/* Status dot */
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px; }
.dot-live { background: #10b981; box-shadow: 0 0 6px #10b981; }
.dot-demo { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1e40af) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.35) !important;
}
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.9rem; }
h1, h2, h3 { color: var(--text-primary) !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _color(direction: str) -> str:
    return DIR_COLOR.get(direction, "#94a3b8")

def _emoji(direction: str) -> str:
    return DIR_EMOJI.get(direction, "⬜")

def _badge_class(direction: str) -> str:
    return DIR_BADGE.get(direction, "signal-neutral")

def _card_class(direction: str) -> str:
    d = direction.lower()
    if "buy" in d:  return "buy"
    if "sell" in d: return "sell"
    return "neutral"

def _strength_label(strength: int) -> str:
    return {1: "Strong Sell", 2: "Sell", 3: "Neutral", 4: "Buy", 5: "Strong Buy"}.get(strength, "Neutral")


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "signals":    {},      # ticker → AlphaSignal
        "stock_dfs":  {},      # ticker → DataFrame (with indicators)
        "articles":   {},      # ticker → list[dict]
        "rag_history": [],     # list[{question, answer, ticker, citations}]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='font-size:1.25rem; font-weight:800; color:#f1f5f9; margin-bottom:4px;'>"
        "📈 Alpha Generator</div>"
        "<div style='font-size:0.72rem; color:#64748b; margin-bottom:12px;'>"
        "RAG · Gemini 1.5 Pro · ChromaDB</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # API Status
    g_dot = "dot-live" if GEMINI_OK else "dot-demo"
    n_dot = "dot-live" if NEWS_OK   else "dot-demo"
    st.markdown(
        f"<div style='font-size:0.82rem; margin-bottom:8px;'>"
        f"<span class='status-dot {g_dot}'></span><b>Gemini:</b> {'Live ✅' if GEMINI_OK else 'Demo mode'}<br>"
        f"<span class='status-dot {n_dot}'></span><b>NewsAPI:</b> {'Live ✅' if NEWS_OK else 'Sample fallback'}"
        f"</div>",
        unsafe_allow_html=True,
    )
    if not GEMINI_OK:
        st.info("Add `GEMINI_API_KEY` to `.env` for live AI signals.", icon="💡")

    st.divider()

    # Ticker selection
    st.markdown("<div style='font-size:0.8rem; color:#94a3b8; font-weight:600; margin-bottom:6px;'>TICKERS</div>",
                unsafe_allow_html=True)
    selected_tickers = st.multiselect(
        "Tickers", options=ALL_TICKERS,
        default=["AAPL", "NVDA"],
        label_visibility="collapsed",
    )

    # Price period
    period = st.select_slider(
        "Price History",
        options=["1mo", "3mo", "6mo", "1y"],
        value="6mo",
    )

    # Force re-embed
    force_refresh = st.checkbox("Force re-embed news", value=False)

    st.divider()
    run_btn = st.button("⚡ Fetch & Analyze", use_container_width=True)
    st.divider()

    # Page navigation nav info
    st.markdown(
        "<div style='font-size:0.72rem; color:#475569;'>"
        "Use the tabs above to switch between:<br>"
        "📊 Alpha Dashboard · 🧠 AI Research<br>"
        "🗄️ Signal History · ❓ How It Works"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if run_btn:
    if not selected_tickers:
        st.warning("Please select at least one ticker.")
    else:
        progress = st.progress(0, text="Initialising pipeline…")
        total = len(selected_tickers)

        for i, ticker in enumerate(selected_tickers):
            progress.progress(i / total, text=f"Analysing {ticker} ({i+1}/{total})…")
            with st.spinner(f"Running alpha pipeline for {ticker}…"):
                try:
                    signal: AlphaSignal = run_alpha_pipeline(
                        ticker=ticker,
                        gemini_api_key=GEMINI_KEY,
                        news_api_key=NEWS_KEY,
                        db_path=DB_PATH,
                        chroma_persist_dir=CHROMA_DIR,
                        force_refresh=force_refresh,
                    )
                    st.session_state.signals[ticker] = signal

                    df = fetch_stock_data(ticker, period=period)
                    if not df.empty:
                        df = compute_technical_indicators(df)
                    st.session_state.stock_dfs[ticker] = df

                    arts = fetch_news(ticker, NEWS_KEY)
                    st.session_state.articles[ticker] = arts

                except Exception as exc:
                    st.error(f"Pipeline failed for {ticker}: {exc}")
                    logger.error("Pipeline error for %s: %s", ticker, exc, exc_info=True)

        progress.progress(1.0, text="Analysis complete ✅")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='font-size:1.9rem; font-weight:800; margin-bottom:2px;'>"
    "📊 AI Financial Alpha Generator</h1>"
    "<p style='color:#475569; font-size:0.88rem; margin-top:0;'>"
    "RAG-powered alpha signals &nbsp;·&nbsp; Gemini 1.5 Pro &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; LangChain</p>",
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Alpha Dashboard",
    "🧠 AI Research Assistant",
    "🗄️ Signal History",
    "❓ How It Works",
    "📉 Backtest",
])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ALPHA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    if not st.session_state.signals:
        st.markdown(
            "<div style='text-align:center; padding:4rem; color:#475569;'>"
            "<div style='font-size:3rem;'>📈</div>"
            "<h2 style='color:#475569; margin-top:0.5rem;'>Select tickers & click ⚡ Fetch & Analyze</h2>"
            "<p>The pipeline will ingest news, generate embeddings, and produce AI alpha signals.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        # ── Summary metric row ────────────────────────────────────────────────
        st.markdown('<div class="section-header">Current Alpha Signals</div>', unsafe_allow_html=True)

        cols = st.columns(len(st.session_state.signals))
        for col, (ticker, sig) in zip(cols, st.session_state.signals.items()):
            c      = _color(sig.direction)
            emoji  = _emoji(sig.direction)
            badge  = _badge_class(sig.direction)
            pct    = int(sig.confidence_score * 100)
            with col:
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div style='font-size:1.4rem; font-weight:800; color:{c};'>{emoji} {ticker}</div>"
                    f"<div style='margin:6px 0;'><span class='signal-badge {badge}'>{sig.direction}</span></div>"
                    f"<div class='metric-value' style='color:{c};'>{pct}<span style='font-size:1rem; color:#64748b;'>%</span></div>"
                    f"<div class='metric-label'>Confidence</div>"
                    f"<div class='conf-bar-bg'>"
                    f"<div class='conf-bar' style='width:{pct}%; background:{c};'></div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Detailed signal cards ──────────────────────────────────────────────
        st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)

        for ticker, sig in st.session_state.signals.items():
            c      = _color(sig.direction)
            emoji  = _emoji(sig.direction)
            pct    = int(sig.confidence_score * 100)
            badge  = _badge_class(sig.direction)
            cc     = _card_class(sig.direction)

            with st.expander(
                f"{emoji} {ticker}  ·  {sig.direction}  ·  Confidence: {pct}%",
                expanded=True,
            ):
                c1, c2, c3 = st.columns([4, 3, 2])

                with c1:
                    st.markdown("**📋 AI Reasoning**")
                    st.markdown(
                        f"<div style='color:#cbd5e1; font-size:0.88rem; line-height:1.7;'>{sig.reasoning}</div>",
                        unsafe_allow_html=True,
                    )
                    if sig.news_citations:
                        st.markdown("<br>**📰 News Citations**", unsafe_allow_html=True)
                        chips = "".join([f"<span class='cite-chip'>📎 {c}</span>" for c in sig.news_citations[:6]])
                        st.markdown(chips, unsafe_allow_html=True)

                with c2:
                    st.markdown("**📊 Technical Context**")
                    df_p = st.session_state.stock_dfs.get(ticker, pd.DataFrame())
                    if not df_p.empty:
                        tech = summarise_technicals(df_p, ticker)
                        st.code(tech, language=None)
                    else:
                        st.info("Price data not loaded.")

                with c3:
                    st.markdown("**📌 Signal Metadata**")
                    rsi_val = f"{sig.ticker_rsi:.1f}" if sig.ticker_rsi == sig.ticker_rsi else "N/A"
                    ma_val  = f"${sig.ticker_ma20:.2f}" if sig.ticker_ma20 == sig.ticker_ma20 else "N/A"
                    vc_val  = f"{sig.vol_change_pct:+.1f}%" if sig.vol_change_pct == sig.vol_change_pct else "N/A"
                    st.markdown(
                        f"<div style='font-size:0.82rem; line-height:2;'>"
                        f"🔢 <b>Strength:</b> {sig.signal_strength}/5<br>"
                        f"📈 <b>RSI(14):</b> {rsi_val}<br>"
                        f"〰️ <b>MA20:</b> {ma_val}<br>"
                        f"📦 <b>Volume Δ:</b> {vc_val}<br>"
                        f"🤖 <b>Model:</b> {sig.model_version}<br>"
                        f"🕐 <b>Generated:</b> {sig.generated_at[:10]}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── Price Chart ────────────────────────────────────────────────────────
        if st.session_state.stock_dfs:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Price Chart</div>', unsafe_allow_html=True)

            chart_ticker = st.selectbox(
                "Select Ticker",
                list(st.session_state.stock_dfs.keys()),
                key="chart_select",
            )
            df = st.session_state.stock_dfs.get(chart_ticker, pd.DataFrame())

            if df.empty:
                st.warning(f"No price data for {chart_ticker}.")
            else:
                df = df.copy()
                if "MA20" not in df.columns:
                    df["MA20"] = df["Close"].rolling(20).mean()
                df["MA50"] = df["Close"].rolling(50).mean()

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"], name=chart_ticker,
                    increasing_line_color="#10b981", decreasing_line_color="#ef4444",
                    increasing_fillcolor="rgba(16,185,129,0.25)",
                    decreasing_fillcolor="rgba(239,68,68,0.25)",
                ))
                fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20",
                    line=dict(color="#3b82f6", width=1.5, dash="dot")))
                fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50",
                    line=dict(color="#f59e0b", width=1.5, dash="dash")))
                if "RSI" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df["RSI"], name="RSI(14)", yaxis="y3",
                        line=dict(color="#a78bfa", width=1.2),
                    ))
                fig.add_trace(go.Bar(
                    x=df.index, y=df["Volume"], name="Volume",
                    yaxis="y2", marker_color="rgba(59,130,246,0.2)",
                ))

                # Signal annotation
                if chart_ticker in st.session_state.signals:
                    sig = st.session_state.signals[chart_ticker]
                    col_ann = _color(sig.direction)
                    fig.add_annotation(
                        x=df.index[-1], y=df["Close"].iloc[-1],
                        text=f" {sig.direction} ({int(sig.confidence_score*100)}%)",
                        showarrow=True, arrowhead=2, arrowcolor=col_ann,
                        font=dict(color=col_ann, size=12, family="Inter"),
                        bgcolor="rgba(0,0,0,0.75)", borderpad=5,
                    )

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(15,25,41,0.9)",
                    font=dict(family="Inter", color="#94a3b8"),
                    xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                               title="Price (USD)"),
                    yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                title="Volume", tickformat=".2s", showticklabels=False),
                    yaxis3=dict(overlaying="y", side="right", anchor="free",
                                position=0.97, range=[0, 100],
                                title="RSI", showgrid=False, showticklabels=False),
                    legend=dict(orientation="h", y=1.02, x=0),
                    height=520,
                    margin=dict(l=10, r=60, t=50, b=10),
                    title=dict(
                        text=f"<b>{chart_ticker}</b> — Price + MA20/50 + RSI + Volume",
                        font=dict(size=15, color="#f1f5f9"),
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — AI RESEARCH ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">🧠 AI Research Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b; font-size:0.88rem;'>"
        "Ask anything about ingested news, e.g. <em>\"Why is NVDA bullish?\"</em> or "
        "<em>\"Compare TSLA and AAPL signals.\"</em></p>",
        unsafe_allow_html=True,
    )

    # Ticker scope
    scope_options = ["All (first analysed)"] + list(st.session_state.signals.keys())
    qa_ticker_label = st.selectbox(
        "Scope to ticker",
        scope_options,
        key="qa_ticker_select",
        help="The RAG search is scoped to the selected ticker's ChromaDB collection.",
    )
    qa_ticker = (
        list(st.session_state.signals.keys())[0]
        if (qa_ticker_label == "All (first analysed)" and st.session_state.signals)
        else qa_ticker_label
    )

    # Suggested questions
    st.markdown(
        "<div style='font-size:0.77rem; color:#475569; margin-bottom:6px;'>💡 Try: </div>",
        unsafe_allow_html=True,
    )
    sugg_cols = st.columns(3)
    suggestions = [
        "What are the key growth catalysts?",
        "What are the main downside risks?",
        "Is the current valuation justified?",
    ]
    if "qa_prefill" not in st.session_state:
        st.session_state.qa_prefill = ""
    for i, sugg in enumerate(suggestions):
        with sugg_cols[i]:
            if st.button(sugg, key=f"sugg_{i}", use_container_width=True):
                st.session_state.qa_prefill = sugg

    question = st.text_area(
        "Your question",
        value=st.session_state.qa_prefill,
        placeholder="e.g. What catalysts are driving NVDA's growth?",
        height=90,
        key="qa_question",
    )

    ask_btn = st.button("🔍 Ask Gemini", use_container_width=True, key="ask_btn")

    if ask_btn:
        if not question.strip():
            st.warning("Please type a question first.")
        elif not st.session_state.signals:
            st.warning("Run the pipeline first (⚡ Fetch & Analyze) to ingest news into ChromaDB.")
        else:
            with st.spinner("Querying Gemini via RAG…"):
                try:
                    if GEMINI_OK:
                        from src.embeddings import get_or_create_vectorstore
                        from src.rag_pipeline import build_rag_chain, query_rag
                        vs = get_or_create_vectorstore(
                            collection_name=f"{qa_ticker.lower()}_news",
                            persist_dir=CHROMA_DIR,
                            api_key=GEMINI_KEY,
                        )
                        chain_tuple = build_rag_chain(vs, api_key=GEMINI_KEY)
                        result = query_rag(chain_tuple, question)
                        answer    = result["answer"]
                        sources   = result.get("sources", [])
                        citations = list({
                            doc.metadata.get("title", "")
                            for doc in sources if doc.metadata.get("title")
                        })
                    else:
                        answer = (
                            f"[Demo Mode — No Gemini Key]\n\n"
                            f"You asked: \"{question}\"\n\n"
                            "In live mode, Gemini 1.5 Pro would analyse the ingested ChromaDB \n"
                            "collection and produce a detailed, citation-backed answer.\n\n"
                            "To enable: set GEMINI_API_KEY in your .env file."
                        )
                        citations = []

                    st.session_state.rag_history.append({
                        "question": question,
                        "answer": answer,
                        "ticker": qa_ticker,
                        "citations": citations,
                    })
                    st.session_state.qa_prefill = ""

                except Exception as exc:
                    st.error(f"RAG query failed: {exc}")

    # ── Conversation History ───────────────────────────────────────────────────
    if st.session_state.rag_history:
        st.markdown("---")
        st.markdown('<div class="section-header">💬 Conversation History</div>', unsafe_allow_html=True)

        for item in reversed(st.session_state.rag_history[-8:]):
            st.markdown(
                f"<div style='font-size:0.82rem; color:#64748b; margin-bottom:2px;'>"
                f"📌 [{item['ticker']}] &nbsp;·&nbsp; "
                f"{datetime.now().strftime('%H:%M')}</div>"
                f"<div style='font-weight:700; color:#f1f5f9; margin-bottom:6px;'>Q: {item['question']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f'<div class="rag-box">{item["answer"]}</div>', unsafe_allow_html=True)

            # Source citations
            cits = item.get("citations", [])
            if cits:
                with st.expander(f"📎 Sources ({len(cits)} articles used)", expanded=False):
                    for c in cits:
                        st.markdown(f"- {c}")
            st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SIGNAL HISTORY
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">🗄️ Signal History — SQLite Store</div>', unsafe_allow_html=True)

    # Fetch all signals from DB
    all_db_signals = get_signals(DB_PATH, limit=500)

    if not all_db_signals:
        st.info("No signals in the database yet. Run ⚡ Fetch & Analyze to generate some.", icon="ℹ️")
    else:
        df_hist = pd.DataFrame(all_db_signals)

        # Friendly column rename
        rename_map = {
            "ticker":           "Ticker",
            "direction":        "Direction",
            "signal_strength":  "Strength (1-5)",
            "confidence_score": "Confidence",
            "reasoning":        "Reasoning",
            "rsi":              "RSI",
            "ma20":             "MA20",
            "vol_change_pct":   "Vol Δ%",
            "generated_at":     "Generated At",
            "model_version":    "Model",
        }
        df_display = df_hist.rename(columns={k: v for k, v in rename_map.items() if k in df_hist.columns})

        # ── Filters ───────────────────────────────────────────────────────────
        f1, f2, f3 = st.columns(3)
        with f1:
            ticker_filter = st.multiselect(
                "Filter by Ticker",
                options=sorted(df_display["Ticker"].unique().tolist()) if "Ticker" in df_display else [],
            )
        with f2:
            dir_options = sorted(df_display["Direction"].unique().tolist()) if "Direction" in df_display else []
            dir_filter = st.multiselect("Filter by Direction", options=dir_options)
        with f3:
            date_filter = st.date_input(
                "From date",
                value=date.today() - timedelta(days=30),
            )

        # Apply filters
        filtered = df_display.copy()
        if ticker_filter:
            filtered = filtered[filtered["Ticker"].isin(ticker_filter)]
        if dir_filter:
            filtered = filtered[filtered["Direction"].isin(dir_filter)]
        if "Generated At" in filtered.columns:
            filtered["Generated At"] = pd.to_datetime(filtered["Generated At"], utc=True, errors="coerce")
            cutoff = pd.Timestamp(date_filter, tz="UTC")
            filtered = filtered[filtered["Generated At"] >= cutoff]
            filtered["Generated At"] = filtered["Generated At"].dt.strftime("%Y-%m-%d %H:%M UTC")

        # Format confidence as %
        if "Confidence" in filtered.columns:
            filtered["Confidence"] = (filtered["Confidence"] * 100).round(1).astype(str) + "%"

        # Display columns
        display_cols = [c for c in ["Ticker", "Direction", "Strength (1-5)", "Confidence",
                                     "RSI", "MA20", "Vol Δ%", "Model", "Generated At"]
                        if c in filtered.columns]

        def _style_dir(val):
            m = {
                "Strong Buy": "#10b981", "Buy": "#34d399",
                "Neutral": "#94a3b8",
                "Sell": "#f87171", "Strong Sell": "#ef4444",
            }
            return f"color: {m.get(val, 'white')}; font-weight: 700;"

        styled = (
            filtered[display_cols]
            .style
            .applymap(_style_dir, subset=["Direction"] if "Direction" in display_cols else [])
            .format({
                "RSI":    lambda x: f"{x:.1f}" if pd.notna(x) else "N/A",
                "MA20":   lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
                "Vol Δ%": lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A",
            }, na_rep="N/A")
            .set_properties(**{"font-size": "0.82rem"})
        )

        st.markdown(f"<div style='font-size:0.8rem; color:#64748b; margin-bottom:6px;'>"
                    f"Showing {len(filtered)} signal(s)</div>", unsafe_allow_html=True)
        st.dataframe(styled, use_container_width=True, height=420)

        # ── Download ──────────────────────────────────────────────────────────
        csv = filtered[display_cols].to_csv(index=False)
        st.download_button(
            label="⬇️ Export to CSV",
            data=csv,
            file_name=f"alpha_signals_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ── Raw news articles from DB ─────────────────────────────────────────
        with st.expander("📰 View Raw News Articles (from DB)", expanded=False):
            t_opts = ["All"] + sorted(df_display["Ticker"].unique().tolist()) if "Ticker" in df_display else ["All"]
            t_pick = st.selectbox("Filter articles by ticker", t_opts, key="art_pick")
            articles_db = get_articles(DB_PATH, ticker=None if t_pick == "All" else t_pick, limit=50)
            if articles_db:
                df_art = pd.DataFrame(articles_db)[["ticker","title","source","published_at","sentiment_score"]]
                df_art.columns = ["Ticker","Headline","Source","Published","Sentiment"]
                st.dataframe(df_art, use_container_width=True, height=300)
            else:
                st.write("No articles in DB yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">❓ How It Works</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#64748b; font-size:0.9rem; max-width:700px;'>"
        "This system combines <b style='color:#f1f5f9;'>Retrieval-Augmented Generation (RAG)</b> "
        "with quantitative technical analysis to produce institutional-grade alpha signals. "
        "Here's the full data pipeline:"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Pipeline diagram ──────────────────────────────────────────────────────
    steps = [
        ("📰", "News Fetch",       "NewsAPI\n+ yfinance"),
        ("✂️", "Chunking",          "500 tokens\n50-token overlap"),
        ("🟣", "Sentiment",         "TextBlob\npolarity per chunk"),
        ("🗄️", "ChromaDB",          "Google Embeddings\nVector Store"),
        ("🤖", "Gemini 1.5 Pro",    "LCEL RAG Chain\nk=5 retrieval"),
        ("📊", "Alpha Signal",      "1–5 Strength\n+ Confidence Score"),
        ("💾", "SQLite / BigQuery", "Persist &\nAudit Trail"),
    ]

    pipe_cols = st.columns(len(steps))
    for col, (icon, name, detail) in zip(pipe_cols, steps):
        with col:
            st.markdown(
                f"<div class='pipeline-step'>"
                f"<div class='step-icon'>{icon}</div>"
                f"<div class='step-name'>{name}</div>"
                f"<div class='step-label'>{detail}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Arrows between steps
    arrow_cols = st.columns(len(steps) * 2 - 1)
    for i in range(len(steps) * 2 - 1):
        with arrow_cols[i]:
            if i % 2 == 1:
                st.markdown(
                    "<div style='text-align:center; color:#3b82f6; font-size:1.2rem; padding-top:28px;'>→</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Columns: Technical Confidence + BigQuery ───────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("#### 🔬 Technical Confidence Scoring")
        st.markdown(
            "<p style='color:#64748b; font-size:0.85rem;'>"
            "The final confidence score blends the LLM's self-reported certainty with "
            "independent confirmation from technical indicators:</p>",
            unsafe_allow_html=True,
        )
        st.markdown("""
| Condition | Direction | Adjustment |
|-----------|-----------|------------|
| RSI < 30 (oversold) | Buy / Strong Buy | **+15%** ✅ |
| RSI > 70 (overbought) | Sell / Strong Sell | **+15%** ✅ |
| RSI < 30 (oversold) | Sell (conflict) | **−10%** ⚠️ |
| RSI > 70 (overbought) | Buy (conflict) | **−10%** ⚠️ |
| Volume > +50% avg | Any | **+10%** 📈 |
| Volume > +100% avg | Any | **+15%** 🔥 |
""")

    with right:
        st.markdown("#### ☁️ Scaling to Production (BigQuery)")
        st.markdown(
            "<p style='color:#64748b; font-size:0.85rem;'>"
            "The SQLite schema is designed to be <b style='color:#f1f5f9;'>BigQuery-compatible</b>. "
            "To scale to production:</p>",
            unsafe_allow_html=True,
        )
        st.code("""
# 1. Create a GCP project & BigQuery dataset
gcloud projects create my-alpha-project
bq mk --dataset my-alpha-project:alpha_signals

# 2. Set credentials
export GOOGLE_APPLICATION_CREDENTIALS=key.json

# 3. Push SQLite → BigQuery
bq load \\
  --source_format=CSV \\
  --autodetect \\
  alpha_signals.signals \\
  alpha_signals_export.csv
""", language="bash")

        st.markdown("""
**BigQuery benefits at scale:**
- 🚀 Query billions of rows in seconds
- 📡 Real-time streaming via Pub/Sub
- 🔒 IAM roles for team access control
- 📊 Looker Studio dashboards
- 🔗 Connect to Vertex AI for model serving
""")

    st.divider()

    # ── Signal strength legend ─────────────────────────────────────────────────
    st.markdown("#### 📡 Signal Strength Scale")
    strength_cols = st.columns(5)
    strength_data = [
        (1, "Strong Sell", "#ef4444"),
        (2, "Sell",        "#f87171"),
        (3, "Neutral",     "#94a3b8"),
        (4, "Buy",         "#34d399"),
        (5, "Strong Buy",  "#10b981"),
    ]
    for col, (s, label, color) in zip(strength_cols, strength_data):
        with col:
            st.markdown(
                f"<div style='text-align:center; background:#0f1929; border:1px solid #1e293b; "
                f"border-radius:10px; padding:0.8rem; border-top: 3px solid {color};'>"
                f"<div style='font-size:1.6rem; font-weight:800; color:{color};'>{s}</div>"
                f"<div style='font-size:0.75rem; color:{color}; font-weight:600;'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Tech stack ────────────────────────────────────────────────────────────
    st.markdown("#### 🛠️ Tech Stack")
    stack_cols = st.columns(4)
    stack = [
        ("🤖", "Gemini 1.5 Pro",  "google-generativeai"),
        ("🦜", "LangChain LCEL",  "langchain, langchain-chroma"),
        ("🟣", "ChromaDB",        "Local vector store"),
        ("📊", "yfinance + ta",   "RSI, MA20, vol analysis"),
        ("💬", "TextBlob",        "Sentiment per chunk"),
        ("📰", "NewsAPI",         "Financial news ingestion"),
        ("💾", "SQLite / BQ",     "Production-ready schema"),
        ("🖥️", "Streamlit",       "Real-time dashboard"),
    ]
    for i, (icon, name, detail) in enumerate(stack):
        with stack_cols[i % 4]:
            st.markdown(
                f"<div style='background:#0f1929; border:1px solid #1e293b; border-radius:8px; "
                f"padding:0.7rem; margin-bottom:0.5rem;'>"
                f"<span style='font-size:1.1rem;'>{icon}</span> "
                f"<b style='color:#f1f5f9; font-size:0.85rem;'>{name}</b><br>"
                f"<span style='color:#475569; font-size:0.72rem;'>{detail}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown(
        "<h2 style='font-size:1.4rem; font-weight:700; margin-bottom:4px;'>"
        "📉 Strategy Backtest — 1 Year</h2>"
        "<p style='color:#64748b; font-size:0.83rem; margin-top:0;'>"
        "RSI(14) + MA20/MA50 crossover signal · Long-only · 0.1% transaction cost · "
        "vs equal-weight buy-and-hold benchmark</p>",
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    bt_col1, bt_col2, bt_col3 = st.columns([2, 1, 1])
    with bt_col1:
        bt_tickers = st.multiselect(
            "Tickers to backtest",
            ALL_TICKERS,
            default=["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"],
            key="bt_tickers",
        )
    with bt_col2:
        bt_longonly = st.toggle("Long-only mode", value=True, key="bt_long")
    with bt_col3:
        bt_tc = st.selectbox(
            "Transaction cost",
            [0.0, 0.001, 0.002, 0.005],
            index=1,
            format_func=lambda x: f"{x*100:.1f}%",
            key="bt_tc",
        )

    run_bt = st.button("▶ Run Backtest", type="primary", key="run_bt", use_container_width=True)

    if run_bt or "bt_result" in st.session_state:
        if run_bt:
            if not bt_tickers:
                st.warning("Select at least one ticker.")
                st.stop()

            with st.spinner("Fetching 1 year of price data and running simulation…"):
                try:
                    from src.backtest import run_backtest
                    bt_result, bt_sim = run_backtest(
                        tickers=bt_tickers,
                        period="1y",
                        long_only=bt_longonly,
                        transaction_cost=bt_tc,
                    )
                    st.session_state["bt_result"] = bt_result
                    st.session_state["bt_sim"] = bt_sim
                except Exception as e:
                    st.error(f"Backtest failed: {e}")
                    st.stop()

        bt_result = st.session_state["bt_result"]
        bt_sim    = st.session_state["bt_sim"]

        # ── Section: Performance Summary ──────────────────────────────────────
        st.markdown("### 📊 Performance Summary")

        def _color(val: float) -> str:
            return "#10b981" if val >= 0 else "#ef4444"

        def _metric_card(label: str, value: str, subtext: str = "", color: str = "#f1f5f9") -> str:
            return (
                f"<div style='background:#0f1929; border:1px solid #1e293b; border-radius:10px; "
                f"padding:1rem 1.2rem; margin-bottom:0.5rem;'>"
                f"<div style='font-size:0.72rem; color:#64748b; font-weight:600; text-transform:uppercase; "
                f"letter-spacing:0.05em;'>{label}</div>"
                f"<div style='font-size:1.6rem; font-weight:800; color:{color}; margin:2px 0;'>{value}</div>"
                f"<div style='font-size:0.75rem; color:#475569;'>{subtext}</div>"
                f"</div>"
            )

        row1 = st.columns(4)
        row2 = st.columns(4)

        tr   = bt_result.total_return
        cagr = bt_result.cagr
        sr   = bt_result.sharpe_ratio
        md   = bt_result.max_drawdown
        so   = bt_result.sortino_ratio
        cal  = bt_result.calmar_ratio
        wr   = bt_result.win_rate
        alp  = bt_result.alpha_vs_benchmark

        with row1[0]:
            st.markdown(_metric_card(
                "Total Return", f"{tr*100:+.2f}%",
                f"Benchmark: {bt_result.bm_total_return*100:+.2f}%",
                _color(tr)), unsafe_allow_html=True)
        with row1[1]:
            st.markdown(_metric_card(
                "CAGR", f"{cagr*100:+.2f}%",
                "Compound Annual Growth Rate",
                _color(cagr)), unsafe_allow_html=True)
        with row1[2]:
            sr_color = "#10b981" if sr >= 1 else "#f59e0b" if sr >= 0 else "#ef4444"
            st.markdown(_metric_card(
                "Sharpe Ratio", f"{sr:.2f}",
                "≥1 Good · ≥2 Excellent · <0 Bad",
                sr_color), unsafe_allow_html=True)
        with row1[3]:
            st.markdown(_metric_card(
                "Max Drawdown", f"{md*100:.2f}%",
                f"Benchmark: {bt_result.bm_max_drawdown*100:.2f}%",
                _color(md)), unsafe_allow_html=True)

        with row2[0]:
            st.markdown(_metric_card(
                "Sortino Ratio", f"{so:.2f}",
                "Downside-adjusted Sharpe",
                "#10b981" if so >= 1 else "#f59e0b"), unsafe_allow_html=True)
        with row2[1]:
            st.markdown(_metric_card(
                "Calmar Ratio", f"{cal:.2f}",
                "CAGR / Max Drawdown",
                "#10b981" if cal >= 0.5 else "#f59e0b"), unsafe_allow_html=True)
        with row2[2]:
            st.markdown(_metric_card(
                "Win Rate", f"{wr*100:.1f}%",
                f"Avg Win: {bt_result.avg_win*100:+.2f}% | Avg Loss: {bt_result.avg_loss*100:+.2f}%",
                "#10b981" if wr >= 0.5 else "#f59e0b"), unsafe_allow_html=True)
        with row2[3]:
            st.markdown(_metric_card(
                "Alpha vs Benchmark", f"{alp*100:+.2f}%",
                f"Beta: {bt_result.beta:.2f}",
                _color(alp)), unsafe_allow_html=True)

        # ── Section: Extra stats row ───────────────────────────────────────────
        st.markdown("---")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Ann. Volatility", f"{bt_result.volatility_ann*100:.2f}%",
                  f"BM: {bt_result.bm_volatility_ann*100:.2f}%")
        s2.metric("Best Day",  f"{bt_result.best_day*100:+.2f}%")
        s3.metric("Worst Day", f"{bt_result.worst_day*100:+.2f}%")
        s4.metric("BM Sharpe", f"{bt_result.bm_sharpe:.2f}")
        s5.metric("Risk-Free Rate", "5.00% p.a.")

        # ── Section: Equity Curve ─────────────────────────────────────────────
        st.markdown("### 📈 Equity Curve vs Benchmark")
        dates = pd.to_datetime(bt_sim["date"])
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=dates, y=bt_result.equity_curve, name="Strategy",
            line=dict(color="#3b82f6", width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.07)",
        ))
        fig_eq.add_trace(go.Scatter(
            x=dates, y=bt_result.bm_equity_curve, name="Benchmark (E/W B&H)",
            line=dict(color="#f59e0b", width=1.8, dash="dot"),
        ))
        fig_eq.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="Portfolio Value ($)", gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
            height=320,
        )
        st.plotly_chart(fig_eq, use_container_width=True)

        # ── Section: Drawdown ─────────────────────────────────────────────────
        st.markdown("### 📉 Drawdown")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=dates, y=bt_result.drawdown_curve * 100,
            name="Strategy", line=dict(color="#ef4444", width=2),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
        ))
        fig_dd.add_trace(go.Scatter(
            x=dates, y=bt_result.bm_drawdown_curve * 100,
            name="Benchmark", line=dict(color="#f59e0b", width=1.4, dash="dot"),
        ))
        fig_dd.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f1f5f9"),
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="Drawdown (%)", gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified",
            height=240,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Section: Monthly Returns ──────────────────────────────────────────
        if not bt_result.monthly_returns.empty:
            st.markdown("### 🗓️ Monthly Returns")
            df_m = bt_result.monthly_returns.copy()
            df_m["return_pct"] = (df_m["return"] * 100).round(2)
            df_m["month_year"] = df_m["date"].dt.strftime("%b %Y")
            df_m["color"]      = df_m["return_pct"].apply(
                lambda x: f"{'🟢' if x >= 0 else '🔴'} {x:+.2f}%"
            )
            display_m = df_m[["month_year", "return_pct", "color"]].copy()
            display_m.columns = ["Month", "Return (%)", "Signal"]
            display_m = display_m.reset_index(drop=True)

            st.dataframe(
                display_m.style.applymap(
                    lambda v: "color:#10b981; font-weight:600;" if isinstance(v, float) and v >= 0
                    else ("color:#ef4444; font-weight:600;" if isinstance(v, float) else ""),
                    subset=["Return (%)"]
                ),
                use_container_width=True,
                hide_index=True,
                height=320,
            )

        # ── Disclaimer ────────────────────────────────────────────────────────
        st.markdown(
            "<div style='background:#0f1929; border:1px solid #f59e0b40; border-radius:8px; "
            "padding:0.9rem 1.1rem; margin-top:1rem; font-size:0.78rem; color:#94a3b8;'>"
            "⚠️ <b>Disclaimer:</b> This backtest uses a rule-based RSI+MA signal as a proxy for the "
            "AI-generated alpha signals. Past performance does not guarantee future results. "
            "No slippage or market-impact modelling applied. For research purposes only."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='text-align:center; padding:4rem; color:#475569;'>"
            "<div style='font-size:3rem;'>📉</div>"
            "<div style='font-size:1.1rem; font-weight:600; margin-top:0.5rem;'>Run the 1-Year Backtest</div>"
            "<div style='font-size:0.85rem; margin-top:0.5rem;'>Select tickers above and click "
            "<b>▶ Run Backtest</b> to simulate strategy performance.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
