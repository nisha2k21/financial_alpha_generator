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
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

# ── Path / Env Setup ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.alpha_engine import run_alpha_pipeline, AlphaSignal
from src.database import (
    init_db, get_signals, get_articles, get_prices,
    get_paper_portfolio, get_paper_trades, get_alerts,
    get_sentiment_history, get_journals, get_weekly_reviews
)
from src.ingestion import (
    fetch_news, fetch_stock_data,
    compute_technical_indicators, summarise_technicals,
    load_sample_news,
)
from src.paper_trader import PaperPortfolio
from src.backtester import run_backtest
from src.alert_engine import create_price_alert, check_all_alerts
from src.sentiment_tracker import SentimentTracker
from src.ai_coach import AICoach
from src.embeddings import get_or_create_vectorstore
from src.rag_pipeline import build_rag_chain, query_rag

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
        "paper_portfolio": PaperPortfolio(DB_PATH),
        "sentiment_tracker": SentimentTracker(DB_PATH),
        "ai_coach": AICoach(DB_PATH),
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Alpha Dashboard",
    "🧠 AI Research Assistant",
    "💰 Paper Trading",
    "⏪ Backtest",
    "🔔 Price Alerts",
    "🌡️ Sentiment Heatmap",
    "📔 Trading Journal & Coach",
    "🗄️ Signal History",
    "❓ How It Works",
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
        # ── Portfolio summary bar ─────────────────────────────────────────────
        st.markdown('<div class="section-header">Portfolio Position Summary</div>', unsafe_allow_html=True)
        long_count = sum(1 for s in st.session_state.signals.values() if s.position == "LONG")
        short_count = sum(1 for s in st.session_state.signals.values() if s.position == "SHORT")
        total_size = sum(s.position_size_pct for s in st.session_state.signals.values())
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"**Long Positions:** {long_count}")
        with c2:
            st.warning(f"**Short Positions:** {short_count}")
        with c3:
            st.success(f"**Allocated:** {total_size:.1f}%")

        # ── Trade Recommendations Table ───────────────────────────────────────
        st.markdown('<div class="section-header">Long/Short Trade Recommendations</div>', unsafe_allow_html=True)
        trade_data = []
        for ticker, sig in st.session_state.signals.items():
            if sig.position != "NO_TRADE":
                trade_data.append({
                    "Ticker": ticker,
                    "Pos": sig.position,
                    "Size": f"{sig.position_size_pct}%",
                    "Entry": f"${sig.entry_price:.2f}",
                    "Stop Loss": f"${sig.stop_loss_price:.2f}",
                    "Take Profit": f"${sig.take_profit_price:.2f}",
                    "R:R": f"1:{sig.risk_reward_ratio:.1f}"
                })
        if trade_data:
            st.table(pd.DataFrame(trade_data))
        else:
            st.info("No actionable trades generated (confidence < 55%)")

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

                    st.markdown("**📌 Position Parameters**")
                    if sig.position != "NO_TRADE":
                        st.markdown(
                            f"<div style='background:rgba(255,255,255,0.05); padding:10px; border-radius:8px;'>"
                            f"<b>Entry:</b> ${sig.entry_price:.2f}<br>"
                            f"<b>Stop:</b> <span style='color:#ef4444;'>${sig.stop_loss_price:.2f}</span><br>"
                            f"<b>Target:</b> <span style='color:#10b981;'>${sig.take_profit_price:.2f}</span><br>"
                            f"<b>R:R:</b> 1:{sig.risk_reward_ratio:.1f}<br>"
                            f"<b>Size:</b> {sig.position_size_pct}%"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        if st.button(f"⚡ Execute {sig.position} on {ticker}", key=f"exec_{ticker}"):
                            try:
                                result = st.session_state.paper_portfolio.execute_trade(
                                    ticker=ticker,
                                    direction=sig.position,
                                    entry_price=sig.entry_price,
                                    position_size_pct=sig.position_size_pct,
                                    stop_loss=sig.stop_loss_price,
                                    take_profit=sig.take_profit_price,
                                    ai_signal=sig.direction,
                                    ai_confidence=sig.confidence_score
                                )
                                st.success(result)
                                st.balloons()
                            except Exception as e:
                                st.error(str(e))
                    else:
                        st.write("No active trade recommendation.")

                # Full Rationale Expanders
                with st.expander("📝 Full Trade Rationale & Strategy"):
                    st.write(sig.trade_rationale if sig.trade_rationale else "N/A")

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
# PAGE 3 — PAPER TRADING
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">💰 Virtual Paper Trading</div>', unsafe_allow_html=True)
    
    # Refresh positions
    if st.button("🔄 Sync & Refresh Positions"):
        with st.spinner("Updating live prices..."):
            st.session_state.paper_portfolio.update_positions()

    # Portfolio Summary
    summary = st.session_state.paper_portfolio.get_portfolio_summary()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Equity", f"${summary['total_value']:,.2f}", f"{summary['return_pct']:+.2f}%")
    m2.metric("Cash Balance", f"${summary['cash']:,.2f}")
    m3.metric("Win Rate", f"{summary['win_rate']:.1f}%")
    m4.metric("Open Positions", summary["open_positions"])

    # Open Positions Table
    st.markdown("#### 📌 Open Positions")
    positions = st.session_state.paper_portfolio.positions
    if not positions:
        st.info("No open positions. Use the Alpha Dashboard to execute signals.")
    else:
        pos_data = []
        for ticker, pos in positions.items():
            pos_data.append({
                "Ticker": ticker,
                "Type": pos.direction,
                "Qty": pos.quantity,
                "Entry": f"${pos.entry_price:.2f}",
                "Curr": f"${pos.current_price:.2f}",
                "P&L": f"${pos.unrealized_pnl:+.2f}",
                "P&L %": f"{pos.unrealized_pnl_pct:+.2f}%",
                "SL": f"${pos.stop_loss:.2f}",
                "TP": f"${pos.take_profit:.2f}"
            })
        st.table(pd.DataFrame(pos_data))
        
        # Close position selector
        sel_pos = st.selectbox("Close position", ["Select..."] + list(positions.keys()))
        if sel_pos != "Select...":
            if st.button(f"Close {sel_pos}"):
                res = st.session_state.paper_portfolio.close_position(sel_pos)
                st.success(res)
                st.rerun()

    # Trade History
    with st.expander("📜 Closed Trade History"):
        trades = get_paper_trades(DB_PATH)
        if trades:
            st.dataframe(pd.DataFrame(trades), use_container_width=True)
        else:
            st.write("No trades in history.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">⏪ Technical Strategy Backtester</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;'>Simulate institutional technical strategies over historical data.</p>", unsafe_allow_html=True)
    
    b1, b2, b3 = st.columns([2, 2, 2])
    with b1:
        bt_ticker = st.selectbox("Ticker", ALL_TICKERS, index=1)
        bt_start = st.date_input("Start Date", date.today() - timedelta(days=365*2))
    with b2:
        bt_size = st.slider("Position Size %", 5, 50, 10, 5)
        bt_capital = st.number_input("Initial Capital", 10000, 1000000, 100000)
    with b3:
        bt_sl = st.slider("Stop Loss (ATR Mult)", 1.0, 3.0, 1.5, 0.5)
        bt_tp = st.slider("Take Profit (ATR Mult)", 2.0, 5.0, 2.5, 0.5)

    if st.button("🚀 Run Walk-Forward Simulation", type="primary"):
        with st.spinner("Simulating..."):
            try:
                res = run_backtest(
                    ticker=bt_ticker,
                    start_date=bt_start.strftime("%Y-%m-%d"),
                    end_date=date.today().strftime("%Y-%m-%d"),
                    initial_capital=bt_capital,
                    position_size_pct=bt_size,
                    stop_loss_mult=bt_sl,
                    take_profit_mult=bt_tp
                )
                
                # Metrics
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Final Capital", f"${res.final_capital:,.0f}", f"{res.total_return_pct:+.1f}%")
                r2.metric("CAGR", f"{res.cagr:.1f}%")
                r3.metric("Sharpe Ratio", f"{res.sharpe_ratio:.2f}")
                r4.metric("Max Drawdown", f"{res.max_drawdown:.1f}%")
                
                # Equity Chart
                df_eq = pd.DataFrame(res.equity_curve)
                fig = px.line(df_eq, x="date", y=["equity", "benchmark"], 
                               title=f"{bt_ticker} Strategy vs Benchmark",
                               color_discrete_map={"equity": "#3b82f6", "benchmark": "#94a3b8"})
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(pd.DataFrame(res.trade_log), use_container_width=True)
            except Exception as e:
                st.error(f"Backtest error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PRICE ALERTS
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="section-header">🔔 Real-Time Price Alerts</div>', unsafe_allow_html=True)
    
    a1, a2, a3, a4 = st.columns([2, 2, 2, 3])
    with a1:
        at_ticker = st.selectbox("Ticker", ALL_TICKERS, key="at_ticker")
    with a2:
        at_type = st.selectbox("Type", ["PRICE_ABOVE", "PRICE_BELOW"])
    with a3:
        at_val = st.number_input("Trigger Price", value=150.0)
    with a4:
        at_msg = st.text_input("Custom Message", "Target reached!")
    
    if st.button("➕ Set Alert"):
        create_price_alert(DB_PATH, at_ticker, at_type, at_val, at_msg)
        st.success(f"Alert set for {at_ticker} {at_type} {at_val}")

    st.divider()
    
    # Check Alerts Button
    if st.button("⚡ Check Active Alerts Now"):
        triggered = check_all_alerts(DB_PATH)
        if triggered:
            for t in triggered:
                st.toast(t["message"], icon="🔔")
                st.warning(t["message"])
        else:
            st.info("No alerts triggered in this check.")

    # Active Alerts Table
    st.markdown("#### ⏳ Active Alerts")
    active_alerts = get_alerts(DB_PATH, active_only=True)
    if active_alerts:
        df_a = pd.DataFrame(active_alerts)[["ticker", "alert_type", "trigger_value", "created_at"]]
        st.dataframe(df_a, use_container_width=True)
    else:
        st.write("No active alerts.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — SENTIMENT HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown('<div class="section-header">🌡️ Global Sentiment Heatmap</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;'>Aggregated social sentiment from Reddit, News, and AI context.</p>", unsafe_allow_html=True)
    
    # Calculate sentiment for all tickers
    if st.button("📊 Update Sentiment Heatmap"):
        with st.spinner("Scanning social & news..."):
            for t in ALL_TICKERS:
                st.session_state.sentiment_tracker.get_combined_sentiment(t)
    
    # Fetch latest scores
    scores = get_sentiment_history(DB_PATH)
    if scores:
        df_s = pd.DataFrame(scores)
        # Get latest per ticker
        latest_s = df_s.sort_values("timestamp").groupby("ticker").last().reset_index()
        
        fig = px.treemap(latest_s, path=["ticker"], values=[1]*len(latest_s),
                         color="sentiment_score",
                         color_continuous_scale="RdYlGn",
                         range_color=[-100, 100],
                         title="Current Market Sentiment (Reddit + News)")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(latest_s[["ticker", "sentiment_score", "source_counts", "timestamp"]], use_container_width=True)
    else:
        st.info("No sentiment data yet. Click 'Update' to begin scanning.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — AI TRADING JOURNAL & COACH
# ══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.markdown('<div class="section-header">📔 AI Trading Journal & Coach</div>', unsafe_allow_html=True)
    
    coach_tab1, coach_tab2, coach_tab3 = st.tabs(["Self-Reflection", "Weekly Review", "Trading DNA"])
    
    with coach_tab1:
        st.markdown("#### 🤔 Why did you take that trade?")
        recent_trades = get_paper_trades(DB_PATH)
        if not recent_trades:
            st.info("No trades to reflect on.")
        else:
            t_ids = [f"{t['ticker']} ({t['entry_date'][:10]}) - {t['trade_id'][:8]}" for t in recent_trades]
            sel_t_idx = st.selectbox("Select Trade", range(len(t_ids)), format_func=lambda i: t_ids[i])
            sel_t = recent_trades[sel_t_idx]
            
            # Check if journal exists
            journals = get_journals(DB_PATH, trade_id=sel_t["trade_id"])
            if journals:
                st.markdown(f'<div class="rag-box">{journals[0]["content"]}</div>', unsafe_allow_html=True)
            else:
                if st.button("🤖 Generate Gemini Critique"):
                    with st.spinner("AI is analyzing your strategy..."):
                        res = st.session_state.ai_coach.generate_journal_entry(sel_t)
                        st.markdown(f'<div class="rag-box">{res}</div>', unsafe_allow_html=True)
    
    with coach_tab2:
        st.markdown("#### 📅 Weekly Performance Review")
        if st.button("📈 Run Weekly Analysis"):
            with st.spinner("Gemini is reviewing your week..."):
                res = st.session_state.ai_coach.generate_weekly_review()
                st.markdown(f'<div class="rag-box">{res}</div>', unsafe_allow_html=True)
        
        st.divider()
        prev_reviews = get_weekly_reviews(DB_PATH)
        if prev_reviews:
            for rev in reversed(prev_reviews):
                with st.expander(f"Review: {rev['week_start']} to {rev['week_end']}"):
                    st.write(rev["content"])

    with coach_tab3:
        st.markdown("#### 🧬 Trading DNA")
        dna = st.session_state.ai_coach.calculate_trading_dna()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Profile", dna.get("Risk", "N/A"))
        c2.metric("Edge Quality", f"{dna.get('Accuracy', 0):.1f}/100")
        c3.metric("Discipline Score", f"{dna.get('Discipline', 0):.1f}/100")
        
        # DNA Chart
        df_dna = pd.DataFrame({
            "Metric": ["Risk", "Edge", "Discipline", "Timing", "Sizing"],
            "Value": [dna["risk_score"], dna["edge_score"], dna["discipline_score"], 7.5, 6.0]
        })
        fig = px.line_polar(df_dna, r="Value", theta="Metric", line_close=True)
        fig.update_traces(fill="toself", fillcolor="rgba(59,130,246,0.3)")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — SIGNAL HISTORY
# ══════════════════════════════════════════════════════════════════════════════

with tab8:
    st.markdown('<div class="section-header">🗄️ Signal History — SQLite Store</div>', unsafe_allow_html=True)

    # Fetch all signals from DB
    all_signals = get_signals(DB_PATH, limit=500)

    if not all_signals:
        st.info("No signals in the database yet. Run ⚡ Fetch & Analyze to generate some.", icon="ℹ️")
    else:
        df_hist = pd.DataFrame(all_signals)
        
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
                key="hist_ticker_filt"
            )
        with f2:
            dir_options = sorted(df_display["Direction"].unique().tolist()) if "Direction" in df_display else []
            dir_filter = st.multiselect("Filter by Direction", options=dir_options, key="hist_dir_filt")
        with f3:
            date_filter = st.date_input(
                "From date",
                value=date.today() - timedelta(days=30),
                key="hist_date_filt"
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

        # Display
        display_cols = [c for c in ["Ticker", "Direction", "Confidence", "Position", "position_size_pct", "risk_reward_ratio",
                                     "RSI", "MA20", "Vol Δ%", "Model", "Generated At"]
                        if c in filtered.columns]
        
        st.dataframe(
            filtered[display_cols].sort_values("Generated At", ascending=False),
            use_container_width=True,
            height=420
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════

with tab9:
    st.markdown('<div class="section-header">❓ How the Alpha Pipeline Works</div>', unsafe_allow_html=True)
    
    # ── Pipeline diagram ──────────────────────────────────────────────────────
    steps = [
        ("📰", "News Fetch", "NewsAPI/yf"),
        ("🟣", "Sentiment", "Hybrid Score"),
        ("🗄️", "ChromaDB", "RAG Storage"),
        ("🤖", "Gemini 1.5", "Alpha Analyst"),
        ("📊", "Technical", "Signal Check"),
        ("💰", "Paper Trade", "Virtual Exec"),
        ("📔", "AI Coach", "DNA Critique"),
    ]

    pipe_cols = st.columns(len(steps))
    for col, (icon, name, detail) in zip(pipe_cols, steps):
        with col:
            st.markdown(
                f"<div class='pipeline-step'>"
                f"<div style='font-size:1.5rem;'>{icon}</div>"
                f"<div style='font-weight:700; color:#f1f5f9; font-size:0.8rem;'>{name}</div>"
                f"<div style='color:#64748b; font-size:0.7rem;'>{detail}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🔬 Hybrid Alpha Scoring")
        st.markdown("""
        The system uses a **weighted confidence score** (0-100%):
        - **40% News Sentiment**: Extracted via RAG and Hybrid Social Tracker.
        - **30% LLM Analysis**: Gemini's reasoning on catalysts and risks.
        - **30% Technical Filter**: Overbought/Oversold verification via RSI/MAs.
        """)
    with c2:
        st.markdown("#### 💼 Risk Management Engine")
        st.markdown("""
        Every AI signal is paired with strict risk parameters:
        - **Position Sizing**: Based on volatility and account ATR.
        - **Smart Stops**: Dynamic Stop-Loss based on technical support.
        - **Take Profit**: Calculated for an optimal Risk-Reward ratio (1:2+).
        """)
