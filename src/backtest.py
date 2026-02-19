"""
src/backtest.py
1-Year Signal Backtest Engine

Strategy
--------
- Universe : configurable tickers (default AAPL, TSLA, NVDA, MSFT, GOOGL)
- Signal   : rule-based from RSI(14) + MA-crossover (proxies for alpha engine)
             Strong Buy  → full long   (+1.0 weight)
             Buy         → half long   (+0.5 weight)
             Neutral     → flat        (0.0)
             Sell        → half short  (−0.5 weight) [long-only: flat]
             Strong Sell → full short  (−1.0 weight) [long-only: flat]
- Execution: daily rebalance, no transaction costs (first pass), 2nd pass with 0.1% per trade
- Benchmark: Buy-and-hold equal-weight portfolio of same tickers

Metrics returned
----------------
total_return, cagr, sharpe_ratio, sortino_ratio,
max_drawdown, calmar_ratio, win_rate, avg_win, avg_loss,
best_day, worst_day, volatility_ann, benchmark_total_return,
benchmark_sharpe, alpha (vs benchmark)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.05          # 5% annual (approx. 1-yr T-bill 2024)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION  (rule-based proxy for AlphaEngine)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss  = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def generate_signals(price_df: pd.DataFrame) -> pd.Series:
    """
    Rule-based signal from RSI(14) + MA20/MA50 crossover.

    Returns a Series with values in {-1, -0.5, 0, 0.5, 1} indexed by date.
    """
    close = price_df["Close"]
    rsi   = _compute_rsi(close, 14)
    ma20  = close.rolling(20, min_periods=1).mean()
    ma50  = close.rolling(50, min_periods=1).mean()

    signal = pd.Series(0.0, index=close.index)

    for i in range(len(close)):
        r  = rsi.iloc[i]
        m20 = ma20.iloc[i]
        m50 = ma50.iloc[i]

        # Strong Buy: oversold + golden cross
        if r < 30 and m20 > m50:
            signal.iloc[i] = 1.0

        # Buy: mildly oversold OR MA trending up
        elif r < 40 or m20 > m50 * 1.005:
            signal.iloc[i] = 0.5

        # Strong Sell: overbought + death cross
        elif r > 70 and m20 < m50:
            signal.iloc[i] = -1.0

        # Sell: mildly overbought OR MA trending down
        elif r > 60 or m20 < m50 * 0.995:
            signal.iloc[i] = -0.5

        else:
            signal.iloc[i] = 0.0

    return signal


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate_portfolio(
    prices: dict[str, pd.DataFrame],
    long_only: bool = True,
    transaction_cost: float = 0.001,   # 0.1% per trade
) -> pd.DataFrame:
    """
    Simulate a signal-driven portfolio.

    Parameters
    ----------
    prices          : dict of ticker → OHLCV DataFrame (daily, 1yr)
    long_only       : if True, negative signals are treated as flat (hold cash)
    transaction_cost: fraction of trade value charged per position change

    Returns
    -------
    DataFrame with columns: date, portfolio_value, daily_return, benchmark_value
    """
    # Build aligned close-price matrix
    close_dict = {}
    for ticker, df in prices.items():
        close_dict[ticker] = df["Close"].rename(ticker)
    closes = pd.DataFrame(close_dict).dropna(how="all").sort_index()
    closes = closes.fillna(method="ffill").dropna()

    tickers = list(closes.columns)
    n = len(tickers)

    # Signal matrix (same shape as closes)
    sig_dict = {}
    for ticker, df in prices.items():
        raw_sig = generate_signals(df.reindex(closes.index).fillna(method="ffill"))
        if long_only:
            raw_sig = raw_sig.clip(lower=0)   # never short
        sig_dict[ticker] = raw_sig
    signals = pd.DataFrame(sig_dict, index=closes.index)

    # Normalise weights per row so they sum to 1 (equal-weight among active positions)
    weight_sum = signals.abs().sum(axis=1).replace(0, np.nan)
    weights = signals.div(weight_sum, axis=0).fillna(0)

    # Daily returns of each stock
    stock_returns = closes.pct_change().fillna(0)

    # Portfolio daily return = weighted sum of stock returns
    portfolio_daily = (weights.shift(1).fillna(0) * stock_returns).sum(axis=1)

    # Transaction costs: charge on weight changes
    weight_changes = weights.diff().abs().sum(axis=1).fillna(0)
    tc_drag = weight_changes * transaction_cost
    portfolio_daily -= tc_drag

    # Portfolio value (start = 100,000)
    start_value = 100_000.0
    portfolio_value = (1 + portfolio_daily).cumprod() * start_value

    # Benchmark: equal-weight buy-and-hold (no rebalancing)
    bm_returns = stock_returns.mean(axis=1)
    benchmark_value = (1 + bm_returns).cumprod() * start_value

    result = pd.DataFrame({
        "date":             closes.index,
        "portfolio_value":  portfolio_value.values,
        "daily_return":     portfolio_daily.values,
        "benchmark_value":  benchmark_value.values,
        "bm_daily_return":  bm_returns.values,
    })
    result["drawdown"]    = _compute_drawdown(result["portfolio_value"])
    result["bm_drawdown"] = _compute_drawdown(result["benchmark_value"])
    return result.reset_index(drop=True)


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    # Strategy
    total_return:          float = 0.0   # e.g. 0.234 = 23.4%
    cagr:                  float = 0.0
    sharpe_ratio:          float = 0.0
    sortino_ratio:         float = 0.0
    max_drawdown:          float = 0.0   # negative e.g. -0.15
    calmar_ratio:          float = 0.0
    volatility_ann:        float = 0.0
    win_rate:              float = 0.0
    avg_win:               float = 0.0
    avg_loss:              float = 0.0
    best_day:              float = 0.0
    worst_day:             float = 0.0
    total_trades:          int   = 0

    # Benchmark
    bm_total_return:       float = 0.0
    bm_sharpe:             float = 0.0
    bm_max_drawdown:       float = 0.0
    bm_volatility_ann:     float = 0.0

    # Alpha
    alpha_vs_benchmark:    float = 0.0   # excess return over benchmark
    beta:                  float = 0.0

    # Time series (for plotting)
    equity_curve:          pd.Series = field(default_factory=pd.Series)
    drawdown_curve:        pd.Series = field(default_factory=pd.Series)
    bm_equity_curve:       pd.Series = field(default_factory=pd.Series)
    bm_drawdown_curve:     pd.Series = field(default_factory=pd.Series)
    dates:                 pd.Series = field(default_factory=pd.Series)
    monthly_returns:       pd.DataFrame = field(default_factory=pd.DataFrame)


def compute_metrics(sim: pd.DataFrame) -> BacktestResult:
    """Compute all performance metrics from simulation output DataFrame."""
    result = BacktestResult()
    n_days = len(sim)
    years  = n_days / TRADING_DAYS

    dr  = sim["daily_return"].values
    bdr = sim["bm_daily_return"].values

    # ── Strategy ─────────────────────────────────────────────────────────────
    port_val = sim["portfolio_value"]
    result.total_return = float(port_val.iloc[-1] / port_val.iloc[0] - 1)
    result.cagr         = float((1 + result.total_return) ** (1 / max(years, 0.001)) - 1)

    rf_daily         = RISK_FREE_RATE / TRADING_DAYS
    excess_daily     = dr - rf_daily
    result.volatility_ann = float(np.std(dr, ddof=1) * math.sqrt(TRADING_DAYS))

    mean_excess = np.mean(excess_daily)
    std_excess  = np.std(excess_daily, ddof=1)
    result.sharpe_ratio = float(
        (mean_excess * TRADING_DAYS) / (std_excess * math.sqrt(TRADING_DAYS))
        if std_excess > 0 else 0.0
    )

    downside = excess_daily[excess_daily < 0]
    sortino_denom = np.std(downside, ddof=1) * math.sqrt(TRADING_DAYS) if len(downside) > 1 else 1e-9
    result.sortino_ratio = float(mean_excess * TRADING_DAYS / sortino_denom)

    result.max_drawdown = float(sim["drawdown"].min())
    result.calmar_ratio = float(result.cagr / abs(result.max_drawdown)) if result.max_drawdown != 0 else 0.0

    wins  = dr[dr > 0]
    losses = dr[dr < 0]
    result.win_rate  = float(len(wins) / max(len(dr), 1))
    result.avg_win   = float(np.mean(wins))   if len(wins)   > 0 else 0.0
    result.avg_loss  = float(np.mean(losses)) if len(losses) > 0 else 0.0
    result.best_day  = float(np.max(dr))
    result.worst_day = float(np.min(dr))

    # ── Benchmark ─────────────────────────────────────────────────────────────
    bm_val = sim["benchmark_value"]
    result.bm_total_return = float(bm_val.iloc[-1] / bm_val.iloc[0] - 1)
    bm_excess              = bdr - rf_daily
    bm_std                 = np.std(bdr, ddof=1) * math.sqrt(TRADING_DAYS)
    result.bm_sharpe       = float(
        (np.mean(bm_excess) * TRADING_DAYS) / bm_std if bm_std > 0 else 0.0
    )
    result.bm_max_drawdown   = float(sim["bm_drawdown"].min())
    result.bm_volatility_ann = float(bm_std)

    # ── Alpha / Beta ──────────────────────────────────────────────────────────
    result.alpha_vs_benchmark = result.total_return - result.bm_total_return
    cov_matrix = np.cov(dr, bdr)
    result.beta = float(cov_matrix[0, 1] / cov_matrix[1, 1]) if cov_matrix[1, 1] != 0 else 1.0

    # ── Time series ───────────────────────────────────────────────────────────
    result.equity_curve    = sim["portfolio_value"]
    result.drawdown_curve  = sim["drawdown"]
    result.bm_equity_curve = sim["benchmark_value"]
    result.bm_drawdown_curve = sim["bm_drawdown"]
    result.dates = pd.to_datetime(sim["date"]) if "date" in sim.columns else sim.index

    # ── Monthly returns heatmap data ──────────────────────────────────────────
    try:
        indexed = sim.set_index(pd.to_datetime(sim["date"]))["daily_return"]
        monthly = indexed.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        df_monthly = monthly.reset_index()
        df_monthly.columns = ["date", "return"]
        df_monthly["year"]  = df_monthly["date"].dt.year
        df_monthly["month"] = df_monthly["date"].dt.strftime("%b")
        result.monthly_returns = df_monthly
    except Exception:
        pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    tickers: list[str],
    period: str = "1y",
    long_only: bool = True,
    transaction_cost: float = 0.001,
) -> tuple[BacktestResult, pd.DataFrame]:
    """
    Full pipeline: fetch → signal → simulate → metrics.

    Returns (BacktestResult, simulation_df)
    """
    import yfinance as yf

    prices: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            # Download single ticker to avoid MultiIndex column issues
            df = yf.download(ticker, period=period, auto_adjust=True,
                             progress=False, multi_level_index=False)
            # Fallback: some yfinance versions still return MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) > 50:
                prices[ticker] = df
                logger.info("Fetched %d rows for %s", len(df), ticker)
            else:
                logger.warning("Insufficient data for %s, skipping", ticker)
        except Exception as exc:
            logger.error("Failed to fetch %s: %s", ticker, exc)

    if not prices:
        raise ValueError("No price data fetched — check tickers or network.")

    sim = simulate_portfolio(prices, long_only=long_only, transaction_cost=transaction_cost)
    metrics = compute_metrics(sim)
    return metrics, sim

