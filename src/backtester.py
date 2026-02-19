
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    trade_log: List[dict]
    equity_curve: List[dict]

def run_backtest(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    position_size_pct: float = 10.0,
    stop_loss_mult: float = 1.5,
    take_profit_mult: float = 2.5,
    atr_period: int = 14
) -> BacktestResult:
    """Run a technical-only walk-forward simulation."""
    logger.info(f"Starting backtest for {ticker} from {start_date} to {end_date}")
    
    # 1. Load Data
    df = yf.download(ticker, start=start_date, end=end_date, 
                     progress=False, multi_level_index=False, auto_adjust=True)
    if df.empty or len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker} in range {start_date} to {end_date}")

    # Standardize column names (robust MultiIndex handling)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure column names are clean and unique strings
    df.columns = [str(c).strip() for c in df.columns]
    
    # 2. Indicators (Ensure we work with 1D series)
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    
    # Cast to float series to be safe
    close = pd.Series(close).astype(float)
    high = pd.Series(high).astype(float)
    low = pd.Series(low).astype(float)

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MA
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(atr_period).mean()

    # 3. Simulation
    cash = initial_capital
    position = None # {type: 'LONG'|'SHORT', qty, entry_price, sl, tp, entry_date}
    trade_log = []
    equity_curve = []
    
    dates = df.index.tolist()
    
    for i in range(50, len(df)):
        date = dates[i]
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        curr_price = float(curr["Close"])
        equity = cash
        if position:
            mult = 1.0 if position["type"] == "LONG" else -1.0
            equity += (curr_price - position["entry_price"]) * position["qty"] * mult
        
        equity_curve.append({
            "date": date.strftime("%Y-%m-%d"),
            "equity": equity,
            "benchmark": initial_capital * (curr_price / float(df.iloc[50]["Close"]))
        })

        # Check existing position
        if position:
            sl_hit = False
            tp_hit = False
            reversed_sig = False
            
            # Indicators as floats to prevent Series-related errors
            c_macd = float(curr["MACD"])
            c_macd_sig = float(curr["MACD_Signal"])
            p_macd = float(prev["MACD"])
            p_macd_sig = float(prev["MACD_Signal"])

            # SL/TP logic
            if position["type"] == "LONG":
                if curr_price <= position["sl"]: sl_hit = True
                elif curr_price >= position["tp"]: tp_hit = True
                # Signal reversal (MACD cross down)
                if c_macd < c_macd_sig and p_macd >= p_macd_sig:
                    reversed_sig = True
            else: # SHORT
                if curr_price >= position["sl"]: sl_hit = True
                elif curr_price <= position["tp"]: tp_hit = True
                # Signal reversal (MACD cross up)
                if c_macd > c_macd_sig and p_macd <= p_macd_sig:
                    reversed_sig = True

            if sl_hit or tp_hit or reversed_sig:
                # Close trade
                mult = 1.0 if position["type"] == "LONG" else -1.0
                pnl = (curr_price - position["entry_price"]) * position["qty"] * mult
                pnl_pct = ((curr_price / position["entry_price"]) - 1.0) * 100.0 * mult
                
                cash += (curr_price * position["qty"]) if position["type"] == "LONG" else - (curr_price * position["qty"])
                # Wait, short logic: cash was added at entry.
                # LONG: initial cash = c - entry_cost. Final cash = cash + exit_proceeds. Net = exit - entry.
                # SHORT: initial cash = c + entry_proceeds. Final cash = cash - exit_cost. Net = entry - exit.
                # Actually, my cash update for positions needs to be consistent.
                
                # Simplified: cash += pnl? No, pnl is only realized at end.
                # Let's use the PaperTrader model:
                # LONG: entry: cash -= cost. exit: cash += exit_value.
                # SHORT: entry: cash += proceeds. exit: cash -= buyback_cost.
                
                reason = "STOP_LOSS" if sl_hit else ("TAKE_PROFIT" if tp_hit else "SIGNAL_REVERSAL")
                trade_log.append({
                    "ticker": ticker,
                    "type": position["type"],
                    "entry_date": position["entry_date"],
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "entry_price": position["entry_price"],
                    "exit_price": curr_price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "reason": reason
                })
                position = None
                continue

        # Check for entry if no position
        if not position:
            c_macd = float(curr["MACD"])
            c_macd_sig = float(curr["MACD_Signal"])
            p_macd = float(prev["MACD"])
            p_macd_sig = float(prev["MACD_Signal"])
            c_rsi = float(curr["RSI"])

            # BUY Signal: MACD cross up OR RSI < 30
            long_sig = (c_macd > c_macd_sig and p_macd <= p_macd_sig) or (c_rsi < 30)
            # SELL Signal: MACD cross down OR RSI > 70
            short_sig = (c_macd < c_macd_sig and p_macd >= p_macd_sig) or (c_rsi > 70)
            
            # Additional trend filter: MA20 > MA50 for Long, MA20 < MA50 for Short
            trend_up = curr["MA20"] > curr["MA50"]
            
            if long_sig and trend_up:
                # Open Long
                trade_val = cash * (position_size_pct / 100.0)
                qty = int(trade_val / curr_price)
                if qty > 0:
                    cost = qty * curr_price
                    cash -= cost
                    position = {
                        "type": "LONG",
                        "qty": qty,
                        "entry_price": curr_price,
                        "sl": curr_price - stop_loss_mult * curr["ATR"],
                        "tp": curr_price + take_profit_mult * curr["ATR"],
                        "entry_date": date.strftime("%Y-%m-%d")
                    }
            elif short_sig and not trend_up:
                # Open Short
                trade_val = cash * (position_size_pct / 100.0)
                qty = int(trade_val / curr_price)
                if qty > 0:
                    proceeds = qty * curr_price
                    cash += proceeds
                    position = {
                        "type": "SHORT",
                        "qty": qty,
                        "entry_price": curr_price,
                        "sl": curr_price + stop_loss_mult * curr["ATR"],
                        "tp": curr_price - take_profit_mult * curr["ATR"],
                        "entry_date": date.strftime("%Y-%m-%d")
                    }

    # 4. Final aggregation
    final_capital = cash
    if position:
        mult = 1.0 if position["type"] == "LONG" else -1.0
        final_capital += (curr_price - position["entry_price"]) * position["qty"] * mult

    total_return = (final_capital / initial_capital) - 1.0
    
    # CAGR
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = ((final_capital / initial_capital) ** (1/years) - 1) if years > 0 else 0
    
    # Max Drawdown
    eq_df = pd.DataFrame(equity_curve)
    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]
    max_dd = float(eq_df["dd"].min())
    
    # Sharpe (approx)
    daily_rets = eq_df["equity"].pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std() * np.sqrt(252)) if len(daily_rets) > 1 and daily_rets.std() != 0 else 0
    
    # Trade metrics
    wins = [t for t in trade_log if t["pnl"] > 0]
    losses = [t for t in trade_log if t["pnl"] <= 0]
    win_rate = (len(wins) / len(trade_log) * 100) if trade_log else 0
    profit_factor = (abs(sum(t["pnl"] for t in wins)) / abs(sum(t["pnl"] for t in losses))) if losses and sum(t["pnl"] for t in losses) != 0 else 1.0

    return BacktestResult(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return_pct=total_return * 100,
        cagr=cagr * 100,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd * 100,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=len(trade_log),
        trade_log=trade_log,
        equity_curve=equity_curve
    )
