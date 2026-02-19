
import logging
import yfinance as yf
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from src.database import (
    get_paper_portfolio, 
    save_paper_portfolio, 
    save_paper_trade, 
    get_paper_trades
)

logger = logging.getLogger(__name__)

@dataclass
class Position:
    ticker: str
    direction: str          # "LONG" or "SHORT"
    entry_price: float
    quantity: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_date: Optional[str] = None
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

class PaperPortfolio:
    def __init__(self, db_path: str):
        self.db_path = db_path
        data = get_paper_portfolio(db_path)
        self.cash = data["cash"]
        self.total_value = data["total_value"]
        self.positions: Dict[str, Position] = {}
        self._load_positions()

    def _load_positions(self):
        """Load open positions from database."""
        open_trades = get_paper_trades(self.db_path, outcome="OPEN")
        for t in open_trades:
            self.positions[t["ticker"]] = Position(
                ticker=t["ticker"],
                direction=t["direction"],
                entry_price=t["entry_price"],
                quantity=t["quantity"],
                stop_loss=t.get("stop_loss"),
                take_profit=t.get("take_profit"),
                entry_date=t["entry_date"]
            )

    def execute_trade(self, ticker: str, direction: str, position_size_pct: float, 
                      entry_price: float, stop_loss: float, take_profit: float,
                      ai_signal: str = None, ai_confidence: float = None) -> dict:
        """
        Execute a new paper trade.
        quantity = (portfolio_value * size_pct/100) / entry_price
        """
        if ticker in self.positions:
            return {"error": f"Position already open for {ticker}"}

        # Calculate quantity
        trade_value = self.total_value * (position_size_pct / 100.0)
        quantity = int(trade_value / entry_price)
        
        if quantity <= 0:
            return {"error": "Position size too small for 1 share"}

        cost = quantity * entry_price
        if direction == "LONG" and cost > self.cash:
            return {"error": "Insufficient cash for LONG position"}

        # For SHORT: record as negative quantity? 
        # Actually, let's keep quantity positive and rely on direction flag for PnL
        
        if direction == "LONG":
            self.cash -= cost
        else:
            # Shorting: add proceeds to cash (simplified)
            self.cash += cost

        trade_data = {
            "ticker": ticker.upper(),
            "direction": direction.upper(),
            "entry_price": entry_price,
            "quantity": quantity,
            "position_size_pct": position_size_pct,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_date": datetime.now(timezone.utc).isoformat(),
            "outcome": "OPEN",
            "ai_signal": ai_signal,
            "ai_confidence": ai_confidence
        }
        
        trade_id = save_paper_trade(self.db_path, trade_data)
        trade_data["trade_id"] = trade_id
        
        # Add to local tracking
        self.positions[ticker] = Position(
            ticker=ticker,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_date=trade_data["entry_date"]
        )
        
        self.save_state()
        return trade_data

    def update_positions(self) -> dict:
        """Fetch current prices and check SL/TP hits."""
        if not self.positions:
            self.save_state()
            return self.get_portfolio_summary()

        tickers = list(self.positions.keys())
        try:
            # Batch fetch prices
            data = yf.download(tickers, period="1d", interval="1m", progress=False)
            
            for ticker in tickers:
                pos = self.positions[ticker]
                # Handle single vs multiple tickers in download output
                if len(tickers) == 1:
                    current_price = data["Close"].iloc[-1]
                else:
                    current_price = data["Close"][ticker].iloc[-1]
                
                pos.current_price = float(current_price)
                
                # Calculate P&L
                mult = 1.0 if pos.direction == "LONG" else -1.0
                pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity * mult
                pos.unrealized_pnl_pct = ((pos.current_price / pos.entry_price) - 1.0) * 100.0 * mult
                
                # Check SL/TP
                if pos.direction == "LONG":
                    if pos.stop_loss and pos.current_price <= pos.stop_loss:
                        self.close_position(ticker, pos.current_price, "STOPPED_OUT")
                    elif pos.take_profit and pos.current_price >= pos.take_profit:
                        self.close_position(ticker, pos.current_price, "TARGET_HIT")
                else: # SHORT
                    if pos.stop_loss and pos.current_price >= pos.stop_loss:
                        self.close_position(ticker, pos.current_price, "STOPPED_OUT")
                    elif pos.take_profit and pos.current_price <= pos.take_profit:
                        self.close_position(ticker, pos.current_price, "TARGET_HIT")

        except Exception as e:
            logger.error(f"Error updating positions: {e}")

        self.save_state()
        return self.get_portfolio_summary()

    def close_position(self, ticker: str, current_price: float, reason: str = "MANUAL") -> dict:
        """Realize P&L and move to history."""
        if ticker not in self.positions:
            return {"error": "No open position for ticker"}

        pos = self.positions.pop(ticker)
        mult = 1.0 if pos.direction == "LONG" else -1.0
        realized_pnl = (current_price - pos.entry_price) * pos.quantity * mult
        realized_pnl_pct = ((current_price / pos.entry_price) - 1.0) * 100.0 * mult
        
        # Return cash
        if pos.direction == "LONG":
            self.cash += (pos.quantity * current_price)
        else:
            # For SHORT: proceeds were added at entry. 
            # Now we buy back: cash cost = qty * current_price. 
            # Initial proceeds were qty * entry_price.
            # Realized change to cash = (entry - current) * qty
            # Since we added proceeds initially, we just subtract the buyback cost
            self.cash -= (pos.quantity * current_price)

        # Update DB
        trade_logs = get_paper_trades(self.db_path, outcome="OPEN")
        trade = next((t for t in trade_logs if t["ticker"] == ticker), None)
        if trade:
            trade.update({
                "exit_price": current_price,
                "exit_date": datetime.now(timezone.utc).isoformat(),
                "realized_pnl": realized_pnl,
                "realized_pnl_pct": realized_pnl_pct,
                "outcome": "WIN" if realized_pnl > 0 else "LOSS",
                "exit_reason": reason
            })
            save_paper_trade(self.db_path, trade)

        self.save_state()
        return {"ticker": ticker, "pnl": realized_pnl, "reason": reason}

    def save_state(self):
        """Calculate total value and persist portfolio state."""
        pos_value = 0.0
        for pos in self.positions.values():
            if pos.direction == "LONG":
                pos_value += (pos.quantity * pos.current_price)
            else:
                # SHORT equity value = (Entry - Current) * Qty + Initial Margin? 
                # Simpler: Total Value = Cash + Sum(Unrealized PnL)
                pos_value += pos.unrealized_pnl
        
        # In our simplified short model (cash += entry_value), cash already contains short proceeds.
        # So Total Value = Cash + Long_Value - Buyback_Cost_of_Shorts
        # Wait, if Cash += Entry_Value for Shorts, then Total Value = Cash - Current_Value_of_Shorts + Long_Value
        
        self.total_value = self.cash
        for pos in self.positions.values():
            if pos.direction == "LONG":
                self.total_value += (pos.quantity * (pos.current_price - pos.entry_price))
                # Wait, total value should be cash + current_market_value
                # If cash was subtracted at entry: Total Value = Cash + sum(current_price * qty)
            else:
                # If cash was added at entry: Total Value = Cash - sum(current_price * qty)
                pass

        # Let's recalibrate: Total Value = Cash + Unrealized P&L?
        # No, that's not right. 
        # Correct logic:
        # Starting: Cash=100k, Value=100k
        # Buy 10k AAPL: Cash=90k, Position Value=10k, Total Value = 100k.
        # AAPL goes to 11k: Total Value = 90k + 11k = 101k.
        
        equity = 0.0
        for pos in self.positions.values():
            if pos.direction == "LONG":
                equity += (pos.quantity * (pos.current_price or pos.entry_price))
            else:
                # Short: you owe the current market value.
                equity -= (pos.quantity * (pos.current_price or pos.entry_price))
        
        self.total_value = self.cash + equity
        save_paper_portfolio(self.db_path, self.cash, self.total_value)

    def get_portfolio_summary(self) -> dict:
        """Calculate and return performance metrics."""
        history = get_paper_trades(self.db_path)
        closed = [t for t in history if t["outcome"] != "OPEN"]
        wins = [t for t in closed if t["outcome"] == "WIN"]
        losses = [t for t in closed if t["outcome"] == "LOSS"]
        
        total_value = self.total_value
        return_pct = (total_value - 100000.0) / 1000.0 # (val - 100k)/100k * 100
        
        summary = {
            "total_value": total_value,
            "cash": self.cash,
            "return_pct": return_pct,
            "open_positions": len(self.positions),
            "total_trades": len(closed),
            "win_rate": (len(wins) / len(closed) * 100) if closed else 0,
            "avg_win_pct": (sum(t["realized_pnl_pct"] for t in wins) / len(wins)) if wins else 0,
            "avg_loss_pct": (sum(t["realized_pnl_pct"] for t in losses) / len(losses)) if losses else 0,
            "profit_factor": (abs(sum(t["realized_pnl"] for t in wins)) / abs(sum(t["realized_pnl"] for t in losses))) if losses and sum(t["realized_pnl"] for t in losses) != 0 else 1.0,
            "best_trade": max([t["realized_pnl_pct"] for t in closed]) if closed else 0,
            "worst_trade": min([t["realized_pnl_pct"] for t in closed]) if closed else 0,
            "sharpe_ratio": 1.5 # Placeholder for now
        }
        return summary
