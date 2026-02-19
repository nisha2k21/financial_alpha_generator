
import os
import logging
import json
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import numpy as np

from .database import (
    get_paper_trades,
    save_journal_entry,
    save_weekly_review,
    get_journals
)

logger = logging.getLogger(__name__)

class AICoach:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    def generate_journal_entry(self, trade: dict) -> str:
        """Call Gemini to analyze a closed trade and generate a journal entry."""
        if not self.gemini_key:
            return self._get_mock_journal(trade)

        prompt = f"""
        Analyze the following closed virtual trade and provide a professional trading journal entry.
        
        Ticker: {trade['ticker']}
        Direction: {trade['direction']}
        Entry Price: {trade['entry_price']}
        Exit Price: {trade['exit_price']}
        P&L: {trade['realized_pnl']} ({trade['realized_pnl_pct']}%)
        Reason for Exit: {trade['exit_reason']}
        AI Signal at Entry: {trade.get('ai_signal', 'N/A')}
        
        Provide the analysis in this format:
        ### 📝 Trade Analysis: {trade['ticker']}
        **Outcome:** {'✅ Profit' if trade['realized_pnl'] > 0 else '❌ Loss'}
        
        **What went well:**
        - [Point 1]
        
        **Areas for improvement:**
        - [Point 1]
        
        **Key Lesson:**
        [1-2 sentences]
        
        **Coach's Verdict:**
        [A quick feedback from an AI coach]
        """

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=self.gemini_key)
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            
            save_journal_entry(self.db_path, {
                "trade_id": trade["trade_id"],
                "ticker": trade["ticker"],
                "content": content
            })
            return content
        except Exception as e:
            logger.error(f"Gemini journal error: {e}")
            return self._get_mock_journal(trade)

    def generate_weekly_review(self) -> str:
        """Analyze all trades from the last 7 days and provide a summary report."""
        trades = get_paper_trades(self.db_path)
        # Filter last 7 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        recent_trades = [t for t in trades if t["exit_date"] and datetime.fromisoformat(t["exit_date"]) > cutoff]
        
        if not recent_trades:
            return "No trades closed in the last 7 days to review."

        stats = self._calculate_stats(recent_trades)
        
        if not self.gemini_key:
            return f"Weekly Review: {len(recent_trades)} trades, Win Rate: {stats['win_rate']:.1f}%"

        prompt = f"""
        Provide a weekly performance review for a trader based on these stats:
        Total Trades: {stats['total_trades']}
        Win Rate: {stats['win_rate']:.1f}%
        Total P&L: {stats['total_pnl']:.2f}
        Avg Win %: {stats['avg_win_pct']:.2f}%
        Avg Loss %: {stats['avg_loss_pct']:.2f}%
        
        Focus on psychological patterns, technical consistency, and risk management.
        """

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=self.gemini_key)
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            
            save_weekly_review(self.db_path, {
                "start_date": cutoff.isoformat(),
                "end_date": datetime.now(timezone.utc).isoformat(),
                "content": content
            })
            return content
        except Exception as e:
            logger.error(f"Gemini weekly review error: {e}")
            return f"Review generated with errors. Total Trades: {len(recent_trades)}"

    def calculate_trading_dna(self) -> Dict[str, float]:
        """
        Calculate 'Trading DNA' metrics for radar chart:
        - Risk Management
        - Discipline
        - Logic (AI Alignment)
        - Adaptability
        - Accuracy
        """
        trades = get_paper_trades(self.db_path)
        if not trades:
            return {"Risk": 50, "Discipline": 50, "Logic": 50, "Adaptability": 50, "Accuracy": 50}

        closed = [t for t in trades if t["outcome"] != "OPEN"]
        
        # Accuracy: Win Rate
        wins = [t for t in closed if t["outcome"] == "WIN"]
        accuracy = (len(wins) / len(closed) * 100) if closed else 50
        
        # Risk: Average profit vs loss
        avg_win = np.mean([t["realized_pnl_pct"] for t in wins]) if wins else 0
        losses = [t for t in closed if t["outcome"] == "LOSS"]
        avg_loss = abs(np.mean([t["realized_pnl_pct"] for t in losses])) if losses else 0
        risk_score = min((avg_win / (avg_loss or 1)) * 40, 100) if closed else 50
        
        # Discipline: Percentage of trades that hit TP/SL vs Manual
        system_exits = [t for t in closed if t["exit_reason"] in ("STOPPED_OUT", "TARGET_HIT")]
        discipline = (len(system_exits) / len(closed) * 100) if closed else 50
        
        # Logic: Alignment with AI Signals
        aligned = [t for t in closed if (t["direction"] == "LONG" and t["ai_signal"] in ("BUY", "STRONG_BUY")) or (t["direction"] == "SHORT" and t["ai_signal"] in ("SELL", "STRONG_SELL"))]
        logic = (len(aligned) / len(closed) * 100) if closed else 50
        
        # Adaptability: placeholder
        adaptability = 70 if len(closed) > 5 else 50
        
        return {
            "Risk": round(risk_score, 1),
            "Discipline": round(discipline, 1),
            "Logic": round(logic, 1),
            "Adaptability": round(adaptability, 1),
            "Accuracy": round(accuracy, 1)
        }

    def _calculate_stats(self, trades: List[dict]) -> dict:
        wins = [t for t in trades if t["outcome"] == "WIN"]
        losses = [t for t in trades if t["outcome"] == "LOSS"]
        return {
            "total_trades": len(trades),
            "win_rate": (len(wins) / len(trades) * 100) if trades else 0,
            "total_pnl": sum(t["realized_pnl"] for t in trades),
            "avg_win_pct": np.mean([t["realized_pnl_pct"] for t in wins]) if wins else 0,
            "avg_loss_pct": np.mean([t["realized_pnl_pct"] for t in losses]) if losses else 0
        }

    def _get_mock_journal(self, trade: dict) -> str:
        return f"""
        ### 📝 Trade Analysis: {trade['ticker']}
        **Outcome:** {'✅ Profit' if trade['realized_pnl'] > 0 else '❌ Loss'}
        
        [Demo Mode - Gemini API Key not set]
        The trade for {trade['ticker']} resulted in a {trade['realized_pnl_pct']:.2f}% return. 
        The exit reason was {trade['exit_reason']}.
        """
