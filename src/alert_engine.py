
import logging
import yfinance as yf
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dataclasses import dataclass

from .database import (
    save_alert,
    get_alerts,
    delete_alert
)

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    alert_id: str
    ticker: str
    alert_type: str        # PRICE_ABOVE, PRICE_BELOW, RSI_OVERBOUGHT, RSI_OVERSOLD, VOL_SPIKE
    trigger_value: float
    message: str
    is_active: bool = True
    created_at: str = None
    triggered_at: Optional[str] = None
    times_triggered: int = 0

def create_price_alert(db_path: str, ticker: str, alert_type: str, value: float, message: str) -> str:
    """Create a new user alert."""
    alert_data = {
        "ticker": ticker.upper(),
        "alert_type": alert_type,
        "trigger_value": value,
        "message": message,
        "is_active": 1,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    return save_alert(db_path, alert_data)

def check_all_alerts(db_path: str) -> List[dict]:
    """
    Fetch all active alerts, check against current market data, 
    and return list of triggered alerts.
    """
    active_alerts = get_alerts(db_path, active_only=True)
    if not active_alerts:
        return []

    triggered = []
    tickers = list(set(a["ticker"] for a in active_alerts))
    
    try:
        # Fetch current prices
        # period="1d" interval="1m" gives latest minute candle
        data = yf.download(tickers, period="1d", interval="1m", progress=False)
        
        for alert in active_alerts:
            ticker = alert["ticker"]
            if len(tickers) == 1:
                curr_price = float(data["Close"].iloc[-1])
            else:
                curr_price = float(data["Close"][ticker].iloc[-1])
            
            is_triggered = False
            
            # Logic for different alert types
            if alert["alert_type"] == "PRICE_ABOVE" and curr_price >= alert["trigger_value"]:
                is_triggered = True
            elif alert["alert_type"] == "PRICE_BELOW" and curr_price <= alert["trigger_value"]:
                is_triggered = True
            # Note: RSI/Volume alerts would require more history, skipping for MVP
            
            if is_triggered:
                alert["is_active"] = 0
                alert["triggered_at"] = datetime.now(timezone.utc).isoformat()
                alert["times_triggered"] += 1
                save_alert(db_path, alert)
                
                triggered_info = {
                    "alert_id": alert["alert_id"],
                    "ticker": ticker,
                    "message": f"🚨 ALERT: {ticker} {alert['alert_type'].replace('_', ' ')} {alert['trigger_value']} (Current: {curr_price:.2f})",
                    "time": alert["triggered_at"]
                }
                triggered.append(triggered_info)

    except Exception as e:
        logger.error(f"Error checking alerts: {e}")

    return triggered
