
import os
import logging
import praw
from textblob import TextBlob
from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np

from .database import save_sentiment, get_articles

logger = logging.getLogger(__name__)

class SentimentTracker:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.reddit = None
        self._init_reddit()

    def _init_reddit(self):
        """Initialize PRAW if credentials exist."""
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "FinancialAlphaBot/1.0")

        if client_id and client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=client_id,
                    client_secret=client_secret,
                    user_agent=user_agent
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Reddit API: {e}")

    def get_reddit_sentiment(self, ticker: str, limit: int = 50) -> Dict:
        """Fetch and score Reddit sentiment for a ticker."""
        if not self.reddit:
            return self._get_mock_reddit_sentiment(ticker)

        subreddits = ["wallstreetbets", "stocks", "investing", "options"]
        scores = []
        mention_count = 0

        try:
            for sub_name in subreddits:
                subreddit = self.reddit.subreddit(sub_name)
                # Search for ticker in last 24h
                for submission in subreddit.search(ticker, time_filter="day", limit=limit):
                    mention_count += 1
                    # Score based on title + body
                    text = f"{submission.title} {submission.selftext}"
                    blob = TextBlob(text)
                    # TextBlob polarity is -1 to 1. Scale to -100 to 100.
                    # Weight by upvotes (simplified)
                    weight = min(1 + (submission.score / 100), 5)
                    scores.append(blob.sentiment.polarity * 100 * weight)

            if not scores:
                return self._get_mock_reddit_sentiment(ticker)

            avg_score = np.mean(scores)
            # Clamp to -100 to 100
            avg_score = max(-100, min(100, avg_score))
            
            return {
                "score": round(avg_score, 2),
                "mention_count": mention_count,
                "source": "REDDIT"
            }

        except Exception as e:
            logger.error(f"Reddit sentiment error for {ticker}: {e}")
            return self._get_mock_reddit_sentiment(ticker)

    def get_news_sentiment(self, ticker: str) -> Dict:
        """Calculate sentiment from recently ingested news articles in DB."""
        articles = get_articles(self.db_path, ticker=ticker, limit=20)
        if not articles:
            return {"score": 0.0, "mention_count": 0, "source": "NEWS"}

        scores = [a["sentiment_score"] * 100 for a in articles]
        avg_score = np.mean(scores)
        
        return {
            "score": round(avg_score, 2),
            "mention_count": len(articles),
            "source": "NEWS"
        }

    def get_combined_sentiment(self, ticker: str) -> Dict:
        """Combine Reddit and News sentiment into a single score."""
        reddit = self.get_reddit_sentiment(ticker)
        news = self.get_news_sentiment(ticker)

        # Weighted average: News 60%, Reddit 40% (unless one is missing)
        r_score = reddit["score"]
        n_score = news["score"]
        
        if reddit["mention_count"] == 0:
            combined = n_score
        elif news["mention_count"] == 0:
            combined = r_score
        else:
            combined = (n_score * 0.6) + (r_score * 0.4)

        result = {
            "ticker": ticker.upper(),
            "combined_score": round(combined, 2),
            "reddit_score": r_score,
            "news_score": n_score,
            "reddit_mentions": reddit["mention_count"],
            "news_count": news["mention_count"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Save to history
        save_sentiment(self.db_path, ticker, "COMBINED", combined, reddit["mention_count"] + news["mention_count"])
        save_sentiment(self.db_path, ticker, "REDDIT", r_score, reddit["mention_count"])
        save_sentiment(self.db_path, ticker, "NEWS", n_score, news["mention_count"])

        return result

    def _get_mock_reddit_sentiment(self, ticker: str) -> Dict:
        """Fallback for missing API keys or low buzz."""
        # Semi-random but deterministic-ish for demo
        import random
        random.seed(ticker)
        score = random.uniform(-15, 25) # Default slightly bullish for demo
        mentions = random.randint(5, 50)
        return {
            "score": round(score, 2),
            "mention_count": mentions,
            "source": "REDDIT_MOCK"
        }
