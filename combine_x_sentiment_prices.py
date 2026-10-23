import pandas as pd
from pathlib import Path

prices = pd.read_csv("data/bitcoin_prices.csv")
sentiment = pd.read_csv("data/x_posts_sentiment.csv")

prices.columns = prices.columns.str.lower().str.strip()
sentiment.columns = sentiment.columns.str.lower().str.strip()

price_date = "date" if "date" in prices.columns else "timestamp"
sentiment_date = "timestamp_utc"

if price_date not in prices.columns:
    raise SystemExit("Price file mein date/timestamp column nahi mila.")

if sentiment_date not in sentiment.columns:
    raise SystemExit("Sentiment file mein timestamp_utc column nahi mila.")

prices[price_date] = pd.to_datetime(prices[price_date], utc=True, errors="coerce")
sentiment[sentiment_date] = pd.to_datetime(sentiment[sentiment_date], utc=True, errors="coerce")

prices = prices.dropna(subset=[price_date]).sort_values(price_date)
sentiment = sentiment.dropna(subset=[sentiment_date]).sort_values(sentiment_date)

sentiment["date"] = sentiment[sentiment_date].dt.floor("D")
daily = sentiment.groupby("date").agg(
    post_count=("text", "count"),
    avg_sentiment=("sentiment_score", "mean")
).reset_index()

prices["date"] = prices[price_date].dt.floor("D")
combined = prices.merge(daily, on="date", how="left")
combined["post_count"] = combined["post_count"].fillna(0).astype(int)
combined["avg_sentiment"] = combined["avg_sentiment"].fillna(0)

Path("data").mkdir(exist_ok=True)
combined.to_csv("data/combined_x_sentiment_prices.csv", index=False)

print("Saved combined data:", len(combined), "price rows")
print("File: data/combined_x_sentiment_prices.csv")
