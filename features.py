
import pandas as pd
import numpy as np
import os

# Load Bitcoin historical prices
df = pd.read_csv("data/bitcoin_prices.csv")

if "Close" not in df.columns:
    raise ValueError("Close price column not found in CSV.")

df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"]).copy()

# Create a date column for joining daily data
date_col = "Date" if "Date" in df.columns else "date"
df["date"] = pd.to_datetime(df[date_col], utc=True).dt.date

# Price-based features
df["Return"] = df["Close"].pct_change()
df["Return_Lag_1"] = df["Return"].shift(1)
df["Return_Lag_2"] = df["Return"].shift(2)
df["Moving_Average_7"] = df["Close"].rolling(7).mean()
df["Volatility_7"] = df["Return"].rolling(7).std()

# Add social sentiment data, if available
sentiment_file = "data/combined_data.csv"

if os.path.exists(sentiment_file):
    sentiment = pd.read_csv(sentiment_file)

    if "date" in sentiment.columns:
        sentiment["date"] = pd.to_datetime(
            sentiment["date"], utc=True
        ).dt.date

        columns_to_use = [
            col for col in [
                "date", "avg_sentiment", "post_volume",
                "total_likes", "total_reposts", "total_views"
            ]
            if col in sentiment.columns
        ]

        if len(columns_to_use) > 1:
            sentiment = sentiment[columns_to_use].drop_duplicates("date")
            df = df.merge(sentiment, on="date", how="left")

# Add news sentiment data, if available
news_file = "data/bitcoin_news.csv"

if os.path.exists(news_file):
    news = pd.read_csv(news_file)

    if "fetched_at_utc" in news.columns and "sentiment_score" in news.columns:
        news["date"] = pd.to_datetime(
            news["fetched_at_utc"], utc=True
        ).dt.date

        daily_news = news.groupby("date").agg(
            avg_news_sentiment=("sentiment_score", "mean"),
            news_count=("title", "count")
        ).reset_index()

        df = df.merge(daily_news, on="date", how="left")

# Create targets n+0 to n+5
for i in range(6):
    df[f"Target_n_plus_{i}"] = df["Close"].shift(-i)

os.makedirs("data", exist_ok=True)
df.to_csv("data/bitcoin_features.csv", index=False)

print("Feature engineering completed!")
print("Total rows:", len(df))
print("Available sentiment/news features:")
print([
    c for c in df.columns
    if "sentiment" in c or c in ["post_volume", "news_count"]
])
print("Saved to data/bitcoin_features.csv")
