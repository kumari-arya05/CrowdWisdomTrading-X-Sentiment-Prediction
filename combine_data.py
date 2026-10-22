
import pandas as pd
import os

# Load Bitcoin price data
prices = pd.read_csv("data/bitcoin_prices.csv")
prices["Date"] = pd.to_datetime(prices["Date"], utc=True)

# Load hourly sentiment data
sentiment = pd.read_csv("data/hourly_sentiment.csv")
sentiment["hour"] = pd.to_datetime(sentiment["hour"], utc=True)

# Align both datasets by UTC date/hour
prices["hour"] = prices["Date"].dt.floor("h")
sentiment["date"] = sentiment["hour"].dt.floor("D")
prices["date"] = prices["hour"].dt.floor("D")

# Aggregate sentiment to daily frequency
daily_sentiment = sentiment.groupby("date").agg(
    avg_sentiment=("avg_sentiment", "mean"),
    post_volume=("post_volume", "sum"),
    total_likes=("total_likes", "sum"),
    total_reposts=("total_reposts", "sum"),
    total_views=("total_views", "sum")
).reset_index()

# Merge price and sentiment
merged = prices.merge(daily_sentiment, on="date", how="left")

# Mark dates without posts as missing, not neutral sentiment
merged["post_volume"] = merged["post_volume"].fillna(0)
merged["total_likes"] = merged["total_likes"].fillna(0)
merged["total_reposts"] = merged["total_reposts"].fillna(0)
merged["total_views"] = merged["total_views"].fillna(0)

os.makedirs("data", exist_ok=True)
merged.to_csv("data/combined_data.csv", index=False)

print("Price and sentiment data combined!")
print("Total rows:", len(merged))
print("\nLatest rows:")
print(merged.tail())
print("\nSaved to data/combined_data.csv")
