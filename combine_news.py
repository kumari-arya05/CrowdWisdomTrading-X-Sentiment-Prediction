
import pandas as pd
import os

# Load the original price and X sentiment dataset
data = pd.read_csv("data/combined_x_sentiment_prices.csv")

# Standardize date column
data["date"] = pd.to_datetime(data["date"], utc=True).dt.date

# Load news data
news = pd.read_csv("data/bitcoin_news.csv")

if news.empty:
    print("No news found. Run python news.py first.")
    raise SystemExit(1)

# Prepare news dates
news["fetched_at_utc"] = pd.to_datetime(
    news["fetched_at_utc"], utc=True, errors="coerce"
)
news["date"] = news["fetched_at_utc"].dt.date

# Aggregate news sentiment by date
daily_news = news.groupby("date").agg(
    avg_news_sentiment=("sentiment_score", "mean"),
    news_count=("title", "count")
).reset_index()

# Merge without duplicate date columns
merged = data.merge(
    daily_news,
    on="date",
    how="left",
    validate="one_to_one"
)

merged["avg_news_sentiment"] = (
    merged["avg_news_sentiment"].fillna(0)
)
merged["news_count"] = (
    merged["news_count"].fillna(0).astype(int)
)

# Save final dataset
os.makedirs("data", exist_ok=True)
merged.to_csv("data/combined_data.csv", index=False)

print("News sentiment added!")
print("Total rows:", len(merged))
print("News rows matched:", int(merged["news_count"].sum()))
print("Columns:", merged.columns.tolist())
print("Saved to data/combined_data.csv")
