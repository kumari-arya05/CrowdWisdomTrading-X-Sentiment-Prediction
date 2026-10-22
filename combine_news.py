
import pandas as pd
import os

# Load existing combined price and sample sentiment data
data = pd.read_csv("data/combined_data.csv")
data["date"] = pd.to_datetime(data["date"], utc=True).dt.date

# Load real news results and sentiment scores
news = pd.read_csv("data/bitcoin_news.csv")

if news.empty:
    print("No news found. Run python news.py first.")
    raise SystemExit(1)

news["fetched_at_utc"] = pd.to_datetime(
    news["fetched_at_utc"], utc=True
)
news["date"] = news["fetched_at_utc"].dt.date

# Aggregate news sentiment by collection date
daily_news = news.groupby("date").agg(
    avg_news_sentiment=("sentiment_score", "mean"),
    news_count=("title", "count")
).reset_index()

# Merge news features with existing dataset
merged = data.merge(daily_news, on="date", how="left")

merged["news_count"] = merged["news_count"].fillna(0).astype(int)

os.makedirs("data", exist_ok=True)
merged.to_csv("data/combined_data.csv", index=False)

print("News sentiment added!")
print("Total rows:", len(merged))
print("News rows matched:", int(merged["news_count"].sum()))
print("Saved to data/combined_data.csv")
