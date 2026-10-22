
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

analyzer = SentimentIntensityAnalyzer()

# Temporary sample posts for testing the sentiment pipeline.
# Replace these with real X posts before final submission.
posts = [
    {
        "timestamp": "2026-10-20 10:00:00",
        "text": "Bitcoin is rising strongly. The market looks bullish!",
        "likes": 120,
        "reposts": 30,
        "views": 5000
    },
    {
        "timestamp": "2026-10-20 11:00:00",
        "text": "Bitcoin is crashing. Investors are worried about losses.",
        "likes": 80,
        "reposts": 20,
        "views": 3500
    },
    {
        "timestamp": "2026-10-20 12:00:00",
        "text": "Bitcoin price is stable today.",
        "likes": 25,
        "reposts": 5,
        "views": 1000
    }
]

df = pd.DataFrame(posts)

# Calculate sentiment score for each post
df["sentiment_score"] = df["text"].apply(
    lambda text: analyzer.polarity_scores(text)["compound"]
)

# Group posts by hour
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
df["hour"] = df["timestamp"].dt.floor("h")

hourly = df.groupby("hour").agg(
    avg_sentiment=("sentiment_score", "mean"),
    post_volume=("text", "count"),
    total_likes=("likes", "sum"),
    total_reposts=("reposts", "sum"),
    total_views=("views", "sum")
).reset_index()

os.makedirs("data", exist_ok=True)
df.to_csv("data/sentiment_posts.csv", index=False)
hourly.to_csv("data/hourly_sentiment.csv", index=False)

print("Sentiment analysis completed!")
print("\nPost sentiment:")
print(df[["text", "sentiment_score"]])
print("\nHourly sentiment summary:")
print(hourly)
