
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load combined data
df = pd.read_csv("data/combined_data.csv")

# Convert dates
df["Date"] = pd.to_datetime(df["Date"], utc=True)

# Sort chronologically
df = df.sort_values("Date")

os.makedirs("charts", exist_ok=True)

# Chart 1: Bitcoin closing price
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Close"])
plt.title("Bitcoin Closing Price - Historical Data")
plt.xlabel("Date (UTC)")
plt.ylabel("Price (USD)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/bitcoin_price.png")
plt.close()

# Chart 2: Sentiment over time
if "avg_sentiment" in df.columns:
    sentiment_df = df.dropna(subset=["avg_sentiment"])

    plt.figure(figsize=(12, 5))
    plt.plot(
        sentiment_df["Date"],
        sentiment_df["avg_sentiment"]
    )
    plt.title("X Sentiment Over Time")
    plt.xlabel("Date (UTC)")
    plt.ylabel("Average Sentiment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("charts/sentiment_over_time.png")
    plt.close()

# Chart 3: Price and sentiment comparison
if "avg_sentiment" in df.columns:
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(df["Date"], df["Close"], label="BTC Price")
    ax1.set_xlabel("Date (UTC)")
    ax1.set_ylabel("Price (USD)")

    ax2 = ax1.twinx()
    ax2.plot(
        df["Date"],
        df["avg_sentiment"],
        label="Sample Sentiment"
    )
    ax2.set_ylabel("Average Sentiment")

    plt.title("Bitcoin Price vs Sentiment")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig("charts/price_vs_sentiment.png")
    plt.close(fig)

print("Charts created successfully!")
print("Check the charts folder.")
