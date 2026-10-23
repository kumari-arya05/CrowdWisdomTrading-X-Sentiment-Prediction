import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("data/combined_x_sentiment_prices.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date")

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df["date"], df["close"], label="Bitcoin closing price")
ax1.set_xlabel("Date")
ax1.set_ylabel("BTC closing price")
ax1.legend(loc="upper left")

ax2 = ax1.twinx()
ax2.plot(df["date"], df["avg_sentiment"], label="Average sentiment")
ax2.set_ylabel("Average sentiment")
ax2.legend(loc="upper right")

plt.title("Bitcoin Price vs X Sentiment")
plt.tight_layout()
Path("charts").mkdir(exist_ok=True)
plt.savefig("charts/bitcoin_sentiment_price.png", dpi=150)
plt.close()

print("Chart saved: charts/bitcoin_sentiment_price.png")
