
import yfinance as yf
import pandas as pd
import os

print("CrowdWisdomTrading Project Started!")

# Create data folder
os.makedirs("data", exist_ok=True)

# Download last 180 days of Bitcoin price data
print("Downloading Bitcoin price data...")

df = yf.download(
    "BTC-USD",
    period="180d",
    interval="1d",
    auto_adjust=True,
    progress=False
)

if df.empty:
    print("No data downloaded. Check internet connection.")
else:
    # Flatten columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Save data
    df.to_csv("data/bitcoin_prices.csv", index=False)

    print("Bitcoin price data downloaded successfully!")
    print(df.head())
    print("Total rows:", len(df))
    print("Saved to: data/bitcoin_prices.csv")
