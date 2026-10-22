
import pandas as pd
import numpy as np
import os

# Load Bitcoin historical prices
df = pd.read_csv("data/bitcoin_prices.csv")

# Check required columns
if "Close" not in df.columns:
    raise ValueError("Close price column not found in CSV.")

# Convert price to numeric
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
df = df.dropna(subset=["Close"]).copy()

# Create price-based features
df["Return"] = df["Close"].pct_change()
df["Return_Lag_1"] = df["Return"].shift(1)
df["Return_Lag_2"] = df["Return"].shift(2)
df["Moving_Average_7"] = df["Close"].rolling(7).mean()
df["Volatility_7"] = df["Return"].rolling(7).std()

# Predict prices for the next 6 periods: n+0 to n+5
for i in range(6):
    df[f"Target_n_plus_{i}"] = df["Close"].shift(-i)

# Save features
os.makedirs("data", exist_ok=True)
df.to_csv("data/bitcoin_features.csv", index=False)

print("Feature engineering completed!")
print("Total rows:", len(df))
print("\nLatest features:")
print(df.tail())
print("\nSaved to data/bitcoin_features.csv")
