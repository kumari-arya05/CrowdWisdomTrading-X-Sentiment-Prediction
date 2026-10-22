
import pandas as pd
import joblib
import os

# Load feature data
df = pd.read_csv("data/bitcoin_features.csv")

feature_cols = [
    "Return",
    "Return_Lag_1",
    "Return_Lag_2",
    "Moving_Average_7",
    "Volatility_7"
]

# Get latest row with complete features
latest = df.dropna(subset=feature_cols).iloc[[-1]]
X_latest = latest[feature_cols]

# Current Bitcoin price
current_price = float(latest["Close"].iloc[0])

print("Current Bitcoin price in dataset:", round(current_price, 2))
print("\nPredicted prices:")

results = []

for i in range(6):
    model_path = f"models/model_n_plus_{i}.joblib"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    predicted_price = float(model.predict(X_latest)[0])

    results.append({
        "Horizon": f"n+{i}",
        "Predicted_Price_USD": round(predicted_price, 2)
    })

    print(f"n+{i}: ${predicted_price:,.2f}")

# Save predictions
pd.DataFrame(results).to_csv(
    "data/bitcoin_predictions.csv", index=False
)

print("\nPredictions saved to data/bitcoin_predictions.csv")
