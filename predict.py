
import pandas as pd
import joblib
import os

# Load engineered feature data
df = pd.read_csv("data/bitcoin_features.csv")

# Load the exact features used during model training
feature_path = "models/feature_columns.joblib"

if not os.path.exists(feature_path):
    raise FileNotFoundError(
        "Feature list nahi mili. Pehle python model.py run karo."
    )

feature_cols = joblib.load(feature_path)

# Check required columns
missing = [col for col in feature_cols if col not in df.columns]

if missing:
    raise ValueError(f"Missing feature columns: {missing}")

# Convert features to numeric values
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace missing/infinite feature values safely
df[feature_cols] = df[feature_cols].replace(
    [float("inf"), float("-inf")], float("nan")
)
df[feature_cols] = df[feature_cols].fillna(0)

# Get the latest available row
if df.empty:
    raise ValueError("bitcoin_features.csv mein data nahi hai.")

latest = df.iloc[[-1]]
X_latest = latest[feature_cols]

# Current Bitcoin price
current_price = float(latest["Close"].iloc[0])

print("Current Bitcoin price in dataset:", round(current_price, 2))
print("\nPredicted prices:")

results = []

# Predict n+0 through n+5
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
os.makedirs("data", exist_ok=True)

output_file = "data/bitcoin_predictions.csv"
pd.DataFrame(results).to_csv(output_file, index=False)

print("\nPredictions saved to", output_file)
print("Prediction process completed!")
