
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

# Load engineered data
df = pd.read_csv("data/bitcoin_features.csv")

# Features available at prediction time
feature_cols = [
    "Return",
    "Return_Lag_1",
    "Return_Lag_2",
    "Moving_Average_7",
    "Volatility_7"
]

target_cols = [f"Target_n_plus_{i}" for i in range(6)]

# Use the latest rows for testing
df = df.dropna(subset=feature_cols + target_cols).copy()

if len(df) < 30:
    raise ValueError("Not enough valid rows to train the model.")

X = df[feature_cols]
y = df[target_cols]

# Chronological train/test split
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Train one model for each forecast horizon
os.makedirs("models", exist_ok=True)

for i, target in enumerate(target_cols):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )
    model.fit(X_train, y_train[target])

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test[target], predictions)

    joblib.dump(model, f"models/model_n_plus_{i}.joblib")

    print(f"n+{i} | Test MAE: ${mae:,.2f}")

print("\nModel training completed!")
print("Six models saved in the models folder.")
