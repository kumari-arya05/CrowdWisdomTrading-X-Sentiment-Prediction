
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

# Load engineered data
df = pd.read_csv("data/bitcoin_features.csv")

# Price features
feature_cols = [
    "Return",
    "Return_Lag_1",
    "Return_Lag_2",
    "Moving_Average_7",
    "Volatility_7"
]

# Add sentiment/news features when present
optional_features = [
    "avg_sentiment",
    "post_volume",
    "total_likes",
    "total_reposts",
    "total_views",
    "avg_news_sentiment",
    "news_count"
]

for col in optional_features:
    if col in df.columns:
        feature_cols.append(col)

target_cols = [f"Target_n_plus_{i}" for i in range(6)]

# Ensure features are numeric
for col in feature_cols + target_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing sentiment values with neutral/zero indicators.
# This is a practical baseline, not proof that sentiment was observed.
for col in feature_cols:
    if col in ["avg_sentiment", "avg_news_sentiment"]:
        df[col] = df[col].fillna(0)
    elif col in optional_features:
        df[col] = df[col].fillna(0)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols + target_cols).copy()

if len(df) < 30:
    raise ValueError("Not enough valid rows to train the model.")

X = df[feature_cols]
y = df[target_cols]

# Chronological train/test split
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

if len(X_test) == 0:
    raise ValueError("Not enough test rows.")

os.makedirs("models", exist_ok=True)

# Save the exact feature names used during training
joblib.dump(feature_cols, "models/feature_columns.joblib")

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
print("Features used:", feature_cols)
print("Six models saved in the models folder.")
print("Note: this is a chronological holdout baseline, not full walk-forward validation.")
