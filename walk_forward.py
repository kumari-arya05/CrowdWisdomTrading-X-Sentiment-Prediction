import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

df = pd.read_csv("data/bitcoin_prices.csv")
df.columns = df.columns.str.lower().str.strip()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").dropna(subset=["close"]).copy()

df["return"] = df["close"].pct_change()
for lag in range(1, 6):
    df[f"return_lag_{lag}"] = df["return"].shift(lag)

df["target"] = df["close"].shift(-1)
df = df.dropna().reset_index(drop=True)

features = [f"return_lag_{lag}" for lag in range(1, 6)]
predictions = []

min_train = 90

for i in range(min_train, len(df)):
    train = df.iloc[:i]
    test = df.iloc[[i]]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(train[features], train["target"])
    prediction = model.predict(test[features])[0]

    predictions.append({
        "date": test["date"].iloc[0],
        "actual_close": test["target"].iloc[0],
        "predicted_close": prediction
    })

results = pd.DataFrame(predictions)
Path("reports").mkdir(exist_ok=True)
results.to_csv("reports/walk_forward_predictions.csv", index=False)

mae = mean_absolute_error(results["actual_close"], results["predicted_close"])
print(f"Walk-forward predictions saved: {len(results)}")
print(f"MAE: {mae:.2f}")
print("File: reports/walk_forward_predictions.csv")
print("Note: This baseline uses price lags only, not X sentiment.")
