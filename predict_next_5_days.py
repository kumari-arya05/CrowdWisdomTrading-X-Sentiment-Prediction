import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/bitcoin_prices.csv")
df.columns = df.columns.str.lower().str.strip()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").dropna(subset=["close"]).copy()

for lag in range(1, 6):
    df[f"close_lag_{lag}"] = df["close"].shift(lag)

features = [f"close_lag_{lag}" for lag in range(1, 6)]
df = df.dropna().reset_index(drop=True)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(df[features], df["close"])

last_values = df[features].iloc[-1].to_dict()
predictions = []

for day in range(1, 6):
    x = pd.DataFrame([last_values], columns=features)
    predicted_price = float(model.predict(x)[0])

    predictions.append({
        "day_ahead": f"n+{day}",
        "predicted_close": round(predicted_price, 2)
    })

    last_values = {
        "close_lag_1": predicted_price,
        "close_lag_2": last_values["close_lag_1"],
        "close_lag_3": last_values["close_lag_2"],
        "close_lag_4": last_values["close_lag_3"],
        "close_lag_5": last_values["close_lag_4"]
    }

Path("reports").mkdir(exist_ok=True)
output = pd.DataFrame(predictions)
output.to_csv("reports/bitcoin_next_5_days.csv", index=False)

print(output.to_string(index=False))
print("\nSaved: reports/bitcoin_next_5_days.csv")
print("Note: Price-only baseline; X sentiment is not yet included.")
