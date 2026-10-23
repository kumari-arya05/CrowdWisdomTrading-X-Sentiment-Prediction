
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("Starting walk-forward prediction...", flush=True)

# Load combined Bitcoin price and X sentiment data
file_path = Path("data/combined_x_sentiment_prices.csv")

if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path.resolve()}")

df = pd.read_csv(file_path)
df.columns = df.columns.str.lower().str.strip()

required = ["date", "close", "post_count", "avg_sentiment"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

for col in ["close", "post_count", "avg_sentiment"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = (
    df.sort_values("date")
    .dropna(subset=["date", "close"])
    .reset_index(drop=True)
)

print(f"Loaded {len(df)} price rows.", flush=True)

# Feature engineering
df["return"] = df["close"].pct_change()
df["sentiment_weighted"] = (
    df["avg_sentiment"] * np.log1p(df["post_count"].clip(lower=0))
)

features = []

for lag in range(1, 6):
    return_col = f"return_lag_{lag}"
    sentiment_col = f"sentiment_lag_{lag}"
    count_col = f"post_count_lag_{lag}"

    df[return_col] = df["return"].shift(lag)
    df[sentiment_col] = df["avg_sentiment"].shift(lag)
    df[count_col] = df["post_count"].shift(lag)

    features.extend([return_col, sentiment_col, count_col])

features.append("sentiment_weighted")

# Replace missing feature values only; targets remain actual prices
df[features] = df[features].replace([np.inf, -np.inf], np.nan)
df[features] = df[features].fillna(0)

Path("reports").mkdir(exist_ok=True)

all_predictions = []
metrics = []
min_train = 90

# Forecast n+0 through n+5
for horizon in range(6):
    print(f"Training horizon n+{horizon}...", flush=True)

    target_col = f"target_n_plus_{horizon}"
    df[target_col] = df["close"].shift(-horizon)

    data = df.dropna(subset=[target_col]).reset_index(drop=True)
    predictions = []

    for i in range(min_train, len(data)):
        train = data.iloc[:i]
        test = data.iloc[[i]]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            min_samples_leaf=3,
            n_jobs=-1
        )

        model.fit(train[features], train[target_col])
        predicted_price = float(model.predict(test[features])[0])

        predictions.append({
            "date": test["date"].iloc[0],
            "horizon": f"n+{horizon}",
            "actual_close": float(test[target_col].iloc[0]),
            "predicted_close": predicted_price,
            "post_count": int(test["post_count"].fillna(0).iloc[0]),
            "avg_sentiment": float(
                test["avg_sentiment"].fillna(0).iloc[0]
            )
        })

    if predictions:
        result = pd.DataFrame(predictions)
        mae = mean_absolute_error(
            result["actual_close"],
            result["predicted_close"]
        )

        metrics.append({
            "horizon": f"n+{horizon}",
            "observations": len(result),
            "MAE": round(mae, 4)
        })

        all_predictions.append(result)
    else:
        print(
            f"Not enough rows for n+{horizon}; skipped.",
            flush=True
        )

# Save results
if all_predictions:
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(metrics)

    predictions_df.to_csv(
        "reports/sentiment_walk_forward_predictions.csv",
        index=False
    )
    metrics_df.to_csv(
        "reports/sentiment_walk_forward_metrics.csv",
        index=False
    )

    print("\nWalk-forward metrics:", flush=True)
    print(metrics_df.to_string(index=False), flush=True)
    print(
        "\nSaved: reports/sentiment_walk_forward_predictions.csv",
        flush=True
    )
    print(
        "Saved: reports/sentiment_walk_forward_metrics.csv",
        flush=True
    )
else:
    print(
        "No predictions generated. Check that at least 91 price rows exist.",
        flush=True
    )

print("\nScript finished.", flush=True)
