
import pandas as pd
import numpy as np
from pathlib import Path

# Load sentiment-model walk-forward predictions
file_path = Path("reports/sentiment_walk_forward_predictions.csv")
df = pd.read_csv(file_path)

print("Available columns:", df.columns.tolist())

# Identify required columns
required = ["horizon", "actual_close", "predicted_close"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        "Check the column names in sentiment_walk_forward_predictions.csv."
    )

df["actual_close"] = pd.to_numeric(df["actual_close"], errors="coerce")
df["predicted_close"] = pd.to_numeric(df["predicted_close"], errors="coerce")
df = df.dropna(subset=required).copy()
df = df.sort_values(["horizon"]).reset_index(drop=True)

results = []

for horizon, group in df.groupby("horizon", sort=True):
    group = group.copy()

    actual = group["actual_close"]
    predicted = group["predicted_close"]

    mae = (actual - predicted).abs().mean()

    # Preliminary directional strategy proxy.
    # Uses next-observation actual return and previous prediction direction.
    actual_return = actual.pct_change()
    predicted_direction = np.sign(predicted.shift(1) - actual.shift(1))
    strategy_return = (predicted_direction * actual_return).dropna()
    strategy_return = strategy_return.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    downside = strategy_return[strategy_return < 0]
    downside_dev = (
        np.sqrt((downside ** 2).mean())
        if len(downside) else np.nan
    )

    equity = (1 + strategy_return).cumprod()
    drawdown = equity / equity.cummax() - 1

    std = strategy_return.std(ddof=1)

    sharpe = (
        np.sqrt(365) * strategy_return.mean() / std
        if len(strategy_return) > 1 and std > 0
        else np.nan
    )

    sortino = (
        np.sqrt(365) * strategy_return.mean() / downside_dev
        if pd.notna(downside_dev) and downside_dev > 0
        else np.nan
    )

    results.append({
        "Horizon": horizon,
        "Observations": len(group),
        "MAE": mae,
        "Sharpe Ratio (annualized, preliminary)": sharpe,
        "Sortino Ratio (annualized, preliminary)": sortino,
        "Maximum Drawdown (preliminary)": (
            drawdown.min() if len(drawdown) else np.nan
        )
    })

report = pd.DataFrame(results)
Path("reports").mkdir(exist_ok=True)
report.to_csv("reports/sentiment_walk_forward_metrics.csv", index=False)

print("\nSentiment-model evaluation:")
print(report.to_string(index=False))
print("\nSaved: reports/sentiment_walk_forward_metrics.csv")
print(
    "\nImportant: Trading metrics are preliminary proxies, "
    "not validated investment performance."
)
