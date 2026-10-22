
import pandas as pd
import numpy as np
import joblib
import os

# Load engineered data
df = pd.read_csv("data/bitcoin_features.csv")

# Load exact features used during training
feature_path = "models/feature_columns.joblib"

if not os.path.exists(feature_path):
    raise FileNotFoundError(
        "Feature list nahi mili. Pehle python model.py run karo."
    )

feature_cols = joblib.load(feature_path)
target_col = "Target_n_plus_1"

# Validate required columns
required_cols = feature_cols + [target_col, "Close"]
missing = [col for col in required_cols if col not in df.columns]

if missing:
    raise ValueError(f"Required columns missing: {missing}")

# Convert model features to numeric
for col in feature_cols + [target_col, "Close"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)

# Match prediction-time preprocessing
df[feature_cols] = df[feature_cols].fillna(0)
df = df.dropna(subset=[target_col, "Close"]).copy()

if len(df) < 30:
    raise ValueError("Evaluation ke liye enough data nahi hai.")

# Chronological holdout evaluation
split = int(len(df) * 0.8)
test = df.iloc[split:].copy()

if test.empty:
    raise ValueError("Test data available nahi hai.")

model = joblib.load("models/model_n_plus_1.joblib")
predictions = model.predict(test[feature_cols])

current_prices = test["Close"].to_numpy()
actual_next_prices = test[target_col].to_numpy()

actual_returns = (actual_next_prices / current_prices) - 1
signals = (predictions > current_prices).astype(float)

strategy_returns = pd.Series(
    signals * actual_returns
).replace([np.inf, -np.inf], np.nan).dropna()

if strategy_returns.empty:
    raise ValueError("No valid strategy returns to evaluate.")

std = strategy_returns.std(ddof=1)

sharpe = (
    np.sqrt(365) * strategy_returns.mean() / std
    if len(strategy_returns) > 1 and std > 0
    else np.nan
)

downside = strategy_returns[strategy_returns < 0]
downside_deviation = np.sqrt((downside ** 2).mean())

sortino = (
    np.sqrt(365) * strategy_returns.mean() / downside_deviation
    if downside_deviation > 0
    else np.nan
)

equity = (1 + strategy_returns).cumprod()
drawdown = equity / equity.cummax() - 1
max_drawdown = drawdown.min()

os.makedirs("reports", exist_ok=True)

report = pd.DataFrame([{
    "Evaluation observations": len(strategy_returns),
    "Sharpe Ratio (annualized)": sharpe,
    "Sortino Ratio (annualized)": sortino,
    "Maximum Drawdown": max_drawdown
}])

report.to_csv("reports/evaluation_metrics.csv", index=False)

print("\nEvaluation metrics calculated:")
print(report.to_string(index=False))
print("\nSaved to reports/evaluation_metrics.csv")
print(
    "\nNote: This is a basic holdout backtest, "
    "not rigorous walk-forward validation."
)
