
import pandas as pd
import numpy as np
import joblib
import os

# Load historical price and engineered features
df = pd.read_csv("data/bitcoin_features.csv")

feature_cols = [
    "Return",
    "Return_Lag_1",
    "Return_Lag_2",
    "Moving_Average_7",
    "Volatility_7"
]

target_col = "Target_n_plus_1"

df = df.dropna(subset=feature_cols + [target_col]).copy()

# Chronological split: train on earlier data, evaluate later data
split = int(len(df) * 0.8)
test = df.iloc[split:].copy()

model = joblib.load("models/model_n_plus_1.joblib")

predictions = model.predict(test[feature_cols])

# Simple illustrative strategy:
# hold BTC when predicted next-period price exceeds current price
current_prices = test["Close"].to_numpy()
actual_next_prices = test[target_col].to_numpy()

actual_returns = (actual_next_prices / current_prices) - 1
signals = (predictions > current_prices).astype(float)

strategy_returns = signals * actual_returns
strategy_returns = pd.Series(strategy_returns).replace(
    [np.inf, -np.inf], np.nan
).dropna()

if strategy_returns.empty:
    raise ValueError("No valid strategy returns to evaluate.")

# Daily risk-free rate assumed to be zero
sharpe = (
    np.sqrt(365) * strategy_returns.mean() / strategy_returns.std(ddof=1)
    if len(strategy_returns) > 1 and strategy_returns.std(ddof=1) > 0
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
    "\nImportant: This is an initial illustrative backtest, "
    "not the final validated assignment result."
)
