import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv("reports/walk_forward_predictions.csv")
df["actual_close"] = pd.to_numeric(df["actual_close"], errors="coerce")
df["predicted_close"] = pd.to_numeric(df["predicted_close"], errors="coerce")
df = df.dropna(subset=["actual_close", "predicted_close"]).copy()

df["actual_return"] = df["actual_close"].pct_change()
df["strategy_return"] = (
    df["predicted_close"].shift(1) / df["actual_close"].shift(1) - 1
) * np.sign(df["predicted_close"].shift(1) - df["actual_close"].shift(1))

r = df["strategy_return"].dropna()
downside = r[r < 0]
downside_dev = np.sqrt((downside ** 2).mean()) if len(downside) else 0
equity = (1 + r).cumprod()
drawdown = equity / equity.cummax() - 1

report = pd.DataFrame([{
    "Evaluation observations": len(df),
    "MAE": (df["actual_close"] - df["predicted_close"]).abs().mean(),
    "Sharpe Ratio (annualized)": np.sqrt(365) * r.mean() / r.std(ddof=1) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan,
    "Sortino Ratio (annualized)": np.sqrt(365) * r.mean() / downside_dev if downside_dev > 0 else np.nan,
    "Maximum Drawdown": drawdown.min() if len(drawdown) else np.nan
}])

Path("reports").mkdir(exist_ok=True)
report.to_csv("reports/walk_forward_metrics.csv", index=False)
print(report.to_string(index=False))
print("\nSaved: reports/walk_forward_metrics.csv")
print("Note: These are preliminary metrics from a price-only baseline; not the X-sentiment strategy.")
