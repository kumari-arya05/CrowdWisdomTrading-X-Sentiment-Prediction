from pathlib import Path
import pandas as pd

df = pd.read_csv("data/x_posts_sentiment.csv")
df["sentiment_score"] = pd.to_numeric(
    df["sentiment_score"], errors="coerce"
)
df["sentiment_label"] = df["sentiment_label"].fillna("unknown")

total = len(df)
avg = df["sentiment_score"].mean() if total else 0
labels = df["sentiment_label"].value_counts().to_dict()

report = f"""# Bitcoin X Sentiment Outlook

## Data coverage
- Available X posts: {total}
- Average sentiment score: {avg:.3f}
- Sentiment labels: {labels}

## Current interpretation
The available sample has {total} post(s). The average sentiment score is
{avg:.3f}. This sample is too small to reliably describe overall Bitcoin
market sentiment or predict future prices.

## Data limitations
- The assignment requires X posts from the previous 180 days.
- The current dataset does not meet that coverage requirement.
- A sentiment score of 0.0 may reflect the sentiment tool's classification;
  it does not prove that the market is neutral.
- More real posts and validated sentiment scores are needed before drawing
  a reliable market outlook.

## Conclusion
Treat this as a preliminary data-quality note, not an investment signal.
"""

Path("reports").mkdir(exist_ok=True)
Path("reports/bitcoin_sentiment_outlook.md").write_text(
    report, encoding="utf-8"
)
print("Saved: reports/bitcoin_sentiment_outlook.md")
print(report)
