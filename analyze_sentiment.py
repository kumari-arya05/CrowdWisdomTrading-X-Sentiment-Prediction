import pandas as pd
from pathlib import Path
from textblob import TextBlob

input_file = Path("data/x_posts_processed.csv")
output_file = Path("data/x_posts_sentiment.csv")

df = pd.read_csv(input_file)

if "text" not in df.columns:
    raise SystemExit("ERROR: CSV mein text column nahi hai.")

def analyze_sentiment(text):
    score = TextBlob(str(text)).sentiment.polarity
    if score > 0.05:
        label = "positive"
    elif score < -0.05:
        label = "negative"
    else:
        label = "neutral"
    return pd.Series([score, label])

df[["sentiment_score", "sentiment_label"]] = df["text"].apply(analyze_sentiment)
df.to_csv(output_file, index=False)

print(f"Sentiment analysis complete for {len(df)} posts.")
print(f"Saved to: {output_file}")
