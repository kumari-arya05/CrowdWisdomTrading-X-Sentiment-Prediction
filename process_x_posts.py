import pandas as pd
from pathlib import Path

input_file = Path("data/x_posts_raw.csv")
output_file = Path("data/x_posts_processed.csv")

if not input_file.exists():
    raise SystemExit("ERROR: data/x_posts_raw.csv file nahi mili.")

df = pd.read_csv(input_file)

if df.empty:
    raise SystemExit("ERROR: CSV mein koi post nahi hai.")

df["text"] = df["text"].fillna("").astype(str)
df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"], utc=True, errors="coerce"
)

df = df.dropna(subset=["timestamp_utc"])
df = df.drop_duplicates(subset=["post_id"], keep="first")

output_file.parent.mkdir(exist_ok=True)
df.to_csv(output_file, index=False)

print(f"Processed {len(df)} posts.")
print(f"Saved to: {output_file}")
