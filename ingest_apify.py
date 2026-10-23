import os
import json
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("APIFY_API_TOKEN")
ACTOR_ID = os.getenv("APIFY_ACTOR_ID")

if not TOKEN or not ACTOR_ID:
    raise SystemExit("Please set APIFY_API_TOKEN and APIFY_ACTOR_ID in your local .env file.")

ROOT = Path(__file__).resolve().parent
with open(ROOT / "config" / "apify_input.json", encoding="utf-8") as f:
    actor_input = json.load(f)

url = f"https://api.apify.com/v2/acts/{ACTOR_ID.replace('/', '~')}/runs"
response = requests.post(
    url,
    params={"token": TOKEN, "waitForFinish": 120},
    json=actor_input,
    timeout=150
)
response.raise_for_status()
run = response.json()["data"]

if run.get("status") != "SUCCEEDED":
    raise SystemExit(f"Apify run status: {run.get('status')}. Check the run in Apify.")

dataset_id = run.get("defaultDatasetId")
data_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"
result = requests.get(
    data_url,
    params={"token": TOKEN, "clean": "true", "format": "json"},
    timeout=120
)
result.raise_for_status()
posts = result.json()

if not posts:
    raise SystemExit("No posts returned. Check your actor settings in config/apify_input.json.")

rows = []
for post in posts:
    author = post.get("author") or post.get("user") or {}
    rows.append({
        "post_id": post.get("id") or post.get("id_str") or post.get("tweetId"),
        "timestamp_utc": post.get("createdAt") or post.get("created_at") or post.get("date"),
        "text": post.get("text") or post.get("fullText") or post.get("full_text") or "",
        "likes": post.get("likeCount", post.get("like_count", 0)),
        "reposts": post.get("retweetCount", post.get("retweet_count", 0)),
        "views": post.get("viewCount", post.get("view_count", 0)),
        "author_followers": author.get("followers") or author.get("followersCount") or author.get("followers_count", 0),
        "url": post.get("url") or post.get("tweetUrl", "")
    })

df = pd.DataFrame(rows)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
df = df.dropna(subset=["timestamp_utc"])
Path("data").mkdir(exist_ok=True)
df.to_csv("data/x_posts_raw.csv", index=False)
print(f"Saved {len(df)} X posts to data/x_posts_raw.csv")
