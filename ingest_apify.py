
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
    raise SystemExit(
        "ERROR: APIFY_API_TOKEN or APIFY_ACTOR_ID missing in .env"
    )

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config" / "apify_input.json"
OUTPUT_FILE = ROOT / "data" / "x_posts_raw.csv"

if not CONFIG_FILE.exists():
    raise SystemExit(f"ERROR: Config file not found: {CONFIG_FILE}")

with open(CONFIG_FILE, encoding="utf-8") as f:
    actor_input = json.load(f)

print("Starting Apify actor...", flush=True)
print("Actor ID:", ACTOR_ID, flush=True)
print("Search terms:", actor_input.get("searchTerms"), flush=True)

# Start the actor
url = (
    "https://api.apify.com/v2/acts/"
    f"{ACTOR_ID.replace('/', '~')}/runs"
)

try:
    response = requests.post(
        url,
        params={"token": TOKEN, "waitForFinish": 120},
        json=actor_input,
        timeout=150
    )
    response.raise_for_status()
except requests.RequestException as e:
    if e.response is not None:
        print("Apify HTTP status:", e.response.status_code)
        print("Apify error:", e.response.text[:1500])
    raise SystemExit("ERROR: Could not start Apify actor.")

try:
    run = response.json()["data"]
except (ValueError, KeyError, TypeError):
    raise SystemExit("ERROR: Apify returned an unexpected response.")

print("Apify run status:", run.get("status"), flush=True)
print("Apify run ID:", run.get("id"), flush=True)
print("Dataset ID:", run.get("defaultDatasetId"), flush=True)

if run.get("status") != "SUCCEEDED":
    print("Actor did not succeed. Check the run details in Apify Console.")
    print("Status message:", run.get("statusMessage", "Not provided"))
    raise SystemExit(1)

dataset_id = run.get("defaultDatasetId")

if not dataset_id:
    raise SystemExit("ERROR: Apify did not return a dataset ID.")

# Fetch dataset items
data_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items"

try:
    result = requests.get(
        data_url,
        params={"token": TOKEN, "clean": "true", "format": "json"},
        timeout=120
    )
    result.raise_for_status()
    posts = result.json()
except requests.RequestException as e:
    if e.response is not None:
        print("Dataset HTTP status:", e.response.status_code)
        print("Dataset error:", e.response.text[:1500])
    raise SystemExit("ERROR: Could not fetch Apify dataset.")
except ValueError:
    raise SystemExit("ERROR: Dataset response was not valid JSON.")

if not isinstance(posts, list):
    raise SystemExit(
        "ERROR: Dataset response was not a list. Check actor output format."
    )

print("Raw dataset items returned:", len(posts), flush=True)

if not posts:
    print(
        "No posts were returned. Check the actor's input schema, "
        "run log, search terms, date filters, and usage limits."
    )
    raise SystemExit(1)

# Normalize post fields
rows = []

for post in posts:
    if not isinstance(post, dict):
        continue

    author = post.get("author") or post.get("user") or {}

    if not isinstance(author, dict):
        author = {}

    rows.append({
        "post_id": (
            post.get("id")
            or post.get("id_str")
            or post.get("tweetId")
            or post.get("tweet_id")
        ),
        "timestamp_utc": (
            post.get("createdAt")
            or post.get("created_at")
            or post.get("date")
            or post.get("timestamp")
        ),
        "text": (
            post.get("text")
            or post.get("fullText")
            or post.get("full_text")
            or ""
        ),
        "likes": post.get(
            "likeCount", post.get("like_count", 0)
        ),
        "reposts": post.get(
            "retweetCount", post.get("retweet_count", 0)
        ),
        "views": post.get(
            "viewCount", post.get("view_count", 0)
        ),
        "author_followers": (
            author.get("followers")
            or author.get("followersCount")
            or author.get("followers_count", 0)
        ),
        "url": post.get("url") or post.get("tweetUrl", "")
    })

df = pd.DataFrame(rows)

if df.empty:
    raise SystemExit("ERROR: No usable post records were found.")

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"], utc=True, errors="coerce"
)

before = len(df)
df = df.dropna(subset=["timestamp_utc"]).copy()

if df.empty:
    print(f"Raw items: {before}; valid timestamp rows: 0")
    print(
        "The actor's timestamp field may have a different name. "
        "Inspect the actor's output schema."
    )
    raise SystemExit(1)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"Saved {len(df)} X posts to {OUTPUT_FILE}", flush=True)
print("Ingestion finished.", flush=True)
