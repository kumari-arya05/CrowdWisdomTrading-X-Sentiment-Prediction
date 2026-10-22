
import os
import csv
from datetime import datetime, timezone
from dotenv import load_dotenv
from tavily import TavilyClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    print("Error: .env file mein TAVILY_API_KEY check karo.")
    raise SystemExit(1)

client = TavilyClient(api_key=api_key)
analyzer = SentimentIntensityAnalyzer()

try:
    response = client.search(
        query="Latest Bitcoin cryptocurrency news",
        topic="news",
        max_results=10
    )

    results = response.get("results", [])
    os.makedirs("data", exist_ok=True)

    output_file = "data/bitcoin_news.csv"

    with open(output_file, "w", newline="", encoding="utf-8-sig") as file:
        fields = [
            "fetched_at_utc",
            "title",
            "url",
            "content",
            "sentiment_score",
            "sentiment_label"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for item in results:
            title = item.get("title", "")
            content = item.get("content", "")
            text = f"{title}. {content}"

            score = analyzer.polarity_scores(text)["compound"]

            if score >= 0.05:
                label = "positive"
            elif score <= -0.05:
                label = "negative"
            else:
                label = "neutral"

            writer.writerow({
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "title": title,
                "url": item.get("url", ""),
                "content": content,
                "sentiment_score": score,
                "sentiment_label": label
            })

    print("News collected:", len(results))
    print("News sentiment analysis completed!")
    print("Saved to:", output_file)

except Exception as error:
    print("Error:", error)
