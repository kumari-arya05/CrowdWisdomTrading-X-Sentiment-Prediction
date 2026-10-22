
import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    print("API key nahi mili. .env file check karo.")
else:
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query="Latest Bitcoin news",
            max_results=3
        )
        for result in response.get("results", []):
            print("Title:", result.get("title"))
            print("URL:", result.get("url"))
            print()
    except Exception as e:
        print("Error:", e)
