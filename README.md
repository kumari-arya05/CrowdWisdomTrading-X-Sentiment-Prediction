@'
# CrowdWisdomTrading X Sentiment Price Prediction Agent

## Project Overview
A Python-based prototype for Bitcoin price prediction and sentiment analysis.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- VADER Sentiment Analysis
- yfinance
- Matplotlib

## Project Files
- main.py: Setup test
- features.py: Price feature engineering
- model.py: Train six baseline models
- predict.py: Generate price predictions
- sentiment.py: Sample sentiment analysis
- combine_data.py: Combine price and sentiment data
- visualize.py: Generate charts
- evaluate.py: Preliminary strategy evaluation

## How to Run
Install dependencies:
py -m pip install pandas numpy scikit-learn matplotlib yfinance vaderSentiment requests joblib

Run the scripts individually in VS Code Terminal.

## Limitations
The current sentiment analysis uses sample posts, not real X data.
Real Apify data collection, news integration, and rigorous walk-forward validation remain to be implemented.

The results are experimental and are not investment advice.
'@ | Set-Content README.md