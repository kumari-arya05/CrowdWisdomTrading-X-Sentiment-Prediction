# CrowdWisdomTrading — X Sentiment Price Prediction

## 1. Project Overview
This project explores Bitcoin price forecasting using historical price data, X post sentiment, and cryptocurrency news sentiment.

## 2. Dataset
- Historical Bitcoin price rows: 179
- News records matched to price dates: 10
- X post data: Limited; the sample is insufficient to represent overall market sentiment.

## 3. Feature Engineering
The dataset combines historical OHLCV price data, X sentiment scores, post counts, and daily news sentiment features.

## 4. Walk-Forward Evaluation

| Forecast Horizon | Observations | MAE | Preliminary Sharpe Ratio |
|---|---:|---:|---:|
| n+1 | 88 | 9365.25 | 6.68 |
| n+2 | 87 | 9263.67 | 5.97 |
| n+3 | 86 | 9324.49 | 4.10 |
| n+4 | 85 | 9326.41 | 5.78 |
| n+5 | 84 | 9353.57 | 4.86 |

## 5. Limitations
The X post sample is very small, and only 10 news records matched the price dates. Trading metrics are preliminary and do not establish reliable or profitable trading performance.

## 6. Conclusion
This project demonstrates a preliminary workflow for combining price, social sentiment, and news data. Results are experimental and should not be treated as investment advice.
