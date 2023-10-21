# Stock-price-prediction

![image](https://github.com/sai-annadi/Stock-price-prediction/assets/111168434/196f775e-c320-427d-8402-80656605529a)

# Stock Price Prediction Dashboard

## Overview

The Stock Price Prediction Dashboard is an interactive web application built using Streamlit, yfinance, StockNews, Prophet, and Plotly Graph Objects. It provides a comprehensive platform for users to analyze historical stock prices, access relevant news, and predict future stock prices based on historical data. This README will guide you through the features and usage of the dashboard.

## Features

The Stock Price Prediction Dashboard offers the following key features:

### 1. Historical Stock Price Data

- The dashboard fetches historical stock price data using the yfinance library from Yahoo Finance.
- It displays the data as a line plot, allowing users to analyze the stock's performance over time and identify trends.

### 2. Stock News

- Utilizes the StockNews library to provide users with the top 10 news articles related to the selected stock ticker.
- Displays the news sentiment, helping users gauge the overall sentiment surrounding the stock.

### 3. Stock Price Prediction

- Utilizes the Prophet library, an open-source time series forecasting tool, to predict future stock prices based on historical data.
- Visualizes predicted prices alongside the actual historical prices for easy comparison and evaluation of prediction accuracy.

## Getting Started

Follow these steps to get started with the Stock Price Prediction Dashboard:

1. **Installation:**Set up the required Python libraries.
2. **Usage:** Run the dashboard and explore stock data and predictions.

## Installation

To install the necessary libraries, you can use `pip`. Simply run:

```bash
pip install streamlit yfinance stocknews prophet plotly
```

**Website url:**https://stock-price-predictions.streamlit.app/


