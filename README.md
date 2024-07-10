# Stock Price Prediction Using Machine Learning

![image](https://github.com/sai-annadi/Stock-price-prediction/assets/111168434/196f775e-c320-427d-8402-80656605529a)

# Stock Price Prediction Dashboard

### Overview
The Stock Price Prediction Dashboard is an interactive web application built using Streamlit, yfinance, StockNews, Prophet, and Plotly Graph Objects. It provides a comprehensive platform for users to analyze historical stock prices, access relevant news, and predict future stock prices based on historical data. This README will guide you through the features and usage of the dashboard.

![image](https://github.com/sai-annadi/Stock-price-prediction-Using-Machine-Learning/assets/111168434/492ecb07-a07b-4c37-806c-d254013bd129)
![image](https://github.com/sai-annadi/Stock-price-prediction-Using-Machine-Learning/assets/111168434/cda71d04-8c17-4233-a79c-c20c72a54ae0)

### Features
The Stock Price Prediction Dashboard offers the following key features:

1. **Historical Stock Price Data:**
   - The dashboard fetches historical stock price data using the yfinance library from Yahoo Finance.
   - It displays the data as a line plot, allowing users to analyze the stock's performance over time and identify trends.

2. **Stock News:**
   - Utilizes the StockNews library to provide users with the top 10 news articles related to the selected stock ticker.
   - Displays the news sentiment, helping users gauge the overall sentiment surrounding the stock.

3. **Stock Price Prediction:**

   - Utilizes the Prophet library, an open-source time series forecasting tool, to predict future stock prices based on historical data.
   - Visualizes predicted prices alongside the actual historical prices for easy comparison and evaluation of prediction accuracy.
   - Incorporates cross-validation using the TimeSeriesSplit method from sklearn to evaluate the model's performance.
   - Computes Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to assess prediction accuracy.

![image](https://github.com/sai-annadi/Stock-price-prediction-Using-Machine-Learning/assets/111168434/495dded6-2ec8-4b20-82e1-8f9cd0e79bb1)
![image](https://github.com/sai-annadi/Stock-price-prediction-Using-Machine-Learning/assets/111168434/568610ea-5885-48d4-bd90-45a57f1558f0)


### Getting Started

### Installation
To install the necessary libraries, you can use pip. Simply run:
```bash
pip install -r requirements.txt
```

### Running the Application

1.Ensure you have Python installed on your machine.

2.Install the required libraries using the command provided above.

3.Save your Streamlit application code in a file named stock_price_prediction_dashboard.py.

4.Open a terminal or command prompt and navigate to the directory where your stock_price_prediction_dashboard.py file is located.

5.Run the application using the command:
```bash
streamlit run stock.py
```

### Using the Dashboard

1. **Select a Stock Ticker:**
   - Use the dropdown menu to select a stock ticker (e.g., AAPL, GOOGL, MSFT).

2. **Choose a Date Range:**
   - Use the date input fields to select a start and end date for the historical data.

3. **View Historical Data:**
   - The dashboard will display the historical stock prices as a line plot.
   - Additional metrics like annual return, standard deviation, and risk-adjusted return are calculated and displayed.

4. **Explore Stock News:**
   - Navigate to the "Top News" tab to view the latest news articles related to the selected stock.
   - The sentiment for each news article is displayed to help gauge overall sentiment.

5. **Predict Stock Prices:**
   - Navigate to the "Prediction" tab to predict future stock prices.
   - Enter the number of days for which you want to predict the stock prices.
   - The dashboard will display the predicted prices along with historical prices for comparison.
   - The cross-validation results, including MAE and RMSE, are displayed to evaluate the model's performance.
   - Box plots for MAE and RMSE scores are generated to visualize the prediction accuracy.

**Website url:**https://stock-price-predictions.streamlit.app/


