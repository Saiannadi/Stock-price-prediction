import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from stocknews import StockNews
from datetime import datetime
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Stock prices", page_icon="chart_with_upwards_trend", layout="wide")
st.title("Stock Dashboard")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

def prophet_cv(data, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mae_scores = []
    rmse_scores = []
    
    for train_index, test_index in tscv.split(data):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        
        m = Prophet(daily_seasonality=True)
        m.fit(train)
        
        future = m.make_future_dataframe(periods=len(test))
        forecast = m.predict(future)
        
        y_true = test['y'].values
        y_pred = forecast.iloc[-len(test):]['yhat'].values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
    
    return mae_scores, rmse_scores
def predict(ticker, days):
    yfin = yf.Ticker(ticker)
    hist = yfin.history(period="max")
    hist = hist[['Close']]
    hist.reset_index(level=0, inplace=True)
    hist = hist.rename({'Date': 'ds', 'Close': 'y'}, axis='columns')
    hist['ds'] = hist['ds'].dt.tz_localize(None)
    m = Prophet(daily_seasonality=True)
    m.fit(hist)
    last_observed_date = hist['ds'].max()
    next_business_days = pd.bdate_range(start=last_observed_date + pd.Timedelta(days=1), periods=days)
    future = pd.DataFrame({"ds": next_business_days})
    if len(next_business_days) == 0:
        st.error("No future dates available for prediction.")
        return
    forecast = m.predict(future)
    st.write("Predicted Data")
    df1 = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    st.dataframe(df1, use_container_width=True)
    if days <= 30:
        start_date = hist['ds'].max() - pd.DateOffset(months=5)
    elif days <= 365:
        start_date = hist['ds'].max() - pd.DateOffset(years=1)
    else:
        start_date = hist['ds'].min()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], mode='lines', name="Actual Close Prices", line=dict(color='red')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name="Predicted Close Prices", line=dict(color='green')))

    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(color='hsla(0, 100%, 30%, 0.3)'), fill='tonexty', fillcolor='rgba(255, 182, 193, 0.4)', name="Prediction Upper Interval"))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(color='hsla(0, 100%, 30%, 0.3)'), fill='tonexty', fillcolor='rgba(255, 182, 193, 0.2)', name="Prediction Lower Interval"))

    fig.update_layout(title="Actual and Predicted Close Prices with Prediction Interval",
                    xaxis_title="Date", yaxis_title="Close Price",
                    xaxis_range=[start_date, forecast['ds'].max()])
    st.plotly_chart(fig, use_container_width=True)
    # Perform cross-validation
    mae_scores, rmse_scores = prophet_cv(hist)
    
    st.write("Cross-Validation Results:")
    st.write(f"Mean Absolute Error: {np.mean(mae_scores):.2f} (+/- {np.std(mae_scores):.2f})")
    st.write(f"Root Mean Squared Error: {np.mean(rmse_scores):.2f} (+/- {np.std(rmse_scores):.2f})")
    
    # Create box plots for MAE and RMSE
    fig = go.Figure()
    fig.add_trace(go.Box(y=mae_scores, name="MAE"))
    fig.add_trace(go.Box(y=rmse_scores, name="RMSE"))
    fig.update_layout(title="Cross-Validation Scores", yaxis_title="Score")
    st.plotly_chart(fig)
min_date = datetime(1995, 1, 1).date()

if 'session_state' not in st.session_state:
    st.session_state.ticker = None
    st.session_state.start_date = None
    st.session_state.end_date = None
    st.session_state.days = 5 # Default value for prediction days

col1, col2, col3 = st.columns(3)

with col1:
    ticker_options = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA','NVDA','AMD','META','UBER',]
    ticker = st.session_state.ticker or ticker_options[0]
    st.session_state.ticker = st.selectbox(label="Choose a ticker", options=ticker_options, help='Please select a ticker from the dropdown.')

with col2:
    start_date = st.session_state.start_date or min_date
    st.session_state.start_date = st.date_input("Start Date", min_value=min_date, max_value=datetime.today().date(), value=start_date)

with col3:
    end_date = st.session_state.end_date or datetime.today().date()
    st.session_state.end_date = st.date_input("End Date", value=end_date)

if st.session_state.ticker not in ticker_options:        
    st.warning("Please select a valid ticker from the dropdown.")
else:
    data = yf.download(st.session_state.ticker, start=st.session_state.start_date, end=st.session_state.end_date)
    if data.empty:
        st.error("No data available for the given ticker and date range.")
    elif st.session_state.start_date == st.session_state.end_date:
        st.error("Start date and end date cannot be the same.")
    else:
        st.markdown("""
            <style>
            .big2-font {
                font-size: 25px;
                text-align: center;
                margin-bottom: 20px;
            }
            .centered-dataframe {
                display: flex;
                justify-content: center;
            }
            </style>
            """, unsafe_allow_html=True)

        st.markdown(f'<p class="big2-font">PLOTS FOR {st.session_state.ticker}</p>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open Price', line=dict(color='blue')))
        fig.update_layout(title=f"{st.session_state.ticker} - Plots of price", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)
        pricing_data, News, prediction = st.tabs(['Pricing Data', 'Top News', 'Prediction'])
        with pricing_data:
            st.write(f"<p class='big2-font'>Prices movements of {st.session_state.ticker}</p>", unsafe_allow_html=True)
            df = data
            df['% change'] = df['Adj Close'] / data['Adj Close'].shift(1) - 1
            df.dropna(inplace=True)
            st.dataframe(df, use_container_width=True)
            annual_return = df['% change'].mean() * 252 * 100
            st.write("Annual return: {:.2f}%".format(annual_return))
            stdev = np.std(df['% change']) * np.sqrt(252)
            st.write("Standard deviation: {:.2f}%".format(stdev * 100))
            st.write("Risk Adjusted Return is :{:.2f}%".format(annual_return / (stdev * 100)))
        with News:
            st.write(f"<p class='big2-font'>Top News related to {st.session_state.ticker}</p>", unsafe_allow_html=True)
            sn = StockNews(st.session_state.ticker, save_news=False)
            news = sn.read_rss()
            for i in range(10):
                st.subheader(f"News {i+1}")
                st.write(news['published'][i])
                st.write(news['title'][i])
                st.write(news['summary'][i])
                title_sentiment = news['sentiment_title'][i]
                st.write(f"Title sentiment: {title_sentiment}")
                news_sentiment = news['sentiment_summary'][i]
                st.write(f"News sentiment: {news_sentiment}")
        with prediction:
            st.markdown("""
                <style>
                .big3-font {
                    font-size: 10px;
                    text-align: center;
                    margin-bottom: 20px;
                }
                .centered-dataframe {
                    display: flex;
                    justify-content: center;
                }
                </style>
                """, unsafe_allow_html=True)
            st.write(f"<p class='big2-font'>Prediction of {st.session_state.ticker}</p>", unsafe_allow_html=True)
            st.write("<p class='big3-font'>HINT: FOR THE PREDICTION TRY TO GIVE MORE DATA </p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("")
            with col2:
                days =  st.number_input(label="Enter the Number of Days for Prediction", min_value=1, max_value=365, value=st.session_state.days)
                st.session_state.days = days
            with col3:
                st.write("")
            predict(st.session_state.ticker, st.session_state.days)
