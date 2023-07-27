import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from stocknews import StockNews
from datetime import datetime
import plotly.graph_objects as go
from prophet import Prophet
st.set_page_config(page_title="Stock prices", page_icon=":minidisc:", layout="wide")
st.title("Stock Dashboard")
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")
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
    df1=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    st.dataframe(df1,use_container_width=True)
    if days <= 30:
           start_date = hist['ds'].max() - pd.DateOffset(months=5)
    elif days <= 365:
          start_date = hist['ds'].max() - pd.DateOffset(years=1)
    else:
          start_date = hist['ds'].min()                           
                    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['ds'], y=hist['y'], mode='lines', name="Actual Close Prices",line=dict(color='red')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name="Predicted Close Prices",line=dict(color='green')))
    fig.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast['yhat_upper'],
                        mode='lines',
                        line=dict(color='rgba(255, 0, 0, 0)'),
                        fill='tozeroy',
                        fillcolor='rgba(255, 182, 193, 0.2)',
                        name="Prediction Interval"
                    ))  
    fig.update_layout(title="Actual and Predicted Close Prices",
                                    xaxis_title="Date", yaxis_title="Close Price",
                                    xaxis_range=[start_date, forecast['ds'].max()])
    st.plotly_chart(fig, use_container_width=True)              
                    
min_date = datetime(1995, 1, 1).date()
if 'session_state' not in st.session_state:
    st.session_state.session_state = {
        'ticker': None,
        'start_date': None,
        'end_date': None,
        'days': 0
    }
col1, col2, col3 = st.columns(3)
with col1:
    ticker_options = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    ticker = st.selectbox(label = "Choose a ticker", options = ticker_options, help='Please select a ticker from the dropdown.')
    if not ticker:
        st.warning('Please select a ticker from the dropdown.')
with col2:
    start_date = st.date_input("Start Date", min_value=min_date, max_value=datetime.today().date())
with col3:
    end_date = st.date_input("end_date")
with col2:
    submit_button = st.button("Submit")

if submit_button:
    if ticker not in ticker_options:
        st.warning("Please select a valid ticker from the dropdown.")
    else:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data available for the given ticker and date range.")
        elif start_date == end_date:
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

        st.markdown(f'<p class="big2-font">PLOTS FOR {ticker}</p>', unsafe_allow_html=True) 
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='close Price', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open Price', line=dict(color='blue')))
        fig.update_layout(title=f"{ticker} - Plots of price", xaxis_title="Date", yaxis_title=" Price")
        st.plotly_chart(fig, use_container_width=True)
        pricing_data, News, prediction = st.tabs(['Pricing Data', 'Top News', 'Prediction'])
        with pricing_data:
            st.write(f"<p class='big2-font'>Prices movements of {ticker}</p>", unsafe_allow_html=True)
            df = data
            df['% change'] = df['Adj Close'] / data['Adj Close'].shift(1) - 1
            df.dropna(inplace=True)
            st.dataframe(df, use_container_width=True)
            annual_return = df['% change'].mean() * 252 * 100
            st.write("Annual return: {:.2f}%".format(annual_return))
            stdev = np.std(df['% change']) * np.sqrt(252)
            st.write("standard deviation: {:.2f}%".format(stdev * 100))
            st.write("risk Adj Return is :{:.2f}%".format(annual_return / (stdev * 100)))
        with News:
            st.write(f"<p class='big2-font'>Top News related to {ticker}</p>", unsafe_allow_html=True)
            sn = StockNews(ticker, save_news=False)
            news = sn.read_rss()
            for i in range(10):
                st.subheader(f"news{i+1}")
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
            st.write(f"<p class='big2-font'>Prediction of {ticker}</p>", unsafe_allow_html=True)
            st.write("<p class='big3-font'>HINT:FOR THE PREDICTION TRY TO GIVE MORE DATA </p>", unsafe_allow_html=True)
            col1,col2,col3 = st.columns(3)
            with col1:
                st.write("")
            with col2:
                days = st.number_input(label="Enter the Number of days", value=0, step=1, format="%d")
                
            with col3:
                st.write("")
            predict(ticker, days)