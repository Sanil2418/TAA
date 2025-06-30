import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Stock Closing Price", page_icon="💹", layout="centered")

# --- App Title ---
st.title("📈 Stock Closing Price Viewer")

# --- Input from User ---
ticker_input = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, AAPL, TCS.NS):", value="RELIANCE.NS")

# --- Fetch Data on Button Click ---
if st.button("Fetch Closing Price"):
    if ticker_input:
        try:
            # Define date range (last 7 days to handle weekends/holidays)
            end_date = datetime.today()
            start_date = end_date - timedelta(days=7)

            # Fetch data from yfinance
            stock_data = yf.download(ticker_input, start=start_date, end=end_date)

            # Check if data exists
            if not stock_data.empty and 'Close' in stock_data.columns:
                close_series = stock_data['Close'].dropna()

                if not close_series.empty:
                    latest_close = float(close_series.iloc[-1])
                    latest_date = close_series.index[-1].strftime('%Y-%m-%d')
                    st.success(f"✅ Latest closing price of **{ticker_input}** on **{latest_date}** is ₹{latest_close:.2f}")
                else:
                    st.error("⚠️ Closing price data not available.")
            else:
                st.error("⚠️ No data found for the given ticker.")
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
    else:
        st.warning("⚠️ Please enter a valid stock ticker.")
