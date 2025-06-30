import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import time

# --- Main Data Fetching and Caching Function ---

@st.cache_data(ttl=3600)
def fetch_technical_data(ticker, benchmark_ticker, run_date, years=3, retries=3, delay=2):
    """
    Fetches historical stock data with a retry mechanism to improve reliability.
    The 'run_date' parameter ensures the cache is invalidated daily.
    """
    for attempt in range(retries):
        try:
            # Using yf.Ticker().history() can be more reliable for single tickers
            stock = yf.Ticker(ticker)
            # Fetching a slightly longer period to ensure enough data for calculations
            df = stock.history(period=f"{years+1}y", auto_adjust=True)

            # Check if the primary ticker data was successfully fetched
            if not df.empty and len(df) > 200:
                # If the main ticker is successful, now fetch the benchmark
                bench_df = yf.Ticker(benchmark_ticker).history(period=f"{years+1}y", auto_adjust=True)
                if bench_df.empty:
                     st.warning(f"Failed to fetch benchmark data for {benchmark_ticker} on attempt {attempt + 1}.")
                     time.sleep(delay) # Wait before the next retry
                     continue

                # --- Data successfully fetched, now proceed with calculations ---
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                bench_df = bench_df[['Close']].rename(columns={'Close': 'Benchmark_Close'})
                latest_price = df['Close'].iloc[-1]

                # Performance Calculations
                perf_1m = (latest_price / df['Close'].iloc[-21] - 1) * 100 if len(df) > 21 else np.nan
                perf_3m = (latest_price / df['Close'].iloc[-63] - 1) * 100 if len(df) > 63 else np.nan
                perf_6m = (latest_price / df['Close'].iloc[-126] - 1) * 100 if len(df) > 126 else np.nan
                perf_12m = (latest_price / df['Close'].iloc[-252] - 1) * 100 if len(df) > 252 else np.nan

                # Moving Averages
                for w in [50, 100, 200]:
                    df[f'{w}DMA'] = df['Close'].rolling(window=w).mean()
                    df[f'Dist_{w}DMA_pct'] = (df['Close'] - df[f'{w}DMA']) / df[f'{w}DMA'] * 100

                # Relative Performance
                merged = df[['Close']].merge(bench_df, left_index=True, right_index=True, how='left').fillna(method='ffill')
                rel_perf = {}
                for days, name in zip([63, 126, 252], ['3M', '6M', '12M']):
                    if len(merged) > days and not merged.iloc[-days:].isnull().values.any():
                        stock_ret = (merged['Close'].iloc[-1] / merged['Close'].iloc[-days] - 1) * 100
                        bench_ret = (merged['Benchmark_Close'].iloc[-1] / merged['Benchmark_Close'].iloc[-days] - 1) * 100
                        rel_perf[name] = stock_ret - bench_ret
                    else:
                        rel_perf[name] = np.nan

                # Volatility and Trend Strength
                sd_21d = df['Close'].pct_change().rolling(21).std().iloc[-1] * 100
                r2_50d = np.nan
                if len(df) >= 50:
                    X = np.arange(-49, 1).reshape(-1, 1)
                    y = df['Close'].iloc[-50:].values.reshape(-1, 1)
                    reg = LinearRegression().fit(X, y)
                    r2_50d = reg.score(X, y)

                # Highs and Lows
                high_52w = df['High'].rolling(252).max().iloc[-1] if len(df) >= 252 else np.nan
                low_52w = df['Low'].rolling(252).min().iloc[-1] if len(df) >= 252 else np.nan
                ath = df['High'].max()

                # Volume
                avg_vol_20d = df['Volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else np.nan
                last_vol = df['Volume'].iloc[-1]

                # --- Assemble Final Dictionary for Display ---
                latest_stats = {
                    'Latest_Date': df.index[-1].strftime('%Y-%m-%d'),
                    'Latest_Price': latest_price,
                    '1M_Return_%': perf_1m,
                    '3M_Return_%': perf_3m,
                    '6M_Return_%': perf_6m,
                    '12M_Return_%': perf_12m,
                    '50_DMA': df['50DMA'].iloc[-1],
                    'Dist_from_50DMA_%': df['Dist_50DMA_pct'].iloc[-1],
                    '200_DMA': df['200DMA'].iloc[-1],
                    'Dist_from_200DMA_%': df['Dist_200DMA_pct'].iloc[-1],
                    'Rel_Perf_3M_%': rel_perf.get('3M', np.nan),
                    'Rel_Perf_12M_%': rel_perf.get('12M', np.nan),
                    'Volatility_21D_%': sd_21d,
                    'Trend_Strength_50D_R2': r2_50d,
                    '52W_High': high_52w,
                    'Pct_off_52W_High': (latest_price / high_52w - 1) * 100 if high_52w else np.nan,
                    '52W_Low': low_52w,
                    'All_Time_High': ath,
                    'Last_Volume': last_vol,
                    'Avg_Vol_20D': avg_vol_20d,
                }
                # On success, exit the loop and return the DataFrame
                return pd.DataFrame([latest_stats]).T.rename(columns={0: 'Value'})

            else: # If df is empty or has insufficient data
                st.warning(f"Attempt {attempt + 1}/{retries}: No sufficient data for {ticker}. Retrying in {delay} seconds...")
                time.sleep(delay)

        except Exception as e:
            st.warning(f"An error occurred on attempt {attempt + 1}: {e}. Retrying...")
            time.sleep(delay)

    # This part is reached only if all retries fail
    st.error(f"Could not retrieve sufficient historical data for **{ticker}** after {retries} attempts. The service may be temporarily unavailable.")
    return None


# --- Streamlit App Interface ---
st.set_page_config(page_title="Stock Technical Dashboard", page_icon="📊", layout="wide")
st.title("📊 Technical Analysis Dashboard")
st.caption("Data is sourced from Yahoo Finance. If a valid ticker fails, the app will retry automatically.")

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., BAJAJ-AUTO.NS):", value="BAJAJ-AUTO.NS")
with col2:
    benchmark_ticker = st.text_input("Enter Benchmark Ticker (e.g., ^NSEI):", value="^NSEI")

# --- Fetch and Display Data ---
if st.button("📈 Generate Technical Snapshot"):
    with st.spinner("Fetching and analyzing data... Please wait."):
        # Pass the current date to the function to ensure the cache updates daily
        today_date_str = datetime.today().strftime('%Y-%m-%d')
        data_df = fetch_technical_data(stock_ticker, benchmark_ticker, run_date=today_date_str)

    if data_df is not None and not data_df.empty:
        st.success(f"✅ Analysis complete for **{stock_ticker}**")
        latest_date_str = data_df.loc['Latest_Date', 'Value']
        st.info(f"All metrics are calculated as of the market close on **{latest_date_str}**.")
        # Display the formatted DataFrame
        st.dataframe(data_df.style.format(precision=2, na_rep="-"), use_container_width=True)
    # Error messages are now handled inside the fetch_technical_data function
