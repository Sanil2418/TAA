import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# --- Main Data Fetching and Caching Function ---

@st.cache_data(ttl=3600) # Cache data for 1 hour (3600 seconds) to prevent API rate-limiting
def fetch_technical_data(ticker, benchmark_ticker, years=3):
    """
    Fetches, calculates, and returns a DataFrame of technical metrics for a stock.
    The results are cached.
    """
    try:
        # Fetch historical data using yfinance
        start_date = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
        stock = yf.Ticker(ticker)
        bench = yf.Ticker(benchmark_ticker)

        df = stock.history(start=start_date)
        bench_df = bench.history(start=start_date)

        if df.empty or bench_df.empty:
            st.warning(f"No data returned for {ticker} or {benchmark_ticker}. Ticker might be invalid.")
            return None

        # --- Calculations ---
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        bench_df = bench_df[['Close']].rename(columns={'Close': 'Benchmark_Close'})

        if df['Close'].empty:
            return None

        latest_price = df['Close'].iloc[-1]

        # Performance
        perf_1m = (latest_price / df['Close'].iloc[-21] - 1) * 100 if len(df) > 21 else np.nan
        perf_3m = (latest_price / df['Close'].iloc[-63] - 1) * 100 if len(df) > 63 else np.nan
        perf_6m = (latest_price / df['Close'].iloc[-126] - 1) * 100 if len(df) > 126 else np.nan
        perf_12m = (latest_price / df['Close'].iloc[-252] - 1) * 100 if len(df) > 252 else np.nan

        # Moving Averages
        for w in [50, 100, 200]:
            df[f'{w}DMA'] = df['Close'].rolling(window=w).mean()
            df[f'Dist_{w}DMA_pct'] = (df['Close'] - df[f'{w}DMA']) / df[f'{w}DMA'] * 100

        # Relative Performance
        merged = df[['Close']].merge(bench_df, left_index=True, right_index=True, how='left')
        rel_perf = {}
        for days, name in zip([63, 126, 252], ['3M', '6M', '12M']):
            if len(merged) > days:
                stock_ret = (merged['Close'].iloc[-1] / merged['Close'].iloc[-days] - 1) * 100
                bench_ret = (merged['Benchmark_Close'].iloc[-1] / merged['Benchmark_Close'].iloc[-days] - 1) * 100
                rel_perf[name] = stock_ret - bench_ret
            else:
                rel_perf[name] = np.nan

        # Volatility and Trend
        sd_21d = df['Close'].pct_change().rolling(21).std().iloc[-1] * 100
        X = np.arange(-49, 1).reshape(-1, 1)
        r2_50d = np.nan
        if len(df) >= 50:
            y = df['Close'].iloc[-50:].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            r2_50d = reg.score(X, y)

        # Highs and Lows
        high_52w = df['High'].rolling(252).max().iloc[-1] if len(df) >= 252 else df['High'].max()
        low_52w = df['Low'].rolling(252).min().iloc[-1] if len(df) >= 252 else df['Low'].min()
        ath = df['High'].max()

        # Volume
        avg_vol_20d = df['Volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else np.nan
        last_vol = df['Volume'].iloc[-1]

        # --- Assemble Final Dictionary ---
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
            'Rel_Perf_6M_%': rel_perf.get('6M', np.nan),
            'Rel_Perf_12M_%': rel_perf.get('12M', np.nan),
            'Volatility_21D_%': sd_21d,
            'Trend_Strength_50D_R2': r2_50d,
            '52W_High': high_52w,
            'Pct_off_52W_High': (latest_price / high_52w - 1) * 100,
            '52W_Low': low_52w,
            'All_Time_High': ath,
            'Last_Volume': last_vol,
            'Avg_Vol_20D': avg_vol_20d,
        }

        # Transpose the DataFrame for better readability
        latest_stats_df = pd.DataFrame([latest_stats])
        return latest_stats_df.T.rename(columns={0: 'Value'})

    except Exception as e:
        st.error(f"An error occurred in fetch_technical_data: {e}")
        return None

# --- Streamlit App Interface ---
st.set_page_config(page_title="Stock Technical Dashboard", page_icon="📊", layout="wide")
st.title("📊 Technical Analysis Dashboard")
st.caption("Data is sourced from Yahoo Finance. Caching is used to prevent rate-limiting.")

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    # Changed default to a common US stock and index to be more universal
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS):", value="AAPL")
with col2:
    # Using SPY as a common benchmark for US stocks
    benchmark_ticker = st.text_input("Enter Benchmark Ticker (e.g., SPY, ^NSEI):", value="SPY")

# --- Fetch and Display Data ---
if st.button("📈 Generate Technical Snapshot"):
    # Show a spinner while the function runs for the first time
    with st.spinner("Fetching and analyzing data... Please wait."):
        data_df = fetch_technical_data(stock_ticker, benchmark_ticker)
    
    if data_df is not None and not data_df.empty:
        st.success(f"✅ Analysis complete for **{stock_ticker}**")
        
        # Display the date of the data clearly to the user
        latest_date_str = data_df.loc['Latest_Date', 'Value']
        st.info(f"All metrics are calculated as of the market close on **{latest_date_str}**.")
        
        # Display the formatted dataframe
        st.dataframe(data_df.style.format(precision=2, na_rep="-"), use_container_width=True)
        
    else:
        st.error(f"❌ Failed to fetch data for **{stock_ticker}**. Please check the ticker symbol.")
