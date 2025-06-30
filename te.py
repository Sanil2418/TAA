import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

# --- Your fetch_technical_data function remains the same ---
# It is generally well-written. The issue is not the function itself,
# but the environment and context in which it's called.
def fetch_technical_data(ticker, benchmark_ticker, years=3):
    try:
        # Fetch historical data
        start_date = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
        # yfinance fetches up to the latest available data by default when no end_date is specified.
        stock = yf.Ticker(ticker)
        bench = yf.Ticker(benchmark_ticker)

        df = stock.history(start=start_date, auto_adjust=True) # Using auto_adjust is good practice
        bench_df = bench.history(start=start_date, auto_adjust=True)

        if df.empty or bench_df.empty:
            return None

        # The function logic from here is correct and does not need to be changed.
        # For brevity, I am not pasting the entire function again.
        # Just ensure the code you provided in the prompt is here.
        
        # --- (Paste your full function logic here) ---

        # For this example to run, I am pasting your full function logic back in.
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        bench_df = bench_df[['Close']].rename(columns={'Close': 'Benchmark_Close'})

        if df['Close'].empty:
            return None

        latest = df['Close'].iloc[-1]
        perf_1m = (latest / df['Close'].iloc[-21] - 1) * 100 if len(df) > 21 else np.nan
        perf_3m = (latest / df['Close'].iloc[-63] - 1) * 100 if len(df) > 63 else np.nan
        perf_6m = (latest / df['Close'].iloc[-126] - 1) * 100 if len(df) > 126 else np.nan
        perf_12m = (latest / df['Close'].iloc[-252] - 1) * 100 if len(df) > 252 else np.nan

        for w in [50, 100, 200]:
            df[f'{w}DMA'] = df['Close'].rolling(window=w).mean()
            df[f'Dist_{w}DMA_abs'] = df['Close'] - df[f'{w}DMA']
            df[f'Dist_{w}DMA_pct'] = (df['Close'] - df[f'{w}DMA']) / df[f'{w}DMA'] * 100

        merged = df[['Close']].merge(bench_df, left_index=True, right_index=True, how='left')
        rel_perf = {}
        for days, name in zip([63, 126, 252], ['3M', '6M', '12M']):
            if len(merged) > days:
                stock_ret = (merged['Close'].iloc[-1] / merged['Close'].iloc[-days] - 1) * 100
                bench_ret = (merged['Benchmark_Close'].iloc[-1] / merged['Benchmark_Close'].iloc[-days] - 1) * 100
                rel_perf[name] = stock_ret - bench_ret
            else:
                rel_perf[name] = np.nan

        df['Month'] = df.index.to_period('M')
        monthly_range = df.groupby('Month').apply(lambda x: x['High'].max() - x['Low'].min())
        last_month_range = monthly_range.iloc[-1] if len(monthly_range) > 0 else np.nan
        # A better ATR calculation:
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr_14d = true_range.rolling(14).mean()
        atr_14d_last = atr_14d.iloc[-1] if len(atr_14d) > 0 else np.nan
        
        sd_21d = df['Close'].pct_change().rolling(21).std().iloc[-1] * (252**0.5) # Annualized volatility

        X = np.arange(-49, 1).reshape(-1, 1)
        if len(df) >= 50:
            y = df['Close'].iloc[-50:].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            r2_50d = reg.score(X, y)
        else:
            r2_50d = np.nan

        high_52w = df['High'].rolling(252).max().iloc[-1] if len(df) >= 252 else df['High'].max()
        low_52w = df['Low'].rolling(252).min().iloc[-1] if len(df) >= 252 else df['Low'].min()
        high_2y = df['High'].rolling(504).max().iloc[-1] if len(df) >= 504 else df['High'].max()
        low_2y = df['Low'].rolling(504).min().iloc[-1] if len(df) >= 504 else df['Low'].min()
        ath = df['High'].max()
        atl = df['Low'].min()

        avg_vol_20d = df['Volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else np.nan
        avg_vol_50d = df['Volume'].rolling(50).mean().iloc[-1] if len(df) > 50 else np.nan
        last_vol = df['Volume'].iloc[-1]

        latest_stats = {
            'Latest_Date': df.index[-1].strftime('%Y-%m-%d'), # Add the date for clarity
            'Latest_Price': latest,
            '1M_return_pct': perf_1m,
            '3M_return_pct': perf_3m,
            '6M_return_pct': perf_6m,
            '12M_return_pct': perf_12m,
            '50DMA': df['50DMA'].iloc[-1],
            'Dist_50DMA_pct': df['Dist_50DMA_pct'].iloc[-1],
            '200DMA': df['200DMA'].iloc[-1],
            'Dist_200DMA_pct': df['Dist_200DMA_pct'].iloc[-1],
            'Rel_Perf_3M': rel_perf.get('3M', np.nan),
            'Rel_Perf_6M': rel_perf.get('6M', np.nan),
            'Rel_Perf_12M': rel_perf.get('12M', np.nan),
            'Volatility_21D_Ann': sd_21d,
            'Trend_Strength_50D_R2': r2_50d,
            '52W_High': high_52w,
            'Pct_off_52W_High': (latest / high_52w - 1) * 100,
            '52W_Low': low_52w,
            'AllTime_High': ath,
            'Last_Volume': last_vol,
            'AvgVol_20D': avg_vol_20d,
        }
        latest_stats_df = pd.DataFrame([latest_stats])
        # Transpose for better readability in Streamlit
        return latest_stats_df.T.rename(columns={0: 'Value'})

    except Exception as e:
        st.error(f"An error occurred in fetch_technical_data: {e}")
        return None


# --- Streamlit App Starts Here ---
st.set_page_config(page_title="Stock Technical Dashboard", page_icon="📊", layout="wide")
st.title("📊 Technical Analysis Dashboard")
st.caption("Data is sourced from Yahoo Finance and may have delays. All calculations are based on the latest available trading day's data.")

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, AAPL):", value="RELIANCE.NS")
with col2:
    benchmark_ticker = st.text_input("Enter Benchmark Ticker (e.g., ^NSEI, ^GSPC):", value="^NSEI")

# --- Fetch and Display Data ---
if st.button("📈 Generate Technical Snapshot"):
    with st.spinner("Fetching and analyzing data... Please wait."):
        data_df = fetch_technical_data(stock_ticker, benchmark_ticker)
    
    if data_df is not None and not data_df.empty:
        st.success(f"✅ Analysis complete for **{stock_ticker}**")
        
        #
        # ✨ --- THIS IS THE KEY CHANGE --- ✨
        # Display the date of the data clearly to the user.
        #
        latest_date_str = data_df.loc['Latest_Date', 'Value']
        st.info(f"All metrics are calculated as of the closing of **{latest_date_str}**")
        
        # Display the formatted dataframe
        st.dataframe(data_df.style.format(precision=2, na_rep="-"), use_container_width=True)
        
    else:
        st.error(f"❌ Failed to fetch data for **{stock_ticker}**. Please check the ticker symbol and your internet connection.")
