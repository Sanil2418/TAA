import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# --- Technical Data Function using yf.download ---
def fetch_technical_data(ticker, benchmark_ticker, years=3):
    try:
        # Define time window
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * years)

        # Download historical data
        df = yf.download(ticker, start=start_date, end=end_date)
        bench_df = yf.download(benchmark_ticker, start=start_date, end=end_date)

        if df.empty:
            return "error::No data for stock ticker"
        if bench_df.empty:
            return "error::No data for benchmark ticker"

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        bench_df = bench_df[['Close']].rename(columns={'Close': 'Benchmark_Close'})

        # --- Price Performance ---
        latest = float(df['Close'].iloc[-1])
        perf_1m = float((latest / df['Close'].iloc[-21] - 1) * 100) if len(df) > 21 else np.nan
        perf_3m = float((latest / df['Close'].iloc[-63] - 1) * 100) if len(df) > 63 else np.nan
        perf_6m = float((latest / df['Close'].iloc[-126] - 1) * 100) if len(df) > 126 else np.nan
        perf_12m = float((latest / df['Close'].iloc[-252] - 1) * 100) if len(df) > 252 else np.nan

        # --- Moving Averages ---
        for w in [50, 100, 200]:
            dma = df['Close'].rolling(window=w).mean()
            dist_abs = df['Close'] - dma
            dist_pct = (df['Close'] - dma) / dma * 100

            df[f'{w}DMA'] = dma
            df[f'Dist_{w}DMA_abs'] = dist_abs
            df[f'Dist_{w}DMA_pct'] = dist_pct

        # --- Relative Performance ---
        merged = df[['Close']].merge(bench_df, left_index=True, right_index=True, how='left')
        rel_perf = {}
        for days, name in zip([63, 126, 252], ['3M', '6M', '12M']):
            if len(merged) > days:
                stock_ret = (merged['Close'].iloc[-1] / merged['Close'].iloc[-days] - 1) * 100
                bench_ret = (merged['Benchmark_Close'].iloc[-1] / merged['Benchmark_Close'].iloc[-days] - 1) * 100
                rel_perf[name] = float(stock_ret - bench_ret)
            else:
                rel_perf[name] = np.nan

        # --- Monthly Range & Volatility ---
        df['Month'] = df.index.to_period('M')
        monthly_range = df.groupby('Month').apply(lambda x: x['High'].max() - x['Low'].min())
        last_month_range = float(monthly_range.iloc[-1]) if len(monthly_range) > 0 else np.nan
        atr_14d = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
        atr_14d_last = float(atr_14d.iloc[-1]) if len(atr_14d) > 0 else np.nan
        sd_21d = float(df['Close'].pct_change().rolling(21).std().iloc[-1] * 100)

        # --- R² Trend Smoothness ---
        X = np.arange(-49, 1).reshape(-1, 1)
        if len(df) >= 50:
            y = df['Close'].iloc[-50:].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            r2_50d = float(reg.score(X, y))
        else:
            r2_50d = np.nan

        # --- Multi-Year Highs & Lows ---
        high_52w = float(df['High'].iloc[-252:].max()) if len(df) > 252 else float(df['High'].max())
        low_52w = float(df['Low'].iloc[-252:].min()) if len(df) > 252 else float(df['Low'].min())
        high_2y = float(df['High'].iloc[-504:].max()) if len(df) > 504 else float(df['High'].max())
        low_2y = float(df['Low'].iloc[-504:].min()) if len(df) > 504 else float(df['Low'].min())
        ath = float(df['High'].max())
        atl = float(df['Low'].min())

        # --- Volume Stats ---
        avg_vol_20d = float(df['Volume'].rolling(20).mean().iloc[-1]) if len(df) > 20 else np.nan
        avg_vol_50d = float(df['Volume'].rolling(50).mean().iloc[-1]) if len(df) > 50 else np.nan
        last_vol = float(df['Volume'].iloc[-1])

        # --- Compile Data ---
        latest_stats = {
            'Latest_Price': latest,
            '1M_return_pct': perf_1m,
            '3M_return_pct': perf_3m,
            '6M_return_pct': perf_6m,
            '12M_return_pct': perf_12m,
            '50DMA': float(df['50DMA'].iloc[-1]),
            'Dist_50DMA_abs': float(df['Dist_50DMA_abs'].iloc[-1]),
            'Dist_50DMA_pct': float(df['Dist_50DMA_pct'].iloc[-1]),
            '100DMA': float(df['100DMA'].iloc[-1]),
            'Dist_100DMA_abs': float(df['Dist_100DMA_abs'].iloc[-1]),
            'Dist_100DMA_pct': float(df['Dist_100DMA_pct'].iloc[-1]),
            '200DMA': float(df['200DMA'].iloc[-1]),
            'Dist_200DMA_abs': float(df['Dist_200DMA_abs'].iloc[-1]),
            'Dist_200DMA_pct': float(df['Dist_200DMA_pct'].iloc[-1]),
            'Relative_performance_3M': rel_perf['3M'],
            'Relative_performance_6M': rel_perf['6M'],
            'Relative_performance_12M': rel_perf['12M'],
            'Monthly_High_Low_Range': last_month_range,
            'ATR_14D': atr_14d_last,
            'SD_21D_pct': sd_21d,
            'R2_50D': r2_50d,
            '52W_High': high_52w,
            '52W_Low': low_52w,
            '2Y_High': high_2y,
            '2Y_Low': low_2y,
            'AllTime_High': ath,
            'AllTime_Low': atl,
            'Last_Volume': last_vol,
            'AvgVol_20D': avg_vol_20d,
            'AvgVol_50D': avg_vol_50d,
        }

        return pd.DataFrame([latest_stats])

    except Exception as e:
        return f"error::{str(e)}"

# --- Streamlit UI ---
st.set_page_config(page_title="📊 Technical Dashboard", layout="wide")
st.title("📊 Technical Analysis Dashboard")

col1, col2 = st.columns(2)
with col1:
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS)", value="RELIANCE.NS")
with col2:
    benchmark_ticker = st.text_input("Enter Benchmark Ticker (e.g., NIFTYBEES.NS)", value="NIFTYBEES.NS")

# Fetch on button click
if st.button("📥 Fetch Data"):
    with st.spinner("Fetching and analyzing data..."):
        result_df = fetch_technical_data(stock_ticker, benchmark_ticker)

    if isinstance(result_df, pd.DataFrame):
        st.success("✅ Data fetched successfully!")

        format_dict = {col: "{:,.2f}" for col in result_df.columns}
        st.dataframe(result_df.style.format(format_dict), use_container_width=True)
    elif isinstance(result_df, str) and result_df.startswith("error::"):
        st.error(f"❌ Error occurred: {result_df[7:]}")
    else:
        st.error("❌ Unknown error. Please try again.")
