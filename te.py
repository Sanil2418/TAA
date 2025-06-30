import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# --- Utility: Ensure scalar values ---
def to_scalar(val):
    try:
        return float(val)
    except:
        return float(val.iloc[0]) if hasattr(val, 'iloc') else np.nan

# --- Main data fetching function ---
def fetch_technical_data(ticker, benchmark_ticker, years=3):
    try:
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365 * years)

        # Fetch stock and benchmark data
        df = yf.download(ticker, start=start_date, end=end_date)
        bench_df = yf.download(benchmark_ticker, start=start_date, end=end_date)

        if df.empty or bench_df.empty:
            return f"error::No data for {ticker} or benchmark."

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        bench_df = bench_df[['Close']].rename(columns={'Close': 'Benchmark_Close'})

        # --- Price performance ---
        latest = to_scalar(df['Close'].iloc[-1])
        perf_1m = to_scalar((df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100) if len(df) > 21 else np.nan
        perf_3m = to_scalar((df['Close'].iloc[-1] / df['Close'].iloc[-63] - 1) * 100) if len(df) > 63 else np.nan
        perf_6m = to_scalar((df['Close'].iloc[-1] / df['Close'].iloc[-126] - 1) * 100) if len(df) > 126 else np.nan
        perf_12m = to_scalar((df['Close'].iloc[-1] / df['Close'].iloc[-252] - 1) * 100) if len(df) > 252 else np.nan

        # --- Moving Averages & Distances ---
        for w in [50, 100, 200]:
            dma = df['Close'].rolling(window=w).mean()
            df[f'{w}DMA'] = dma
            df[f'Dist_{w}DMA_abs'] = df['Close'] - dma
            df[f'Dist_{w}DMA_pct'] = (df['Close'] - dma) / dma * 100

        # --- Relative performance vs benchmark ---
        merged = df[['Close']].merge(bench_df, left_index=True, right_index=True, how='left')
        rel_perf = {}
        for days, name in zip([63, 126, 252], ['3M', '6M', '12M']):
            if len(merged) > days:
                stock_ret = (merged['Close'].iloc[-1] / merged['Close'].iloc[-days] - 1) * 100
                bench_ret = (merged['Benchmark_Close'].iloc[-1] / merged['Benchmark_Close'].iloc[-days] - 1) * 100
                rel_perf[name] = to_scalar(stock_ret - bench_ret)
            else:
                rel_perf[name] = np.nan

        # --- Volatility & ATR ---
        df['Month'] = df.index.to_period('M')
        monthly_range = df.groupby('Month').apply(lambda x: x['High'].max() - x['Low'].min())
        last_month_range = to_scalar(monthly_range.iloc[-1]) if not monthly_range.empty else np.nan
        atr_14d_last = to_scalar((df['High'] - df['Low']).rolling(14).mean().iloc[-1])
        sd_21d = to_scalar(df['Close'].pct_change().rolling(21).std().iloc[-1] * 100)

        # --- Trend Smoothness (R²) ---
        if len(df) >= 50:
            X = np.arange(-49, 1).reshape(-1, 1)
            y = df['Close'].iloc[-50:].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            r2_50d = to_scalar(reg.score(X, y))
        else:
            r2_50d = np.nan

        # --- Highs and Lows ---
        high_52w = to_scalar(df['High'].iloc[-252:].max()) if len(df) > 252 else to_scalar(df['High'].max())
        low_52w = to_scalar(df['Low'].iloc[-252:].min()) if len(df) > 252 else to_scalar(df['Low'].min())
        high_2y = to_scalar(df['High'].iloc[-504:].max()) if len(df) > 504 else to_scalar(df['High'].max())
        low_2y = to_scalar(df['Low'].iloc[-504:].min()) if len(df) > 504 else to_scalar(df['Low'].min())
        ath = to_scalar(df['High'].max())
        atl = to_scalar(df['Low'].min())

        # --- Volume Analysis ---
        last_vol = to_scalar(df['Volume'].iloc[-1])
        avg_vol_20d = to_scalar(df['Volume'].rolling(20).mean().iloc[-1])
        avg_vol_50d = to_scalar(df['Volume'].rolling(50).mean().iloc[-1])

        # --- Output Dictionary ---
        latest_stats = {
            'Latest_Price': latest,
            '1M_return_pct': perf_1m,
            '3M_return_pct': perf_3m,
            '6M_return_pct': perf_6m,
            '12M_return_pct': perf_12m,
            '50DMA': to_scalar(df['50DMA'].iloc[-1]),
            'Dist_50DMA_abs': to_scalar(df['Dist_50DMA_abs'].iloc[-1]),
            'Dist_50DMA_pct': to_scalar(df['Dist_50DMA_pct'].iloc[-1]),
            '100DMA': to_scalar(df['100DMA'].iloc[-1]),
            'Dist_100DMA_abs': to_scalar(df['Dist_100DMA_abs'].iloc[-1]),
            'Dist_100DMA_pct': to_scalar(df['Dist_100DMA_pct'].iloc[-1]),
            '200DMA': to_scalar(df['200DMA'].iloc[-1]),
            'Dist_200DMA_abs': to_scalar(df['Dist_200DMA_abs'].iloc[-1]),
            'Dist_200DMA_pct': to_scalar(df['Dist_200DMA_pct'].iloc[-1]),
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

# --- Streamlit App UI ---
st.set_page_config(page_title="📊 Technical Dashboard", layout="wide")
st.title("📊 Technical Analysis Dashboard")

col1, col2 = st.columns(2)
with col1:
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS)", value="RELIANCE.NS")
with col2:
    benchmark_ticker = st.text_input("Enter Benchmark Ticker (e.g., NIFTYBEES.NS)", value="NIFTYBEES.NS")

if st.button("📥 Fetch Data"):
    with st.spinner("Fetching and analyzing data..."):
        result_df = fetch_technical_data(stock_ticker, benchmark_ticker)

    if isinstance(result_df, pd.DataFrame):
        st.success("✅ Data fetched successfully!")
        st.dataframe(result_df.style.format("{:,.2f}"), use_container_width=True)
    elif isinstance(result_df, str) and result_df.startswith("error::"):
        st.error(f"❌ {result_df[7:]}")
    else:
        st.error("❌ Unknown error occurred.")
