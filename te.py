import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Define the fetch_technical_data function (your full function pasted here)
def fetch_technical_data(ticker, benchmark_ticker, years=3):
    import numpy as np
    import yfinance as yf
    from sklearn.linear_model import LinearRegression

    try:
        # Fetch historical data
        start_date = (datetime.today() - timedelta(days=365 * years)).strftime("%Y-%m-%d")
        stock = yf.Ticker(ticker)
        bench = yf.Ticker(benchmark_ticker)

        df = stock.history(start=start_date)
        bench_df = bench.history(start=start_date)

        if df.empty or bench_df.empty:
            return None

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
        atr_14d = df['High'].rolling(14).max() - df['Low'].rolling(14).min()
        atr_14d_last = atr_14d.iloc[-1] if len(atr_14d) > 0 else np.nan
        sd_21d = df['Close'].pct_change().rolling(21).std().iloc[-1] * 100

        X = np.arange(-49, 1).reshape(-1, 1)
        if len(df) >= 50:
            y = df['Close'].iloc[-50:].values.reshape(-1, 1)
            reg = LinearRegression().fit(X, y)
            r2_50d = reg.score(X, y)
        else:
            r2_50d = np.nan

        high_52w = df['High'].iloc[-252:].max() if len(df) > 252 else df['High'].max()
        low_52w = df['Low'].iloc[-252:].min() if len(df) > 252 else df['Low'].min()
        high_2y = df['High'].iloc[-504:].max() if len(df) > 504 else df['High'].max()
        low_2y = df['Low'].iloc[-504:].min() if len(df) > 504 else df['Low'].min()
        ath = df['High'].max()
        atl = df['Low'].min()

        avg_vol_20d = df['Volume'].rolling(20).mean().iloc[-1] if len(df) > 20 else np.nan
        avg_vol_50d = df['Volume'].rolling(50).mean().iloc[-1] if len(df) > 50 else np.nan
        last_vol = df['Volume'].iloc[-1]

        latest_stats = {
            'Latest_Price': latest,
            '1M_return_pct': perf_1m,
            '3M_return_pct': perf_3m,
            '6M_return_pct': perf_6m,
            '12M_return_pct': perf_12m,
            '50DMA': df['50DMA'].iloc[-1],
            'Dist_50DMA_abs': df['Dist_50DMA_abs'].iloc[-1],
            'Dist_50DMA_pct': df['Dist_50DMA_pct'].iloc[-1],
            '100DMA': df['100DMA'].iloc[-1],
            'Dist_100DMA_abs': df['Dist_100DMA_abs'].iloc[-1],
            'Dist_100DMA_pct': df['Dist_100DMA_pct'].iloc[-1],
            '200DMA': df['200DMA'].iloc[-1],
            'Dist_200DMA_abs': df['Dist_200DMA_abs'].iloc[-1],
            'Dist_200DMA_pct': df['Dist_200DMA_pct'].iloc[-1],
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
        latest_stats_df = pd.DataFrame([latest_stats])
        return latest_stats_df

    except Exception as e:
        return None


# --- Streamlit App Starts Here ---
st.set_page_config(page_title="Stock Technical Dashboard", page_icon="📊", layout="wide")
st.title("📊 Technical Analysis Dashboard")

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, TCS.NS):", value="RELIANCE.NS")
with col2:
    benchmark_ticker = st.text_input("Enter Benchmark Ticker (e.g., NIFTYBEES.NS):", value="NIFTYBEES.NS")

# --- Fetch and Display Data ---
if st.button("Fetch Technical Data"):
    df = fetch_technical_data(stock_ticker, benchmark_ticker)
    if df is not None:
        st.success("✅ Data fetched successfully!")
        st.dataframe(df.style.format(precision=2))
    else:
        st.error("❌ Failed to fetch data. Please check the ticker symbols or internet connection.")
