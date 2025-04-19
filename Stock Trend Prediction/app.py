import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.title("📈 Stock Trend Prediction")

st.write("""
Enter a stock ticker to visualize its trends and predictions.
""")

# Define available stock tickers and their associated CSV files
ticker_files = {
    "TESLA": "Tesla.csv",
    "POWERGRID": "powergrid.csv",
    "TCS": "TCS.csv",
    "NETFLIX": "Netflix.csv",
    "ZOMATO": "zomato.csv"
}

# Replace text input with a dropdown using selectbox
stock_ticker = st.selectbox(
    "Select Stock ticker:",
    options=list(ticker_files.keys())
)

if st.button("Submit"):
    # The selected stock_ticker is already uppercase
    # Load the dataset for the specified ticker
    df = pd.read_csv(ticker_files[stock_ticker])
    
    # Data Cleaning: Remove the first row and convert necessary columns
    df = df.iloc[1:]
    df.columns = [col.strip() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if 'Close' in df.columns:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    else:
        st.error("The dataset does not have a 'Close' column.")
        st.stop()
    df.dropna(inplace=True)
    
    # --- 1. Closing Price vs Time ---
    st.subheader("1. Closing Price vs Time")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df['Date'], df['Close'], label='Closing Price')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Closing Price")
    ax1.set_title(f"{stock_ticker} Closing Price Over Time")
    ax1.legend()
    st.pyplot(fig1)
    
    # --- 2. Prediction vs Original Trend ---
    st.subheader("2. Prediction vs Original Trend")
    # Generate a dummy prediction series by adding random noise (replace with model predictions as needed)
    actual = df['Close'].values[-100:]
    predicted = actual + np.random.normal(0, 10, size=len(actual))
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(actual, label="Actual Closing Price")
    ax2.plot(predicted, label="Predicted Price", linestyle='--')
    ax2.set_title("Prediction vs Original Trend (Sample)")
    ax2.legend()
    st.pyplot(fig2)
    
    # --- 3. Descriptive Data ---
    st.subheader("3. Descriptive Data from Jan 2000 to Nov 2024")
    # Optionally filter the data by date if necessary:
    # start_date = pd.to_datetime('2000-01-01')
    # end_date = pd.to_datetime('2024-11-30')
    # df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    st.write(df.describe())
