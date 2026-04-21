import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime

# Page configuration
st.set_page_config(page_title="Stock Trend Prediction", page_icon="📈", layout="wide")

st.title("📈 Stock Trend Prediction")

# --- Data Loading ---
ticker_files = {
    "TESLA": "Tesla.csv",
    "POWERGRID": "powergrid.csv",
    "TCS": "TCS.csv",
    "NETFLIX": "Netflix.csv",
    "ZOMATO": "zomato.csv"
}

@st.cache_data
def load_data(ticker):
    file_path = ticker_files[ticker]
    df = pd.read_csv(file_path)
    
    # Robust Cleaning
    if len(df) > 0 and "Ticker" in str(df.iloc[0, 0]):
        df = df.iloc[1:].copy()
    
    df.columns = [col.strip() for col in df.columns]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Ensure numeric columns
    for col in ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=['Date', 'Close'], inplace=True)
    df.sort_values('Date', inplace=True)
    return df

# --- Model Training ---
@st.cache_resource
def train_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    prediction_days = 60
    x_train = []
    y_train = []
    
    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)
    
    return model, scaler

# --- Sidebar ---
st.sidebar.header("User Input Features")
stock_ticker = st.sidebar.selectbox("Select Stock Ticker", options=list(ticker_files.keys()))

# Load data early to get date ranges
df = load_data(stock_ticker)
min_date = df['Date'].min().to_pydatetime()
max_date = df['Date'].max().to_pydatetime()

date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

# Filter data based on selection
filtered_df = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]

# --- Main Page Layout ---
tab1, tab2, tab3 = st.tabs(["📈 Price Trends", "🔮 Predictions", "📊 Descriptive Data"])

with tab1:
    st.subheader(f"{stock_ticker} - Closing Price vs Time")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(filtered_df['Date'], filtered_df['Close'], label='Closing Price', color='royalblue')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with tab2:
    st.subheader("Model-based Prediction (Last 100 Days)")
    with st.spinner("Training model and generating predictions..."):
        # We train on the full dataset for better context but predict on the filtered/recent part
        full_data = df['Close'].values
        model, scaler = train_model(full_data)
        
        # Prepare data for prediction (last 160 days to get 100 predictions with 60 window)
        prediction_days = 60
        model_inputs = df['Close'].values[len(df) - 100 - prediction_days:]
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)
        
        x_test = []
        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x-prediction_days:x, 0])
            
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        actual_prices = df['Close'].values[-100:]
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(actual_prices, color='black', label="Actual Price")
        ax2.plot(predicted_prices, color='green', linestyle='--', label="Predicted Price")
        ax2.set_title(f"{stock_ticker} Price Prediction")
        ax2.set_xlabel("Time (Days)")
        ax2.set_ylabel("Price")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
        st.info("Note: This is a demonstration LSTM model trained on-the-fly. Real-world accuracy depends on significantly more training data and hyperparameter tuning.")

with tab3:
    st.subheader(f"Data Summary: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
    st.write(filtered_df.describe())
    st.markdown("### Raw Data Preview")
    st.dataframe(filtered_df.tail(10))
