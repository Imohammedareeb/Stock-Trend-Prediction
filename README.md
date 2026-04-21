# 📈 Stock Trend Prediction

This is a simple Streamlit application to visualize and analyze stock trend data for several popular stocks, including Tesla, PowerGrid, TCS, Netflix, and Zomato.

## 🚀 Features

- **Interactive Visualization:** Select a stock ticker from a dropdown to see its closing price trends over time.
- **Data Cleaning:** Automatically cleans and processes the stock data (dates, numerical conversion, etc.).
- **Descriptive Statistics:** Displays a statistical summary of the dataset.
- **Trend Prediction:** Visualizes actual vs. predicted prices (currently using a dummy prediction for demonstration).

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/stock-trend-prediction.git
cd stock-trend-prediction
```

### 2. Install dependencies
```bash
pip install -r python_requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

## 📊 Dataset
The application uses CSV files for each stock ticker. The data is pre-cleaned to remove extra header information and handle missing values.

## 📓 Notebook
The project includes a Jupyter notebook `stock_trend_prediction.ipynb` which contains more advanced data analysis and an LSTM-based prediction model.

## 📝 License
This project is for educational purposes.
