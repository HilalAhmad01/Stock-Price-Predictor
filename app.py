
import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dropout, Dense # type: ignore

st.title("📈 Stock Price Predictor using LSTM")

# Sidebar inputs
stock = st.sidebar.text_input("Enter Stock Ticker", "GOOG")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2012-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

if st.sidebar.button("Predict"):
    df = yf.download(stock, start=start_date, end=end_date)

    st.subheader(f"{stock} Closing Price")
    st.line_chart(df['Close'])

    # Moving Averages
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Close'], label='Close Price')
    ax.plot(ma100, label='100-day MA')
    ax.plot(ma200, label='200-day MA')
    ax.set_title("Moving Averages")
    ax.legend()
    st.pyplot(fig)

    # Data prep
    data = df.filter(['Close'])
    dataset = data.values
    dataset = data.values.reshape(-1, 1)
    training_data_len = int(np.ceil(len(dataset) * 0.8))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:training_data_len]
    x_train, y_train = [], []

    for i in range(100, len(train_data)):
        x_train.append(train_data[i-100:i])
        y_train.append(train_data[i])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(60, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(80))
    model.add(Dropout(0.4))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # Testing
    test_data = scaled_data[training_data_len - 100:]
    x_test = []
    y_test = dataset[training_data_len:]

    for i in range(100, len(test_data)):
        x_test.append(test_data[i-100:i])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Plotting prediction
    st.subheader("Predicted vs Actual")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(y_test, label='Actual Price')
    ax2.plot(predictions, label='Predicted Price')
    ax2.legend()
    st.pyplot(fig2)
