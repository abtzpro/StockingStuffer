import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Fetch historical data
ticker = 'AMCR'
data = yf.download(ticker, start='2010-01-01', end='2023-07-16')

# Diversification: Consider other stocks too
other_tickers = ['AAPL', 'MSFT', 'GOOG', 'FB', 'TSLA']  # Add or remove tickers to fit your investment strategy

# Fetch data for other tickers
other_data = {t: yf.download(t, start='2010-01-01', end='2023-07-16') for t in other_tickers}

# Normalize the closing prices
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Prepare training data
x_train, y_train = [], []
for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Define the LSTM network
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(1))

# Compile and train the model
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
model_lstm.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# Fetch the latest data to make a prediction
latest_data = yf.download(ticker, start='2023-07-01', end='2023-07-16')
latest_scaled = scaler.transform(latest_data['Close'].values.reshape(-1,1))
x_test = []
x_test.append(latest_scaled[len(latest_scaled)-61:len(latest_scaled)-1, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict the next day's closing price
predicted_price_lstm = model_lstm.predict(x_test)
predicted_price_lstm = scaler.inverse_transform(predicted_price_lstm)

print(f"Predicted closing price for {ticker} on 2023-07-17 using LSTM is {predicted_price_lstm}")

# Linear Regression Model

# Prepare data for linear regression
lr_data = data[['Close']]
lr_data['Prediction'] = lr_data[['Close']].shift(-1)

X_lr = np.array(lr_data.drop(['Prediction'],1))[:-1]
y_lr = np.array(lr_data['Prediction'])[:-1]

# Train the model
model_lr = LinearRegression()
model_lr.fit(X_lr, y_lr)

# Predict the next day's closing price
X_forecast = np.array(lr_data.drop(['Prediction'],1))[-1:]
predicted_price_lr = model_lr.predict(X_forecast)

print(f"Predicted closing price for {ticker} on 2023-07-17 using Linear Regression is {predicted_price_lr}")

# Diversification: Predict for other tickers as well
predictions_other_tickers = {}
for t, d in other_data.items():
    scaler_other = MinMaxScaler(feature_range=(0, 1))
    scaled_data_other = scaler_other.fit_transform(d['Close'].values.reshape(-1,1))
    x_test_other = []
    x_test_other.append(scaled_data_other[len(scaled_data_other)-61:len(scaled_data_other)-1, 0])
    x_test_other = np.array(x_test_other)
    x_test_other = np.reshape(x_test_other, (x_test_other.shape[0], x_test_other.shape[1], 1))
    predicted_price_other = model_lstm.predict(x_test_other)
    predicted_price_other = scaler_other.inverse_transform(predicted_price_other)
    predictions_other_tickers[t] = predicted_price_other

print("Predicted closing prices for other tickers using LSTM:")
for t, p in predictions_other_tickers.items():
    print(f"{t}: {p}")

# Risk Tolerance
risk_tolerance = 0.05  # A dummy value; replace with your risk tolerance

# A simplified version of regular review: If the predicted price is significantly higher than the current price,
# consider buying; otherwise, consider selling
buy_or_sell = {}
for t, p in predictions_other_tickers.items():
    current_price = other_data[t]['Close'].iloc[-1]
    if p > (1 + risk_tolerance) * current_price:
        buy_or_sell[t] = 'Buy'
    else:
        buy_or_sell[t] = 'Sell'

print("Buy or sell based on predictions and risk tolerance:")
for t, b in buy_or_sell.items():
    print(f"{t}: {b}")
