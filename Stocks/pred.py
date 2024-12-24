import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import matplotlib.dates as mdates  # Import for date formatting

# Data Loading and Preprocessing
start = '2012-01-01'
end = '2024-12-20'
stock = 'GOOG'

# Download stock data
data = yf.download(stock, start, end)
data.reset_index(inplace=True)

# Plot Moving Averages and Closing Prices
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

plt.figure(figsize=(8, 6))
plt.plot(data['Date'], ma_100_days, 'r', label='100-Day Moving Average')
plt.plot(data['Date'], ma_200_days, 'b', label='200-Day Moving Average')
plt.plot(data['Date'], data.Close, 'g', label='Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Prices and Moving Averages')
plt.legend()
plt.grid(True)

# Format x-axis for dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)
plt.show()

# Data Preparation for Training
data.dropna(inplace=True)
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scaled = scaler.fit_transform(data_train)

# Create sequences for LSTM
x_train = []
y_train = []
for i in range(100, data_train_scaled.shape[0]):
    x_train.append(data_train_scaled[i - 100:i])
    y_train.append(data_train_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Model Creation
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1)

# Save the Model
model.save('Stock Predictions Model.keras')

# Prepare Test Data
past_100_days = data_train.tail(100)
data_test_combined = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.transform(data_test_combined)

x_test = []
y_test = []
for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i - 100:i])
    y_test.append(data_test_scaled[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test)
scale = 1 / scaler.scale_
y_pred = y_pred * scale
y_test = y_test * scale

plt.figure(figsize=(10, 8))
plt.plot(y_test, 'g', label='Original Price')
plt.plot(y_pred, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Original vs Predicted Prices')
plt.legend()
plt.grid(True)
plt.show()

last_100_days = data_test.tail(100).values
last_100_days_scaled = scaler.transform(last_100_days)

n = 30 
future_prices = []

current_input = last_100_days_scaled
for _ in range(n):
    current_input = current_input[-100:]  
    current_input_reshaped = np.reshape(current_input, (1, 100, 1))
    predicted_price_scaled = model.predict(current_input_reshaped)
    predicted_price = predicted_price_scaled * scale
    future_prices.append(predicted_price[0][0])
    current_input = np.append(current_input, predicted_price_scaled, axis=0)

future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=n + 1, freq='B')[1:]  

plt.figure(figsize=(10, 8))
plt.plot(data['Date'], data.Close, 'g', label='Historical Price')
plt.plot(future_dates, future_prices, 'r', label='Future Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Historical Prices and Future Predictions')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y')) 
plt.gca().xaxis.set_major_locator(mdates.YearLocator())          
plt.xticks(rotation=45) 
plt.show()

