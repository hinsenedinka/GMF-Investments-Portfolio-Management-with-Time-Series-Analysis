import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Time period and assets
start_date = "2015-07-01"
end_date = "2025-07-31"
assets = ['TSLA', 'BND', 'SPY']

# Extracting Data
data = yf.download(assets, start=start_date, end=end_date)

# Generic Statistical Description
print("data.shape:", data.shape)
print("\nData Columns:")
print(data.columns.levels[0].tolist())
print("\nFirst 5 rows of the dataset")
print(data.head())

# Extracting Adjusted Close prices and handling missing values
price = data['Close'].copy()
print("\nMissing Values:")
print(price.isna().sum())
if price.isna().sum().any():
    price = price.fillna(method='ffill')
    print("Missing values filled using forward fill")

# Trading Volume Trend Plot
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Volume'])
plt.title('Trading Volume Trend')
plt.xlabel('Date')
plt.ylabel('Trading Volume')
plt.legend(assets)
plt.tight_layout()
plt.show()

# Daily returns and price trend plots
returns = price.pct_change().dropna()
plt.figure(figsize=(12, 6))
for asset in assets:
    plt.plot(price.index, price[asset] / price[asset].iloc[0], label=asset)
plt.title('Normalized Price Trend')
plt.ylabel('Normalized Price')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for ticker in assets:
    plt.plot(returns.index, returns[ticker], label=ticker)
plt.title('Daily Percent Change')
plt.ylabel('Daily Returns')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.show()

# Seasonality and Trends using ADF test
print("\nSeasonality and Trends:")
for ticker in price.columns:
    stats_price = adfuller(price[ticker])
    stats_return = adfuller(returns[ticker])
    print(f"ADF Statistics of {ticker} in price: {round(stats_price[0], 2)}")
    print(f"ADF Statistics of {ticker} in daily return: {round(stats_return[0], 2)}")
    print(f"{ticker} p-value in price: {round(stats_price[1], 2)}")
    print(f"{ticker} p-value in daily return: {round(stats_return[1], 2)}")

# Key statistics (Volatility and Sharpe Ratio)
print("\nVolatility:")
summary_stats = pd.DataFrame(index=returns.columns)
summary_stats['annualized return'] = returns.mean() * 252
summary_stats['annualized volatility'] = returns.std() * np.sqrt(252)
summary_stats['sharpe ratio'] = summary_stats['annualized return'] / summary_stats['annualized volatility']
print(summary_stats[['annualized return', 'annualized volatility', 'sharpe ratio']])

# Forecasting for Tesla (TSLA) stock returns
asset_return = returns['TSLA']
split_date = "2024-01-01"
train_data = asset_return.loc[asset_return.index < split_date]
test_data = asset_return.loc[asset_return.index >= split_date]

# Plot ACF and PACF to determine ARIMA parameters
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(asset_return, ax=ax1, lags=20)
ax1.set_title('Autocorrelation Function (ACF)')
plot_pacf(asset_return, ax=ax2, lags=20)
ax2.set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# Fit ARIMA model
p, d, q = 1, 0, 1
model = ARIMA(train_data, order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())

# Forecast and evaluate
forecast_steps = len(test_data)
prediction = model_fit.forecast(steps=forecast_steps)
mae = mean_absolute_error(test_data, prediction)
rmse = np.sqrt(mean_squared_error(test_data, prediction))
print(f"\nModel Evaluation on Test Data:")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

# Plot actual vs. forecasted returns
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data, label='Actual Return', color='blue')
plt.plot(test_data.index, prediction, label='Forecasted Return', color='red')
plt.title('TSLA - Actual vs. Forecasted Returns (ARIMA)')
plt.xlabel('Date')
plt.ylabel('Return')
plt.legend()
plt.tight_layout()
plt.show()


# Preparing data for LSTM
data_close = data['Close']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data_close['TSLA'].values.reshape(-1, 1))

# Creating sequences
def create_sequences(dat, window=60):
    X, y = [], []
    for i in range(len(dat) - window):
        X.append(dat[i:i + window])
        y.append(dat[i + window])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)
split_index = len(X) - 410
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build and train LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train, y_train, epochs=70, batch_size=32, validation_split=0.2, verbose=0)

# Plot training loss
plt.figure(figsize=(15, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training Progress')
plt.legend()
plt.show()

# Predict and plot LSTM results
lstm_forecast = model.predict(X_test)
lstm_forecast = scaler.inverse_transform(lstm_forecast).flatten()
plt.figure(figsize=(15, 6))
plt.plot(data_close['TSLA'].index[-410:], data_close['TSLA'][-410:], label='Actual', color='blue')
plt.plot(data_close['TSLA'].index[-410:], lstm_forecast, label='LSTM Forecast', linestyle='--', color='red')
plt.title('TSLA Stock Price Forecast (LSTM)')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# Assuming 'scaled' and 'model' are already defined from previous steps
# The script uses a rolling forecast to predict 6 months (180 days) into the future
last_60_days = scaled[-100:]
initial_input = np.reshape(last_60_days, (1, 100, 1))
forecast_steps = 180
forecast_list = []
current_input = initial_input.copy()

for _ in range(forecast_steps):
    next_step_scaled = model.predict(current_input, verbose=0)
    forecast_list.append(next_step_scaled[0, 0])
    current_input = np.append(current_input[:, 1:, :], next_step_scaled.reshape(1, 1, 1), axis=1)

lstm_forecast_scaled = np.array(forecast_list).reshape(-1, 1)
lstm_forecast = scaler.inverse_transform(lstm_forecast_scaled)

last_date = data_close.index[-1]
future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1)[1:]

plt.figure(figsize=(15, 6))
plt.plot(data_close.index, data_close['TSLA'], label='Historical Data', color='blue')
plt.plot(future_dates, lstm_forecast, label='LSTM Forecast', linestyle='--', color='red')
plt.axvline(x=data_close.index[-1], color='green', linestyle='-', label='Forecast Start')
plt.title('TSLA Stock Price Forecast for the next 6 months')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()