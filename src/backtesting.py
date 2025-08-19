from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier
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

# Define and download data
start_date = "2015-07-01"
end_date = "2025-07-31"
assets = ['TSLA', 'BND', 'SPY']
data = yf.download(assets, start=start_date, end=end_date)
benchmark_tickers = ['SPY', 'BND']

# Define backtesting period and rebalancing frequency
backtest_start = '2024-08-01'
backtest_end = '2025-07-31'
rebalance_freq = 'ME'

# Download and process benchmark data
benchmark_data = yf.download(benchmark_tickers, start=backtest_start, end=backtest_end)['Close']
benchmark_returns = benchmark_data.pct_change().dropna()

# Prepare data for the backtest loop
backtest_data = data.loc[backtest_start:backtest_end]
rebalancing_dates = pd.to_datetime(backtest_data.resample(rebalance_freq).last().index)
portfolio_cumulative_returns = pd.Series([1.0], index=[backtest_start])
all_period_returns_list = []

# Backtesting loop
for i in range(len(rebalancing_dates) - 1):
    optimization_period_end = rebalancing_dates[i]
    next_rebalance_date = rebalancing_dates[i + 1]
    
    optimization_data = data.loc[:optimization_period_end].iloc[-252:]
    
    if len(optimization_data) < 252:
        continue

    # Calculate mu and S for optimization
    mu = expected_returns.mean_historical_return(optimization_data)
    S = CovarianceShrinkage(optimization_data).ledoit_wolf()
    
    # Optimize for max Sharpe or min volatility
    try:
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe(risk_free_rate=0.01)
    except ValueError:
        ef = EfficientFrontier(mu, S)
        weights = ef.min_volatility()
    
    cleaned_weights = ef.clean_weights()

    # Calculate returns for the current period
    period_returns = data.loc[optimization_period_end:next_rebalance_date].pct_change().dropna()
    period_portfolio_returns = (period_returns @ pd.Series(cleaned_weights)).dropna()
    
    # Update cumulative returns
    all_period_returns_list.append(period_portfolio_returns)
    last_cumulative_return = portfolio_cumulative_returns.iloc[-1]
    new_cumulative_returns = last_cumulative_return * (1 + period_portfolio_returns).cumprod()
    portfolio_cumulative_returns = pd.concat([portfolio_cumulative_returns, new_cumulative_returns.iloc[1:]])

# Drop the initial 1.0 placeholder
portfolio_cumulative_returns = portfolio_cumulative_returns.iloc[1:]
portfolio_returns = pd.concat(all_period_returns_list)

# Create the benchmark portfolio
benchmark_weights = {'SPY': 0.60, 'BND': 0.40}
benchmark_portfolio = (benchmark_returns.loc[backtest_start:backtest_end] @ pd.Series(benchmark_weights)).dropna()
benchmark_cumulative_returns = (1 + benchmark_portfolio).cumprod()
benchmark_cumulative_returns = benchmark_cumulative_returns.reindex(portfolio_cumulative_returns.index, method='pad')

# Plot the cumulative returns
plt.figure(figsize=(15, 8))
plt.plot(portfolio_cumulative_returns, label='Sharpe Ratio Optimized Portfolio', color='blue')
plt.plot(benchmark_cumulative_returns, label='60/40 SPY/BND Benchmark', color='red')
plt.title('Backtesting Performance: Strategy vs. Benchmark')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()

# Calculate Final Performance Metrics
final_portfolio_return = portfolio_cumulative_returns.iloc[-1] - 1
final_benchmark_return = benchmark_cumulative_returns.iloc[-1] - 1
risk_free_rate = 0.01
trading_days = 252

sharpe_portfolio = (portfolio_returns.mean() * trading_days - risk_free_rate) / (portfolio_returns.std() * np.sqrt(trading_days))
sharpe_benchmark = (benchmark_portfolio.mean() * trading_days - risk_free_rate) / (benchmark_portfolio.std() * np.sqrt(trading_days))

print("--- Backtest Results ---")
print(f"Final Strategy Return: {final_portfolio_return:.2%}")
print(f"Final Benchmark Return: {final_benchmark_return:.2%}")
print("-" * 25)
print(f"Strategy Sharpe Ratio: {sharpe_portfolio:.2f}")
print(f"Benchmark Sharpe Ratio: {sharpe_benchmark:.2f}")