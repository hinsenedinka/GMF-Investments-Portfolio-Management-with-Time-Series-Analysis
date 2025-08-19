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

start_date = "2015-07-01"
end_date = "2025-07-31"
assets = ['TSLA', 'BND', 'SPY']
data = yf.download(assets, start=start_date, end=end_date)
# Calculating expected returns and covariance matrix
mu = expected_returns.mean_historical_return(data['Close'])
S = risk_models.sample_cov(data['Close'])

# Generate random portfolios to plot
num_portfolios = 10000
assets = data['Close'].columns
n_assets = len(assets)
np.random.seed(42)
weights = np.random.dirichlet(np.ones(n_assets), num_portfolios)
returns = np.dot(weights, mu)
stddevs = np.sqrt(np.diag(weights @ S @ weights.T))
sharpe_ratios = returns / stddevs

random_portfolios = [{'weights': w, 'return': r, 'stddev': s, 'sharpe': sr} for w, r, s, sr in zip(weights, returns, stddevs, sharpe_ratios)]

# Find exact optimal portfolios using PyPortfolioOpt
ef_minvol = EfficientFrontier(mu, S)
ef_minvol.min_volatility()
min_vol_weights = ef_minvol.clean_weights()
min_vol_return, min_vol_stddev, _ = ef_minvol.portfolio_performance()

ef_maxsharpe = EfficientFrontier(mu, S)
ef_maxsharpe.max_sharpe()
max_sharpe_weights = ef_maxsharpe.clean_weights()
max_sharpe_return, max_sharpe_stddev, _ = ef_maxsharpe.portfolio_performance()

# Compile results
ef_results = {
    'random_portfolios': random_portfolios,
    'min_vol': {'weights': np.array([min_vol_weights[asset] for asset in assets]), 'return': min_vol_return, 'stddev': min_vol_stddev},
    'max_sharpe': {'weights': np.array([max_sharpe_weights[asset] for asset in assets]), 'return': max_sharpe_return, 'stddev': max_sharpe_stddev}
}

# Plot Efficient Frontier
plt.figure(figsize=(15, 10))
plt.scatter([p['stddev'] for p in random_portfolios], [p['return'] for p in random_portfolios], c=sharpe_ratios, marker='o', cmap='viridis', s=10, alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(ef_results['min_vol']['stddev'], ef_results['min_vol']['return'], marker='*', color='r', s=500, label='Minimum Volatility')
plt.scatter(ef_results['max_sharpe']['stddev'], ef_results['max_sharpe']['return'], marker='*', color='g', s=500, label='Maximum Sharpe Ratio')
for i, asset in enumerate(assets):
    asset_vol = np.sqrt(S.iloc[i, i])
    asset_ret = mu[i]
    plt.scatter(asset_vol, asset_ret, marker='o', s=200, color='black')
    plt.annotate(asset, (asset_vol * 1.01, asset_ret * 1.01))
plt.title('Efficient Frontier')
plt.xlabel('Expected Volatility (Standard Deviation)')
plt.ylabel('Expected Annual Return')
plt.legend()
plt.tight_layout()
plt.show()

# Print Portfolio Weights & Metrics
print("\nMinimum Volatility Portfolio:")
print(f"Expected Return: {ef_results['min_vol']['return']:.2%}")
print(f"Expected Volatility: {ef_results['min_vol']['stddev']:.2%}")
print(f"Sharpe Ratio: {ef_results['min_vol']['return'] / ef_results['min_vol']['stddev']:.2f}")
print("Asset Allocation:")
for i, asset in enumerate(assets):
    print(f"{asset}: {ef_results['min_vol']['weights'][i]:.2%}")

print("\nMaximum Sharpe Ratio Portfolio:")
print(f"Expected Return: {ef_results['max_sharpe']['return']:.2%}")
print(f"Expected Volatility: {ef_results['max_sharpe']['stddev']:.2%}")
print(f"Sharpe Ratio: {ef_results['max_sharpe']['return'] / ef_results['max_sharpe']['stddev']:.2f}")
print("Asset Allocation:")
for i, asset in enumerate(assets):
    print(f"{asset}: {ef_results['max_sharpe']['weights'][i]:.2%}")
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.pie(ef_results['min_vol']['weights'], labels=assets, autopct='%1.1f%%')
plt.title('Minimum Volatility Portfolio Allocation')
plt.subplot(1, 2, 2)
plt.pie(ef_results['max_sharpe']['weights'], labels=assets, autopct='%1.1f%%')
plt.title('Maximum Sharpe Ratio Portfolio Allocation')
plt.tight_layout()
plt.show()

# Additional plot using pypfopt's built-in plotting function
ef_opt = EfficientFrontier(mu, S)
weights = ef_opt.max_sharpe()
cleaned_weights = ef_opt.clean_weights()

print("Optimized Portfolio Weights:")
for k, v in cleaned_weights.items():
    print(f"{k}: {100 * v:.2f}%")

fig, ax = plt.subplots(figsize=(15, 10))
plot_efficient_frontier(EfficientFrontier(mu, S), ax=ax, show_assets=True)
plt.title('Efficient Frontier')
plt.tight_layout()
plt.show()