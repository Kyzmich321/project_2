import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading in the data
stock_data = pd.read_csv('csv/stock_data.csv', parse_dates=['Date'], index_col='Date')
benchmark_data = pd.read_csv('csv/benchmark_data.csv', parse_dates=['Date'], index_col='Date').dropna()
print('Stocks\n')
stock_data.info()
print(stock_data.head())
print('\nBenchmarks\n')
benchmark_data.info()
print(benchmark_data.head())
# visualize the stock_data
stock_data.plot(subplots=True, title='Stock Data')
plt.show()
# summarize the stock_data
stock_data.describe()
# plot the benchmark_data
benchmark_data.plot(title='S&P 500')
plt.show()
benchmark_data.describe()
# calculate daily stock_data returns
stock_returns = stock_data.pct_change()
stock_returns.plot()
plt.show()
stock_returns.describe()
sp_returns = benchmark_data['S&P 500'].pct_change()
sp_returns.plot()
plt.show()
sp_returns.describe()
excess_returns = stock_returns.sub(sp_returns, axis=0)
excess_returns.plot()
plt.show()
excess_returns.describe()
# calculate the mean of excess_returns
avg_excess_return = excess_returns.mean()
avg_excess_return.plot.bar(title='Mean of the Return Difference')
plt.show()
# calculate the standard deviations
sd_excess_return = excess_returns.std()
sd_excess_return.plot.bar(title='Standard Deviation of the Return Difference')
plt.show()
# calculate the daily sharpe ratio
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)
# annualize the sharpe ratio
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)
annual_sharpe_ratio.plot.bar(title='Annualized Sharpe Ratio: Stocks vs S&P 500')
plt.show()
