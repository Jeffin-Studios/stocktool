
#rolling average for both stock and market


from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def rollingAverage(name, start, end, strict = False, graph = False):
	stock = data.DataReader(name, 'yahoo', start, end)
	close_price = stock['Close']

	if (strict):
		# Getting all weekdays 
		all_weekdays = pd.date_range(start=start, end=end, freq='B')
		close_price = close_price.reindex(all_weekdays)
		close_price = close_price.fillna(method='ffill')

	# weekly_roll
	short_rolling_stock = close_price.rolling(window=5).mean()
	medium_rolling_stock = close_price.rolling(window=20).mean()
	long_rolling_stock = close_price.rolling(window=60).mean()

	if (graph):
		fig, ax = plt.subplots(figsize=(16,9))

		ax.plot(close_price.index, close_price, label=name)
		ax.plot(short_rolling_stock.index, short_rolling_stock, label='5 days rolling')
		ax.plot(long_rolling_stock.index, medium_rolling_stock, label='20 days rolling')
		ax.plot(long_rolling_stock.index, long_rolling_stock, label='60 days rolling')

		ax.set_xlabel('Date')
		ax.set_ylabel('Adjusted closing price ($)')
		ax.legend()
		plt.grid()
		plt.show()

	return (close_price, short_rolling_stock, medium_rolling_stock)

######## Need to make this real time, to detect when climb starts and when dip starts
def fluctuation(name, start, end, graph = False):
	(close_price, short_rolling_stock, medium_rolling_stock) = rollingAverage(name, start, end)
	change = short_rolling_stock - medium_rolling_stock
	# Starts climbing when short term average passes long term average
	# Starts dipping when short term average goes below long term average
	buy_dates = []
	sell_dates = []

	if (graph):
		fig, ax = plt.subplots(figsize=(16,9))
		ax.plot(change.index, change, label="change")
		ax.set_xlabel('Date')
		ax.set_ylabel('Difference')
		ax.legend()
		plt.axhline(0, color='black')
		plt.grid()
		plt.show()

	return (buy_dates, sell_dates)

def get_rsi(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi


if __name__ == "__main__":
	start="2015-01-01"
	end="2018-9-20"
	stocks = ['CRON', 'AMD']

	(close_price, short_rolling_stock, medium_rolling_stock) = rollingAverage(stocks[1], start, end)
	rsi = get_rsi(close_price)
	print(rsi[-1])



