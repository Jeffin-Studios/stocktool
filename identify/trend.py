
#rolling average for both stock and market


from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class stocktrend():
	
	# Initialization requires a ticker symbol
	def __init__(self, stock_name, start_date = None, end_date = None, draw_graph = False):
		 self.name = stock_name.upper()
		 self.start = start_date
		 self.end = end_date
		 self.graph = draw_graph
		 self.stock = data.DataReader(stock_name, 'yahoo', start, end)

	# Basic Historical Plots and Basic Statistics
	def plot_stock(self, stats=[], series = [], serieslabels = [], xlabel="Date", ylabel="Price"):
		fig, ax = plt.subplots(figsize=(16,9))
		fig.suptitle(self.name, fontsize=20)
		for stat in stats:
			ax.plot(self.stock[stat].index, self.stock[stat], label=stat)
		for i, data in enumerate(series):
			ax.plot(data.index, data, label=serieslabels[i])
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.legend()
		plt.axhline(0, color='black')
		plt.grid()
		plt.show()


	def rollingAverage(self, strict = False):
		close_price = self.stock['Close']

		if (strict):
			# Getting all weekdays 
			all_weekdays = pd.date_range(start=self.start, end=self.end, freq='B')
			close_price = close_price.reindex(all_weekdays)
			close_price = close_price.fillna(method='ffill')

		# weekly_roll
		short_rolling_stock = close_price.rolling(window=5).mean()
		medium_rolling_stock = close_price.rolling(window=20).mean()
		long_rolling_stock = close_price.rolling(window=60).mean()

		if (self.graph):
			self.plot_stock(series=[close_price, short_rolling_stock, medium_rolling_stock, long_rolling_stock], serieslabels=["Closing Price", "5 days rolling", "20 days rolling", "60 days rolling"])

		return (short_rolling_stock, medium_rolling_stock)

	# Buy when this is at a zero or high positive slope
	def daily_change(self):
		if ('Adj. Close' not in self.stock.columns):
			self.stock['Adj. Close'] = self.stock['Close']
			self.stock['Adj. Open'] = self.stock['Open']
			
		self.stock['y'] = self.stock['Adj. Close']
		self.stock['Daily Change'] = self.stock['Adj. Close'] - self.stock['Adj. Open']

		if (self.graph):
			self.plot_stock(stats=['Daily Change'], ylabel="Change in Price")



	def get_rsi(self, n=14):
		prices = self.stock['Close']
		dates = prices.index
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

		if (self.graph):
			fig, ax = plt.subplots(figsize=(16,9))
			fig.suptitle(self.name, fontsize=20)
			ax.plot(dates, rsi, color = "purple", linewidth=1.5, label='RSI')
			ax.axhline(70, color="red")
			ax.axhline(30, color="green")
			ax.fill_between(dates, rsi, 70, where=(rsi>=70), facecolor="red", edgecolor="red", alpha=0.5)
			ax.fill_between(dates, rsi, 30, where=(rsi<=30), facecolor="green", edgecolor="green", alpha=0.5)
			ax.set_yticks([30,70])
			ax.legend()
			ax.tick_params(axis='y')
			ax.tick_params(axis='x')
			ax.set_xlabel("Date")
			ax.set_ylabel("Momentum")
			ax.grid()
			plt.show()
			
		return rsi

	######## Need to make this real time, to detect when climb starts and when dip starts
	def fluctuation(self):
		(short_rolling_stock, medium_rolling_stock) = self.rollingAverage()
		self.stock["Fluctuation"] = short_rolling_stock - medium_rolling_stock
		# Starts climbing when short term average passes long term average
		# Starts dipping when short term average goes below long term average
		### Code determines if change is at a zero, evaluates slope of the change, to see if its climbing or dipping, also concavity to make sure. 
		if (self.graph):
			self.plot_stock(stats=['Fluctuation'], ylabel="Deviation From Average")
			

		# return status



if __name__ == "__main__":
	start="2015-01-01"
	end="2018-9-20"
	tickers = ['CRON', 'AMD', 'CLDR']

	stock = stocktrend(tickers[1], start, end, draw_graph = True)
	stock.fluctuation()
	stock.daily_change()
	stock.get_rsi()




