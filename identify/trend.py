
#rolling average for both stock and market


from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd

# We would like all available data from 01/01/2000 until 12/31/2016.
start="2017-01-01"
end="2018-9-19"

tickers = ['CRON', 'CELG']

# User pandas_reader.data.DataReader to load the desired data. As simple as that.
stock = data.DataReader(tickers, 'yahoo', start, end)
# view it
# stock.to_frame().head(9)

# Getting just the adjusted closing prices. This will return a Pandas DataFrame
# The index in this DataFrame is the major index of the stock.
close = stock['Close']

# Getting all weekdays between 01/01/2000 and 12/31/2016
all_weekdays = pd.date_range(start=start, end=end, freq='B')

# How do we align the existing prices in adj_close with our new set of dates?
# All we need to do is reindex close using all_weekdays as the new index
close = close.reindex(all_weekdays)

# Reindexing will insert missing values (NaN) for the dates that were not present
# in the original set. To cope with this, we can fill the missing by replacing them
# with the latest available price for each instrument.
close = close.fillna(method='ffill')

print(all_weekdays)

close.head(10)

close.describe()

# Get the MSFT timeseries. This now returns a Pandas Series object indexed by date.
msft = close.loc[:, 'CELG']

# Calculate the 20 and 100 days moving averages of the closing prices
short_rolling_msft = msft.rolling(window=20).mean()
medium_rolling_msft = msft.rolling(window=60).mean()
long_rolling_msft = msft.rolling(window=100).mean()

# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(msft.index, msft, label='CELG')
ax.plot(short_rolling_msft.index, short_rolling_msft, label='20 days rolling')
ax.plot(long_rolling_msft.index, medium_rolling_msft, label='60 days rolling')
ax.plot(long_rolling_msft.index, long_rolling_msft, label='100 days rolling')

ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()

plt.show()

