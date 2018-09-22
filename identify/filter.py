import csv
from datetime import *
import math
from trend import stocktrend


def get_stocks(input_path, sector = None):
	stock_list = []
	with open(input_path) as csvfile:
		stock_reader = csv.reader(csvfile) 
		for row in stock_reader:
			if (sector):
				if (row[6] == sector):
					stock_list.append(row[0])
			else:
				stock_list.append(row[0])
	return stock_list

def screen(stock_list, start, end):
	filtered = []
	for stock_name in stock_list:
		stock = stocktrend(stock_name, start, end)
		if (stock.get_rsi()[-1] < 70):
			average = stock.rollingAverage()[0]
			long_average = stock.rollingAverage()[1]
			if (average.std(ddof=0) > 0.05*average[-1]):
				if (average[-1] > average[-2] and average[-2] > average[-3]):
					if (abs(average[-1] - long_average[-1]) < 0.01*average[-1]):
						filtered.append(stock_name)
	return filtered

if __name__ == "__main__":
	input_path = "Data/companylist.csv"
	stocks = get_stocks(input_path, "Technology")

	start = datetime(2018, 8, 1)
	end = datetime.now()
	filtered = screen(stocks, start, end)

	print(filtered)