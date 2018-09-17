from stocktool import stocktool
import matplotlib.pyplot as plt


if __name__ == "__main__":
	name = 'AAPL'
	start="2015-01-01"
	end="2018-9-14"

	stock = stocktool(name, start, end)
	stock.plot_stock()
	# stock.plot_stock(stats=['Daily Change'])
	# model, model_data = stock.create_prophet_model(days=90)

	# stock.evaluate_prediction()


