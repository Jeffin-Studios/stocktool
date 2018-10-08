
How To Create Training Set for Sentiment Classification
========================================================
List of Stocks
List of Months
List of Years
Iterate through loops and scrape news articles 
verify date of news article
analyze stock trend with stocktool





Identify Interesting Stocks
===========================
1. Iterate through database of stocks that match certain parameters (price, market capitalization, industry, etc)
2. Assess volatility by analyzing stock history (this is also one of the search parameters)
3. Determine overall trends for daily, weekly, and monthly. Look at rolling averages (trend.py), sample multiple intervals to reduce local noise
4. From there, determine if stock is at relative peak or relative dip (daily, weekly, monthly)
5. Also assess long term trend for specific industry market of stock



Apply Strategies
=================
1. Run stocktool script that forecasts and predicts future value (in a week or a month) using fbprophet equation fitting model 
2. Factor in external influence score (if too low, disregard what the trends predict)
	First start with data mining code to gather relevant information from the news and recent events
3. Finance strategy (buy, sell signals) trained over stock history, yielding good return if accuracy  50% (random chance)

Real Time
=========
1. Track stocks real time, analyze momentum (slope) to predict detect dip and peak


Factors Beyond Mathemtical Fitting (historical trends)
======================================================
1. Perception about the company (how many times the company’s name was searched on Google)
2. Performance of other companies working in the same field as this company, competitors (pro or con)
3. Industry developments
4. Politics
5. News Articles (detect bias with machine learning in separte project, implement here)


Sentiment Analysis
==================
We can use a web scraper to get news articles (and headlines) for each of the stocks in the S&P 500 over a few years, and for each news article we can label the sentiment by running our stocktool scripts to see how the stock changed in the days around the date of publication. After we successfully label the effect/sentiment of each news article, we can run text classification on that training set to teach an algorithm how to predict it. Run separate classifiers on the actual news article and the news headlines. 



