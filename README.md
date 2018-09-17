Identify Interesting Stocks
===========================
1. Iterate through database of stocks that match certain parameters (price, market capitalization, industry, etc)
2. Assess volatility by analyzing stock history (this is also one of the search parameters)
3. Determine overall trends for daily, weekly, and monthly. Look at rolling averages (trend.py), sample multiple intervals to reduce local noise
4. From there, determine if stock is at relative peak or relative dip (daily, weekly, monthly)



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