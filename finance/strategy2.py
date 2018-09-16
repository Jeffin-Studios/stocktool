# Machine learning classification libraries
from sklearn.svm import SVC
from sklearn.metrics import scorer
from sklearn.metrics import accuracy_score
 
# For data manipulation
import pandas as pd
import numpy as np
 
# To plot
import matplotlib.pyplot as plt
import seaborn
 
# To fetch data
from pandas_datareader import data as pdr



Df = pdr.DataReader('SPY', 'yahoo', start="2012-01-01", end="2017-10-01")        
Df= Df.dropna()
Df.Close.plot(figsize=(10,5))
plt.ylabel("S&P500 Price")
plt.show()



y = np.where(Df['Close'].shift(-1) > Df['Close'],1,-1)



Df['Open-Close'] = Df.Open - Df.Close
Df['High-Low'] = Df.High - Df.Low
 
X=Df[['Open-Close','High-Low']]



split_percentage = 0.8
split = int(split_percentage*len(Df))
 
# Train data set
X_train = X[:split]
y_train = y[:split]
 
# Test data set
X_test = X[split:]
y_test = y[split:]



cls = SVC().fit(X_train, y_train)




accuracy_train = accuracy_score(y_train, cls.predict(X_train))
 
accuracy_test = accuracy_score(y_test, cls.predict(X_test))
print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))




Df['Predicted_Signal'] = cls.predict(X)

count1 = 0
count2 = 0
change = []
for date, signal in Df['Predicted_Signal'].items():
	if signal == 0:
		count0 +=1
	elif signal ==1:
		count1 +=1
	else:
		count2 +=1
		change.append(date)

# print(count1)
print(change)
# print(count2)

# Calculate log returns
Df['Return'] = np.log(Df.Close.shift(-1) / Df.Close)*100
Df['Strategy_Return'] = Df.Return * Df.Predicted_Signal
Df.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10,5))
plt.ylabel("Strategy Returns (%)")
plt.show()

