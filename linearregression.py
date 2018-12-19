import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

#df is data frame
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PTC'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PTC_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

#           price          x          x           x
df = df[['Adj. Close','HL_PTC','PTC_Change','Adj. Volume']]

forecast_col = 'Adj. Close'
#Fill non-avaliable data with -99999 as we can't have empty data cell (Simply outlier)
df.fillna(-99999, inplace=True)

#rounds to whole (10% of the data frame)
forecast_out = int(math.ceil(0.1*len(df)))
#print (forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)

#Features (as we are dropping lable)
X = np.array (df.drop(['label','Adj. Close'],1))
#Scale the data
X= preprocessing.scale(X)
#last 30days data
X_lately = X[-forecast_out:]
X= X[:-forecast_out]

df.dropna(inplace=True)
#lable
y = np.array (df['label'])

df.dropna(inplace=True)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#Linear Regression
#clf = LinearRegression()
#number of processors for training

clf = LinearRegression(n_jobs=-1) 

#SVM regression, default is liner, we can try with polynomial
#clf = svm.SVR(kernel='poly')

clf.fit(X_train, y_train)

#pickle (saving the train data)
with open('linearregression.pickle', 'wb') as f:
        pickle.dump(clf, f)
#####uncomment this traing the classifer and saving this as pickel
#clf = LinearRegression(n_jobs=-1)
#clf.fit(X_train, y_train)

#with open('linearregression.pickle', 'wb') as f:
#        pickle.dump(clf, f)
######

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

#accuracy
accuracy = clf.score(X_test, y_test)
print (accuracy)

#prediction (next 30 days of stock value)
forecast_set = clf.predict(X_lately)

#next 30 days of predicted stock vlaues
#print (forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix +=one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

#print(df.head())
#print(df.tail())

#Plot
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
