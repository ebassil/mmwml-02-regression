import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import math
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import svm


def getStockHistory(stockTicker, sourceSite, startDate, endDate):
    df = web.DataReader(stockTicker, sourceSite, startDate, endDate)
    return df

def computeMovingAverage(close_px, window):
    mavg = close_px.rolling(window=window).mean()
    return mavg

def stocksRRR(retscomp):
    plt.scatter(retscomp.mean(), retscomp.std())
    plt.xlabel('Expected returns')
    plt.ylabel('Risk')
    for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
        plt.annotate(
            label, 
            xy = (x, y), xytext = (20, -20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    plt.show()



def analyzeCompetitorStock():
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2017, 1, 11)
    dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']
    print(dfcomp.tail())
    
    # correlation analysis
    retscomp = dfcomp.pct_change()
    corr = retscomp.corr()

    # scatter plot theh result
    plt.scatter(retscomp.AAPL, retscomp.GE)
    plt.xlabel('Returns AAPL')
    plt.ylabel('Returns GE')
    plt.show()

    # Kernet Density Estimate (KDE)
    pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10))
    plt.show()

    # heat map corrolation
    plt.imshow(corr, cmap='hot', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns)
    plt.yticks(range(len(corr)), corr.columns)
    plt.show()

    # stocks returns rate and risk
    stocksRRR(retscomp)

def predictAndPlot(clf, X_lately, dfreg, confidence, forecast_out):
    forecast_set = clf.predict(X_lately)
    dfreg['Forecast'] = np.nan
    print(forecast_set, confidence, forecast_out)

    # Plot
    dfreg['Forecast'] = np.nan
    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)

    for i in forecast_set:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
    dfreg['Adj Close'].tail(500).plot()
    dfreg['Forecast'].tail(500).plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(type(clf).__name__)
    plt.show()

def predictStockPrices(df):
    dfreg = df.loc[:,['Adj Close', 'Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    print(dfreg.tail())

    # cleanup process
    # --------------------------------------------------------------------

    # drop missing values
    dfreg.fillna(value=-99999, inplace=True)

    # we want to separate 1 percent of the data to forecast
    forecast_out = int(math.ceil(0.01 * len(dfreg)))
    # we want to predict the AdjClose
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'],1))

    # scale the X so that everyone can have the same distribution for linear regression
    X = sk.preprocessing.scale(X)

    # find data series of late X and early X (train) for model generation and evaluation
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    print('Dimension of X',X.shape)
    print('Dimension of y',y.shape)

    # Separation of training and testing of model by cross validation train test split
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.2)

    # Linear Regression
    clfreg = LinearRegression(n_jobs=-1)
    clfreg.fit(X_train, y_train)

    # Lasso Regression
    clflasso = Lasso(selection='random')
    clflasso.fit(X_train, y_train)

    # Quadratic Regression 2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2.fit(X_train, y_train)

    # Quadratic Regression 3
    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    clfpoly3.fit(X_train, y_train)

    # KNN
    clfknn = KNeighborsRegressor(n_neighbors=2)
    clfknn.fit(X_train, y_train)

    # Test the models
    confidencereg = clfreg.score(X_test, y_test)
    confidencepoly2 = clfpoly2.score(X_test,y_test)
    confidencepoly3 = clfpoly3.score(X_test,y_test)
    confidenceknn = clfknn.score(X_test, y_test)
    confidencelasso = clflasso.score(X_test, y_test)

    print("The linear regression confidence is ",confidencereg)
    print("The quadratic regression 2 confidence is ",confidencepoly2)
    print("The quadratic regression 3 confidence is ",confidencepoly3)
    print("The knn regression confidence is ",confidenceknn)
    print("The knn lasso confidence is ",confidencelasso)
    
    # Predict
    predictAndPlot(clfreg, X_lately, dfreg.copy(), confidencereg, forecast_out)
    predictAndPlot(clfpoly2, X_lately, dfreg.copy(), confidencepoly2, forecast_out)
    predictAndPlot(clfpoly3, X_lately, dfreg.copy(), confidencepoly3, forecast_out)
    predictAndPlot(clfknn, X_lately, dfreg.copy(), confidenceknn, forecast_out)
    predictAndPlot(clflasso, X_lately, dfreg.copy(), confidencelasso, forecast_out)


def forecast():
    # reading stock from yahoo finance
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2017, 1, 11)
    df = getStockHistory('AAPL', 'yahoo', start, end)
    print(df.tail())

    #compute rolling mean and rate of return
    close_px = df['Adj Close']
    print(close_px.tail())
    mavg = computeMovingAverage(close_px, 100)

    # Adjusting the size of matplotlib
    mpl.rc('figure', figsize=(8, 7))
    print("mpl version = ", mpl.__version__)
    
    # Adjusting the style of matplotlib
    style.use('ggplot')
    close_px.plot(label='AAPL')
    mavg.plot(label='mavg')
    plt.legend()
    plt.show()

    # compute returns
    rets = close_px / close_px.shift(1) - 1
    rets.plot(label='return')
    plt.show()

    # Analysing competitors stock
    analyzeCompetitorStock()

    # Prediting stock prices
    predictStockPrices(df)


if __name__ == "__main__":
    forecast()