import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()

# import data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close'] # close price
    returns = stockData.pct_change() # percent change
    meanReturns = returns.mean() # mean returns
    covMatrix = returns.cov() # covariance matrix
    return meanReturns, covMatrix
stockList = ['^GSPC', '^TNX', 'BTC-USD', '0050.TW', '2330.TW', 'MSFT']
# stockList =['APA','BKR', 'DVN', 'FANG', 'ALL', 'NWSA', 'NWS']
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300) # this time range is important that make the covariance matrix different 
meanReturns, covMatrix = get_data(stockList, startDate, endDate)
# print(stockList)  # check if you get the data
weights = np.random.random(len(meanReturns)) # get random between at [0,1)
weights /= np.sum(weights) # weights matrix equal to 1
# weights = [0.3638,0.0053,0.093,0.0092,0.1252,0.3236,0.0799]
print(weights) # check if the weights matrix is right
# Monte Carlo Method
mc_sims = 10000 # number of simulations
T = 100 #timeframe in days
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns) # means matrix
meanM = meanM.T # takes transpose in order to do computation
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0) # portfolio simulations matrix
initialPortfolio = 10000 # initial portfolio value
for m in range(0, mc_sims): # MC loops
    Z = np.random.normal(size=(T, len(weights)))# uncorrelated RV's
    L = np.linalg.cholesky(covMatrix) # Cholesky decomposition to Lower Triangular Matrix
    dailyReturns = meanM + np.inner(L, Z) # Correlated daily returns for individual stocks
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio # what the portfolio each day and take the accumulated product of daily returns
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a portfolio')
# plt.show()
def mcVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: percentile on return distribution to a given confidence level alpha
    """
    if isinstance(returns, pd.Series): # isinstance() function returns True if the specified object is of the specified type, otherwise False
        return np.percentile(returns, alpha)
    else:
        raise TypeError("Expected a pandas data series.")
def mcCVaR(returns, alpha=5):
    """ Input: pandas series of returns
        Output: CVaR or Expected Shortfall to a given confidence level alpha
    """
    if isinstance(returns, pd.Series):
        belowVaR = returns <= mcVaR(returns, alpha=alpha)
        return returns[belowVaR].mean()
    else:
        raise TypeError("Expected a pandas data series.")
portResults = pd.Series(portfolio_sims[-1,:])
VaR = initialPortfolio - mcVaR(portResults, alpha=5)
CVaR = initialPortfolio - mcCVaR(portResults, alpha=5)
print('VaR ${}'.format(round(VaR,2)))
print('CVaR ${}'.format(round(CVaR,2)))