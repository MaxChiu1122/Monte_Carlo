import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pandas_datareader import data as web
import scipy.optimize as opt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()
plt.style.use('ggplot')
#symbols = ['AAPL','GOOG','AMZN']
symbols = [ 'MSFT','GOOG', 'AMZN', 'AMAT']
# symbols = ['ATVI','GOOG', 'FOXA', 'LUMN', 'NWSA', 'NWS','TWTR','AMZN']

def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close'] # close price
    returns = stockData.pct_change() # percent change
    meanReturns = returns.mean() # mean returns
    covMatrix = returns.cov() # covariance matrix
    return stockData
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)
stockdata = get_data(symbols, startDate, endDate)
adj_close = stockdata
# defining utility functions

def calc_returns(price_data, resample=None, ret_type="arithmatic"):
    """
    Parameters
        price_data: price timeseries pd.DataFrame object.
        resample:   DateOffset, Timedelta or str. `None` for not resampling. Default: None
                    More on Dateoffsets : https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        ret_type:   return calculation type. \"arithmatic\" or \"log\"

    Returns:
        returns timeseries pd.DataFrame object
    """
    if ret_type=="arithmatic":
        ret = price_data.pct_change().dropna()
    elif ret_type=="log":
        ret = np.log(price_data/price_data.shift()).dropna()
    else:
        raise ValueError("ret_type: return calculation type is not valid. use \"arithmatic\" or \"log\"")

    if resample != None:
        if ret_type=="arithmatic":
            ret = ret.resample(resample).apply(lambda df: (df+1).cumprod(axis=0).iloc[-1]) - 1
        elif ret_type=="log":
            ret = ret.resample(resample).apply(lambda df: df.sum(axis=0))
    return(ret)
        

def calc_returns_stats(returns):
    """
    Parameters
        returns: returns timeseries pd.DataFrame object

    Returns:
        mean_returns: Avereage of returns
        cov_matrix: returns Covariance matrix
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return(mean_returns, cov_matrix)

def portfolio(weights, mean_returns, cov_matrix):

    portfolio_return = np.dot(weights.reshape(1,-1), mean_returns.values.reshape(-1,1))
    portfolio_var = np.dot(np.dot(weights.reshape(1,-1), cov_matrix.values), weights.reshape(-1,1))
    portfolio_std = np.sqrt(portfolio_var)

    return(np.squeeze(portfolio_return),np.squeeze(portfolio_var),np.squeeze(portfolio_std))

monthly_ret_arith = calc_returns(adj_close, resample='BM', ret_type="arithmatic")
monthly_ret_log = calc_returns(adj_close, resample='BM', ret_type="log")
equal_weights = np.array([1/len(symbols) for _ in range(len(symbols))])

mean_returns, cov_matrix = calc_returns_stats(monthly_ret_arith)
portfolio_return, portfolio_var, portfolio_std = portfolio(equal_weights, mean_returns, cov_matrix)
#print("monthly portfolio return:\t%", round(100*portfolio_return.item(),2))
#print("\nmonthly portfolio variance:\t%",round(100*portfolio_var.item(),2))
#print("\nmean monthly returns of individual assets:\n",mean_returns)
#print("\ncovariance matrix of assets returns:\n",cov_matrix)
num_iter = 1000000

porfolio_var_list = []
porfolio_ret_list = []
w_list =[]

max_sharpe = 0
max_sharpe_var = None
max_sharpe_ret = None
max_sharpe_w = None

daily_ret = calc_returns(adj_close, resample=None, ret_type="log")
mean_returns, cov_matrix = calc_returns_stats(daily_ret)

for i in range(1,num_iter+1):
    rand_weights = np.random.random(len(symbols))
    rand_weights = rand_weights/np.sum(rand_weights)

    porfolio_ret, porfolio_var, portfolio_std = portfolio(rand_weights, mean_returns, cov_matrix)

    # Anuualizing
    porfolio_ret = porfolio_ret * 252
    porfolio_var = porfolio_var * 252
    portfolio_std = portfolio_std * (252**0.5)

    sharpe = (porfolio_ret/(porfolio_var**0.5)).item()
    if sharpe > max_sharpe:
        max_sharpe = sharpe
        max_sharpe_var = porfolio_var.item()
        max_sharpe_ret = porfolio_ret.item()
        max_sharpe_w = rand_weights

    porfolio_var_list.append(porfolio_var)
    porfolio_ret_list.append(porfolio_ret)
    w_list.append(rand_weights)
    if ((i/num_iter)*100)%10 == 0:
        print(f'%{round((i/num_iter)*100)}...',end='')
stat = np.hstack([np.array(porfolio_ret_list).reshape(-1,1),
                 np.array(porfolio_var_list).reshape(-1,1),
                 (np.array(porfolio_ret_list)/np.sqrt(np.array(porfolio_var_list))).reshape(-1,1)]
                 )
stat=pd.DataFrame(stat, columns=['Return','Variance','Sharpe ratio'])
# minimum risk porfolio

min_vol = stat[stat['Variance'] == stat['Variance'].min()]
min_vol_index = min_vol.index.item()
min_vol_weights = w_list[min_vol_index]
print("Portfolio with minimum volatility:\n")
print(f"Annual Sharpe Ratio: {round(stat['Sharpe ratio'][min_vol_index],2)} | Annual Return: % {round(stat['Return'][min_vol_index]*100,2)} | Annual Volatility: % {round(stat['Variance'][min_vol_index]*100,2)}\n")
for index,symbol in enumerate(symbols):
    print(f'{symbol}:\t% {round(min_vol_weights[index]*100,2)}')
# maximum sharpe ratio porfolio

#print("Portfolio with maximum sharpe ratio:\n")
#print(f"Annual Sharpe Ratio: {round(max_sharpe,2)} | Annual Return: % {round(max_sharpe_ret*100,2)} | Annual Volatility: % {round(max_sharpe_var*100,2)}\n")
#for index,symbol in enumerate(symbols):
    #print(f'{symbol}:\t% {round(max_sharpe_w[index]*100,2)}')
#plotting the portfolios

daily_ret_var = daily_ret.var()
# plt.figure(figsize=(10,8))
# plt.scatter(porfolio_var_list,porfolio_ret_list,c=stat['Sharpe ratio'], alpha=0.8, cmap='plasma')
# for sym in symbols:
#     plt.scatter(daily_ret_var.loc[sym]*252, mean_returns.loc[sym]*252, marker='x', s=100, label=sym)

# plt.scatter([max_sharpe_var], [max_sharpe_ret], marker='*', s=250, label='max sharpe porfolio', c='blue')
# plt.scatter(min_vol['Variance'].values, min_vol['Return'].values, marker='*', s=250, label='minimum volatility porfolio', c='green')

# plt.xlabel('Variance')
# plt.ylabel('Return')
# plt.title('Portfolio Optimization using Monte Carlo Simulation\nlong-only fully invested porfolios')
# plt.legend(loc='upper right')

# cmap = mpl.cm.plasma
# norm = mpl.colors.Normalize(vmin=stat['Sharpe ratio'].min(), vmax=stat['Sharpe ratio'].max())
# plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),label='Sharpe Ratio',);


# Optimization using scipy optimize module (Sequential Least Squares Programming)

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    portfolio_return, portfolio_var, portfolio_std = portfolio(weights, mean_returns, cov_matrix)
    sr = ((portfolio_return - risk_free_rate)/portfolio_std) * (252**0.5) # annualized
    return(-sr)

def portfolio_variance(weights, mean_returns, cov_matrix):
    portfolio_return, portfolio_var, portfolio_std = portfolio(weights, mean_returns, cov_matrix)
    return(portfolio_var*252)
daily_ret = calc_returns(adj_close, resample=None, ret_type="log")
mean_returns, cov_matrix = calc_returns_stats(daily_ret)
def optimize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0, w_bounds=(0,1)):
    "This function finds the portfolio weights which minimize the negative sharpe ratio"

    init_guess = np.array([1/len(mean_returns) for _ in range(len(mean_returns))])
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = opt.minimize(fun=neg_sharpe_ratio,
                          x0=init_guess,
                          args=args,
                          method='SLSQP',
                          bounds=tuple(w_bounds for _ in range(len(mean_returns))),
                          constraints=constraints,
                          )
    
    if result['success']:
        print(result['message'])
        opt_sharpe = - result['fun']
        opt_weights = result['x']
        opt_return, opt_variance, opt_std = portfolio(opt_weights, mean_returns, cov_matrix)
        return(opt_sharpe, opt_weights, opt_return.item()*252, opt_variance.item()*252, opt_std.item()*(252**0.5))
    else:
        print("Optimization operation was not succesfull!")
        print(result['message'])
        return(None)
opt_sharpe, opt_weights, opt_return, opt_variance, opt_std = optimize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0, w_bounds=(0,1))
# maximum sharpe ratio porfolio
print("Portfolio with maximum sharpe ratio:\n")
print(f"Annual Sharpe Ratio: {round(opt_sharpe,2)} | Annual Return: % {round(opt_return*100,2)} | Annual Volatility: % {round(opt_variance*100,2)}\n")
for index,symbol in enumerate(symbols):
    print(f'{symbol}:\t% {round(opt_weights[index]*100,2)}')
def minimize_portfolio_variance(mean_returns, cov_matrix, w_bounds=(0,1)):
    "This function finds the portfolio weights which minimize the portfolio volatility(variance)"

    init_guess = np.array([1/len(mean_returns) for _ in range(len(mean_returns))])
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    result = opt.minimize(fun=portfolio_variance,
                          x0=init_guess,
                          args=args,
                          method='SLSQP',
                          bounds=tuple(w_bounds for _ in range(len(mean_returns))),
                          constraints=constraints,
                          )
    
    if result['success']:
        print(result['message'])
        min_var = result['fun']
        min_var_weights = result['x']
        min_var_return, min_var_variance, min_var_std = portfolio(min_var_weights, mean_returns, cov_matrix)
        min_var_sharpe = (min_var_return/min_var_std)*(252**0.5)
        return(min_var_sharpe, min_var_weights, min_var_return.item()*252, min_var_variance.item()*252, min_var_std.item()*(252**0.5))
    else:
        print("Optimization operation was not succesfull!")
        print(result['message'])
        return(None)
min_var_sharpe, min_var_weights, min_var_return, min_var_variance, min_var_std = minimize_portfolio_variance(mean_returns, cov_matrix, w_bounds=(0,1))
# minimum volatility porfolio
# print("Portfolio with maximum sharpe ratio:\n")
# print(f"Annual Sharpe Ratio: {round(min_var_sharpe,3)} | Annual Return: % {round(min_var_return*100,2)} | Annual Volatility: % {round(min_var_variance*100,3)}\n")
# for index,symbol in enumerate(symbols):
#     print(f'{symbol}:\t% {round(min_var_weights[index]*100,2)}')
# def calc_portfolio_return(weights, mean_returns, cov_matrix):
#     portfolio_return, portfolio_var, portfolio_std = portfolio(weights, mean_returns, cov_matrix)
#     return(portfolio_return.item()*252)

def efficient_portfolio(mean_returns, cov_matrix, target_return, w_bounds=(0,1)):
    """retuens the portfolio weights with minimum variance for a specific level of expected portfolio return"""    

    init_guess = np.array([1/len(mean_returns) for _ in range(len(mean_returns))])
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x)-1},
                   {'type': 'eq', 'fun': lambda x: 252*np.squeeze(np.dot(x.reshape(1,-1),mean_returns.values.reshape(-1,1))) - target_return})
    result = opt.minimize(fun=portfolio_variance,
                          x0=init_guess,
                          args=args,
                          method='SLSQP',
                          bounds=tuple(w_bounds for _ in range(len(mean_returns))),
                          constraints=constraints,
                          )
    if not result['success']:
        print(result['message'])
    efficient_variance = result['fun']
    efficient_weights = result['x']
    efficient_return, _ , efficient_std = portfolio(efficient_weights, mean_returns, cov_matrix)
    efficient_sahrpe = (efficient_return/efficient_std)*(252**0.5)
    return(efficient_sahrpe, efficient_weights, efficient_return.item()*252, efficient_variance, efficient_std.item()*(252**0.5))
expected_return = 0.05
efficient_sharpe, efficient_weights, efficient_return, efficient_variance, efficient_std = efficient_portfolio(mean_returns,
                                                                                                                cov_matrix,
                                                                                                                target_return=expected_return,
                                                                                                                w_bounds=(0,1))
# efficient porfolio
print("Efficient portfolio for 5% return level:\n")
print(f"Annual Sharpe Ratio: {round(efficient_sharpe ,3)} | Annual Return: % {round(efficient_return*100,2)} | Annual Volatility: % {round(efficient_variance*100,3)}\n")
for index,symbol in enumerate(symbols):
    print(f'{symbol}:\t% {round(efficient_weights[index]*100,2)}')

#plotting the Efficient Frontier

daily_ret_var = daily_ret.var()
plt.figure(figsize=(10,8))

target_rets = np.arange(-0.14,0.30,0.002)
efficient_vars = np.array([efficient_portfolio(mean_returns,cov_matrix,target_return=x ,w_bounds=(0,1))[3] for x in target_rets])
plt.plot(efficient_vars,target_rets,marker='.',label="efficient frontier")

plt.scatter(porfolio_var_list,porfolio_ret_list,c=stat['Sharpe ratio'], alpha=0.8, cmap='plasma')
for sym in symbols:
    plt.scatter(daily_ret_var.loc[sym]*252, mean_returns.loc[sym]*252, marker='x', s=100, label=sym)

plt.scatter([opt_variance], [opt_return], marker='*', s=250, label='max sharpe p', c='blue')
plt.scatter(min_var_variance, min_var_return, marker='*', s=250, label='min vol p', c='green')

plt.xlabel('Variance')
plt.ylabel('Return')
plt.title('Efficient Frontier')
plt.legend(loc='upper right')

cmap = mpl.cm.plasma
norm = mpl.colors.Normalize(vmin=stat['Sharpe ratio'].min(), vmax=stat['Sharpe ratio'].max())
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),label='Sharpe Ratio',);
plt.show()