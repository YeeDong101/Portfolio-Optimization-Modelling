# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 11:49:36 2024

@author: jason wang
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
# 1.
# Obtain the data for the following assets from May 2018 to May 2023
# In Excel file, we choose a Treasury Bonds ETF, now we replace it with MSFT
assets = ['MSFT','AAPL','AMZN','TGT','PG','GLD']
df = pd.DataFrame()
for x in assets:
    df[x] = yf.download(x, start='2018-04-30', end='2023-06-01', interval='1mo')['Adj Close']
    df.rename(columns={'Adj Close':x})
df
# Check if null value exist
df.isnull().sum() # No
# Calculate Monthly Return
df_return = df.pct_change() # Series
df_return = pd.DataFrame(df_return)
df_return.drop(index=df.index[0], axis=0, inplace=True)
df_return
# Average and Annualize it
avg = pd.DataFrame(df_return.mean()) #Series
avg_return = avg.set_axis(['Average Return'],axis=1)
avg_return['Annualized Average Return']=avg_return['Average Return']*12
avg_return
# Get Sample Standard Deviation with numpy function std
avg_return['Standard Deviation']=df_return.std(ddof=1) # ddof=1 means unbiased(sample)
avg_return['Annualized Standard Deviation']=avg_return['Standard Deviation']*12**0.5
avg_return.iloc[:,2:4] # use iloc to look at std
# Constuct MVF (minimum variance frontier) with these 6 assets 
# First we get Covariance Matrix
df_return.corr()
cov_matrix = df_return.cov(ddof=1)*12 # Annualize it
cov_matrix = pd.DataFrame(cov_matrix)
cov_matrix = cov_matrix.set_axis(assets, axis=1)
cov_matrix = cov_matrix.set_axis(assets, axis=0)
cov_matrix
# Then we find the MVP (minimum variance portfolio) using Scipy function "minimize"
from scipy.optimize import minimize
asset_returns = avg_return['Annualized Average Return']
portfolio_size = len(assets) 
def minimize_risk(cov_matrix, portfolio_size):
    # To get MVP, we have to minimize the portfolio std
    def objective(x,cov_matrix):
        return np.dot(x,np.dot(cov_matrix,x))
    # Constraint of weights (sum of x)
    def eq_constraint(x):
        return np.sum(x)-1
    # Initial value we guess, which is the eqaual weight (1/6)
    x_init = np.repeat(1/portfolio_size,portfolio_size)
    constraints={'type':'eq','fun':eq_constraint}
    bounds = tuple([(0,1) for _ in range(portfolio_size)])
    opt = minimize(objective,x0=x_init,args=(cov_matrix,)\
                   ,bounds=bounds,constraints=constraints)
    return opt       
result = minimize_risk(cov_matrix, portfolio_size)
weights_mvp = result.x
std_mvp = np.sqrt(result.fun)
return_mvp = np.dot(asset_returns,weights_mvp)
weights_mvp,std_mvp,return_mvp # The MVP we want
# Given expected returns, we need to minimize std
max_ret=asset_returns.max() # The max return we can get
portfolio_returns = []
portfolio_std = []
# To get MVP, we have to minimize the portfolio std
for target_return in range(10*int(100*max_ret)): 
    def minimize_risk_with_return(asset_returns,cov_matrix, portfolio_size):
        def objective(x,cov_matrix):
            return np.dot(x,np.dot(cov_matrix,x))
        # Constraint of weights (sum of x)
        def eq_constraint(x):
            return np.sum(x)-1
        # Return
        def ret_constraint(x):
            expected_return = np.dot(asset_returns,x)
            return expected_return*100 - target_return/10
        # Initial value we guess, which is the eqaual weight (1/6)
        x_init = np.repeat(1/portfolio_size,portfolio_size)
        constraints=({'type':'eq','fun':eq_constraint},\
                     {'type':'eq','fun':ret_constraint})
        bounds = tuple([(0,1) for _ in range(portfolio_size)])
        opt = minimize(objective,x0=x_init,args=(cov_matrix,)\
                       ,bounds=bounds,constraints=constraints)
        return opt
    result = minimize_risk_with_return(asset_returns,cov_matrix, portfolio_size)
    portfolio_returns.append(np.dot(asset_returns,result.x))
    portfolio_std.append(np.sqrt(result.fun))
portfolio_returns = np.array(portfolio_returns)
portfolio_std = np.array(portfolio_std)
# Plot the Minimum Variance Frontier
portfolios = pd.DataFrame({'Return':portfolio_returns,'Std':portfolio_std})
portfolios.plot(x='Std',y='Return',kind='scatter', figsize=(15,10))
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.plot(std_mvp, return_mvp, "or",ms=15)
plt.text(std_mvp+.003, return_mvp+.003,"MVP" ,fontsize=12)
plt.show() # MVF and MVP
# Now, with risk-free lending rate of 1.40%
# and borrowing rate, we find Optimal Risky Portfolio (ORP1 & orp2)
risk_free_rate = {0:0.014,1:0.05}
orp_std =[]
orp_return=[]
orp_weights=[]
portfolios.plot(x='Std',y='Return',kind='scatter', figsize=(15,10),linewidth=2)

for i in risk_free_rate:
    def orp(asset_returns, cov_matrix, portfolio_size):
        # Maximize the Sharpe Ratio (slope), multiply by 1
        def sharpe_ratio(x,cov_matrix):
            return (-1)*(np.dot(asset_returns, x)-risk_free_rate[i])/(np.sqrt(np.dot(x,np.dot(cov_matrix,x))))
        # Constraint of weights (sum of x)
        def eq_constraint(x):
            return np.sum(x)-1
        # Initial value we guess, which is the eqaual weight (1/6)
        x_init = np.repeat(1/portfolio_size,portfolio_size)
        constraints=({'type':'eq','fun':eq_constraint})
        bounds = tuple([(0,1) for _ in range(portfolio_size)])
        opt = minimize(sharpe_ratio,x0=x_init,args=(cov_matrix,)\
                       ,bounds=bounds,constraints=constraints)
        return opt
    result1 = orp(asset_returns, cov_matrix, portfolio_size)
    orp_std.append(np.sqrt(np.dot(result1.x,np.dot(cov_matrix,result1.x))))
    orp_return.append(np.dot(asset_returns,result1.x))
    orp_weights.append(result1.x)
ORP={'Std':orp_std,'Return':orp_return,'Weights':orp_weights}
ORP # ORP1 & ORP2
# Plot it on the graph
portfolios.plot(x='Std',y='Return',kind='scatter', figsize=(15,10))
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.plot(std_mvp, return_mvp, "or",ms=15)
plt.text(std_mvp+.003, return_mvp+.003,s="MVP" ,fontsize=12)

plt.plot(orp_std, orp_return, "or",ms=15)
plt.plot([0,orp_std[0]],[risk_free_rate[0],orp_return[0]],color='orange',linewidth=3)
plt.plot([0,orp_std[1],2*orp_std[1]],[risk_free_rate[1],\
                         orp_return[1],2*(orp_return[1]-risk_free_rate[1])]\
                         ,color='green',linewidth=3)
plt.text(orp_std[0]+.007,orp_return[0],s='ORP1',fontsize=12)
plt.text(orp_std[1]+.007,orp_return[1],s='ORP2',fontsize=12)
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.show() # MVF, MVP, ORP1 & ORP2
# Given Utility Function U=E(r)-3*Std**2, namely A=6, find Optimal Complete Portfolio (OCP)
# It can be either on orange line y*(rl)<1, MVF y*(r)=1, or green line y*(rb)>1
for i in risk_free_rate:
    y = (orp_return[i]-risk_free_rate[i])/(6*orp_std[i]**2)
    ErC = (1-y)*risk_free_rate[i]+y*orp_return[i]
    std_c = (y*orp_std[i])
    U = (ErC-3*std_c**2)
    print('For', f'OCP{i+1} y*=', y,'E(r)=', ErC,'Std=', std_c,'Utility', U)\
# It seems that OCP2 is the true Optimal Complete Portfolio since y>1 
# (and for OCP1, y>1 where y should be <= 1)
y = (orp_return[1]-risk_free_rate[1])/(6*orp_std[1]**2)
ErC = (1-y)*risk_free_rate[1]+y*orp_return[1]
std_c = (y*orp_std[1])
U = (ErC-3*std_c**2)
# Graph it
portfolios.plot(x='Std',y='Return',kind='scatter', figsize=(15,10))
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.plot(std_mvp, return_mvp, "or",ms=15)
plt.text(std_mvp+.003, return_mvp+.004,s="MVP" ,fontsize=15)
plt.plot(orp_std, orp_return, "or",ms=15)
plt.plot([0,orp_std[0]],[risk_free_rate[0],orp_return[0]],color='orange',linewidth=3)
plt.plot([0,orp_std[1],2*orp_std[1]],[risk_free_rate[1],\
                         orp_return[1],2*(orp_return[1]-risk_free_rate[1])]\
                         ,color='green',linewidth=3)
plt.text(orp_std[0]+.007,orp_return[0]-.003,s='ORP1',fontsize=15)
plt.text(orp_std[1]+.007,orp_return[1]-.005,s='ORP2',fontsize=15)

plt.plot(std_c,ErC,"or",ms=15)
plt.text(std_c+.011, ErC, s="OCP" ,fontsize=15)
# Define the range of x and y values for plotting
x_values = np.linspace(0, 0.5, 400)
y_values = np.linspace(0, 0.5, 400)
# Create a grid of x and y values
X, Y = np.meshgrid(x_values, y_values)
Z = Y-3*(X**2)
plt.contour(X, Y, Z, levels=[U], colors='r')
plt.contour(X, Y, Z+0.05, levels=[U], colors='r')
plt.contour(X, Y, Z-0.05, levels=[U], colors='r')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.show() # MVF, MVP, ORP1 & ORP2 and OCP

# 2.
# Getting the monthly data of the Fama-French 3-factor model from Kenneth French Data Library 
ff_df = pd.read_csv('C:/Users/jason wang/Desktop/大學課業/三上/投資資產組合/Exercise 1/Fama French 3 Factors.CSV',skiprows=[0,1,2])
ff_df = ff_df.rename(columns={'Unnamed: 0':'Date'})
ff_df = ff_df.iloc[:1171]
ff_df['Date']=ff_df['Date'].astype(int)
ff_df = ff_df[(ff_df['Date']>201804) & (ff_df['Date']<202306)]
ff_df['Date']=pd.to_datetime(ff_df['Date'], format='%Y%m')
ff_df = ff_df.set_index(['Date'])
ff_df = ff_df.astype(float).multiply(0.01)
ff_df.head(10),ff_df.tail(10)
# Calculate Monthly Return based on the weight of OCP
ocp_weights = list(y*orp_weights[1].astype(float))
ocp_weights.append((1-y))
m_return = pd.concat([df_return,ff_df['RF']], axis=1)
m_return = m_return.drop('2018-05-01',axis=0)
mo_return = np.dot(ocp_weights,m_return.T)
# Get Portfolio Excess Return & Market Return
Return = pd.DataFrame()
Return['Monthly Excess Return'] = list(mo_return)-m_return['RF']
y = Return['Monthly Excess Return']
Return['Market Excess Return'] = ff_df['Mkt-RF'].iloc[1:]
X = Return['Market Excess Return']
Return['Market Return'] = Return['Market Excess Return'] + m_return['RF']
# Compare Excess Return and Market Return with Linear Regression
from scipy.stats import linregress
test_result = linregress(X, y)
test_result
# Examine the Alpha & Beta
import statsmodels.api as sm
X1 = sm.add_constant(X)
res = sm.OLS(y, X1).fit()
res.summary() # It seems that both Alpha & Beta are significant
alpha = test_result.intercept
beta = test_result.slope
# Draw the graph
plt.scatter(X,y,color='red')
plt.plot(X,X*beta+alpha)
plt.show()
# Evaluate the performance of our OCP
# Sharpe Ratio
ER = y.mean()*12
StdE = mo_return.std(ddof=1)*(12**0.5)
Sharpe_ratio = ER / StdE
Sharpe_ratio
# Treynor Measure
Treynor_measure = 100*ER / beta
Treynor_measure
# M^2 Measure
MER = X.mean()*12
Mstde = Return['Market Return'].std(ddof=1)*(12**0.5)
MSP = MER / Mstde
M2_measure =(Sharpe_ratio - MSP)* Mstde
M2_measure
# Information Ratio
R2 = 0.629 # From the table 
Information_ratio = alpha / ((1-R2)*StdE**2)*0.5
Information_ratio
# Fama French 3 Factors Model  
X3 = ff_df[['Mkt-RF','SMB','HML']].iloc[1:]
X3_ = sm.add_constant(X3)
res = sm.OLS(y, X3_).fit()
res.summary() # It seems that both Alpha & Beta are significant
res.params = res.params.rename({'const':'Alpha'}) # Alpha and Beta
# Graph
fig, ax = plt.subplots(2,2)
ax[0][0].scatter(X3['Mkt-RF'],y,color='red')
ax[0][0].plot(X3['Mkt-RF'],X3['Mkt-RF']*res.params[1]+res.params[0])
ax[0][0].set_title('Mkt-RF')
ax[0][1].scatter(X3['SMB'],y,color='red')
ax[0][1].plot(X3['SMB'],X3['SMB']*res.params[2]+res.params[0])
ax[0][1].set_title('SMB')
ax[1][0].scatter(X3['HML'],y,color='red')
ax[1][0].plot(X3['HML'],X3['HML']*res.params[3]+res.params[0])
ax[1][0].set_title('HML',y=-0.4)
plt.show()