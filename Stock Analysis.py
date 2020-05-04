#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''In this portfolio project we will be looking at data from the stock market, particularly some technology stocks. We will learn how to use pandas to get stock information, visualize different aspects of it, and finally we will look at a few ways of analyzing the risk of a stock, based on its previous performance history. We will also be predicting future stock prices through a Monte Carlo method!

We'll be answering the following questions along the way:

1.) What was the change in price of the stock over time?
2.) What was the daily return of the stock on average?
3.) What was the moving average of the various stocks?
4.) What was the correlation between different stocks' closing prices?
4.) What was the correlation between different stocks' daily returns?
5.) How much value do we put at risk by investing in a particular stock?
6.) How can we attempt to predict future stock behavior?'''


# In[20]:


import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas_datareader import data, wb


# In[21]:


from __future__ import division


# In[16]:


get_ipython().system('pip install future')


# In[34]:


from datetime import datetime
tech_list = ['AAPL','GOOG','MSFT','AMZN']
end = datetime.now()
start = datetime(end.year-1,end.month,end.day)
for stock in tech_list:
    globals()[stock] = data.DataReader(stock,'yahoo',start,end)


# In[ ]:


#Stock Analysis


# In[33]:


AAPL.describe()


# In[35]:


AAPL.info()


# In[36]:


AAPL['Adj Close'].plot(legend=True,figsize=(10,4))


# In[37]:


AAPL['Volume'].plot(legend=True,figsize=(10,4))


# In[40]:


ma_day = [10,20,50]
for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    AAPL[column_name] = AAPL['Adj Close'].rolling(ma).mean()


# In[47]:


AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False)


# In[50]:


AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(legend=True,marker='o',linestyle="--")


# In[52]:


sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color="purple")


# In[53]:


AAPL['Daily Return'].hist(bins=100)


# In[56]:


closing_df = data.DataReader(tech_list,'yahoo',start,end)['Adj Close']
closing_df.head()


# In[58]:


tech_rets = closing_df.pct_change()
tech_rets.head()


# In[59]:


sns.jointplot('GOOG','GOOG',tech_rets,kind = "scatter",color='seagreen')


# In[60]:


sns.jointplot('GOOG','MSFT',tech_rets,kind="scatter")


# In[61]:


from IPython.display import SVG
SVG(url='http://upload.wikimedia.org/wikipedia/commons/d/d4/Correlation_examples2.svg')


# In[63]:


#correlation between different stocks
sns.pairplot(tech_rets.dropna())


# In[66]:


return_fig = sns.PairGrid(tech_rets.dropna())
return_fig.map_upper(plt.scatter,color='purple')
return_fig.map_lower(sns.kdeplot,cmap='cool_d')
return_fig.map_diag(plt.hist,bins=30)


# In[67]:


return_fig = sns.PairGrid(closing_df)
return_fig.map_upper(plt.scatter,color='purple')
return_fig.map_lower(sns.kdeplot,cmap='cool_d')
return_fig.map_diag(plt.hist,bins=30)


# In[ ]:


#apple and microsoft and google are highly corealted as we can see in the graph above


# In[75]:


sns.heatmap(tech_rets.dropna())


# In[76]:


sns.heatmap(closing_df)


# In[ ]:


#Risk Analysis


# In[81]:


rets = tech_rets.dropna()
area = np.pi*20

plt.scatter(rets.mean(), rets.std(),alpha = 0.5,s =area)
plt.xlabel('Expected Return')
plt.ylabel('Risk')

# Label the scatter plots, for more info on how this is done, chekc out the link below
# http://matplotlib.org/users/annotations_guide.html
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))


# In[ ]:


# Apple has the low risk and high return


# In[ ]:


# Risk Value
# We can treat value at risk as the amount of money we could expect to lose (aka putting at risk) for a given confidence interval. Theres several methods we can use for estimating a value at risk.


# In[82]:


sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')


# In[83]:


rets.head()


# In[84]:


rets['AAPL'].quantile(0.05)


# In[ ]:


#The 0.05 empirical quantile of daily returns is at -0.034. That means that with 95% confidence, our worst daily loss will not exceed 3.4%. If we have a 1 million dollar investment, our one-day 5% VaR is 0.034 * 1,000,000 = $34,000.


# In[ ]:


# Value at Risk using the Monte Carlo method


# In[85]:


days = 365
dt = 1/days
mu = rets.mean()['GOOG']
sigma = rets.std()['GOOG']


# In[90]:


def stock_monte_carlo(start_price,days,mu,sigma):
    price = np.zeros(days)
    price[0] = start_price
    shock = np.zeros(days)
    drift = np.zeros(days)
    for x in range(1,days):
        shock[x] = np.random.normal(loc=mu*dt,scale=sigma*np.sqrt(dt))
        drift[x] = mu*dt
        price[x] = price[x-1] + (price[x-1]*(drift[x]+shock[x]))
    return price


# In[91]:


GOOG.head()


# In[93]:


start_price = 1173.650024
for run in range(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for Google')


# In[96]:


runs = 1000
simulations = np.zeros(runs)
for run in range(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1]


# In[97]:


q = np.percentile(simulations,1)
plt.hist(simulations,bins=200)
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold');


# In[ ]:


# Now we have looked at the 1% empirical quantile of the final price distribution to estimate the Value at Risk for the Google stock, which looks to be $59.45 for every investment of $1114.20 (the price of one inital google stock).

# This basically means for every initial stock you purchase your putting about $59.45 at risk 99% of the time from our Monte Carlo Simulation.

