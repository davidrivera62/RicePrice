#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools
#from tabulate import tabulate


# In[2]:


#Web Scraping - https://pythonprogramminglanguage.com/web-scraping-with-pandas-and-beautifulsoup/
res = requests.get("http://www.fedearroz.com.co/new/precios.php")
soup = BeautifulSoup(res.content,'lxml')
table = soup.find_all('table')[0] 
df = pd.read_html(str(table))
#print( tabulate(df[0], headers='keys', tablefmt='psql') )


# In[3]:


arroz=df[0]
arroz


# In[ ]:


#Data Wrangling
arroz=df[0]
arroz.columns = arroz.iloc[0]
arroz = arroz.set_index(['Mes'])
arroz = arroz.drop(['Mes'])
arroz = arroz.rename(index={'Enero': '1', 
                            'Febrero': '2', 
                            'Marzo': '3', 
                            'Abril': '4', 
                            'Mayo': '5',
                            'Junio': '6',
                            'Julio': '7',
                            'Agosto': '8',
                            'Septiembre': '9',
                            'Octubre': '10',
                            'Noviembre': '11',
                            'Diciembre': '12'})
arroz = arroz.reset_index()
arroz = pd.melt(arroz, id_vars=['Mes'],var_name='Year',value_name='Price')
arroz = arroz.rename(columns={'Mes': 'Month'})
arroz['Year'] = arroz['Year'].astype(int)
arroz['Month'] = arroz['Month'].astype(int)
arroz = arroz.dropna()

#pd.to_datetime((df.Y*10000+df.M*100+df.D).apply(str),format='%Y%m%d')
arroz['Date']=pd.to_datetime((arroz.Year*10000+arroz.Month*100+1).apply(str),format='%Y%m%d')

arroz=arroz[['Date','Price']]
arroz['Price']=arroz['Price']/1000

arroz = arroz.set_index('Date')


# In[ ]:


arroz.head(12)


# In[ ]:


arroz.tail(8)


# In[ ]:


#Timeseries and Distribuition Plot

fig = plt.figure()
sns.set()

f, axs = plt.subplots(2,2,figsize=(15,7))

plt.subplot(121)
sns.lineplot(x=arroz.index, 
             y="Price", 
             data=arroz)
plt.title('Fedearroz Rice Price')
plt.ylabel('Price ($)')

plt.subplot(122)
sns.distplot(arroz.Price,
                    kde=False,
                    color="b")
plt.title('Fedearroz Rice Price Distribution')
plt.ylabel('Price Frequency')

plt.suptitle('Fedearroz Rice Price Analysis', fontsize=16)
plt.show()


# In[ ]:


arroz2018 = arroz['2018']
arroz2018.plot()


# In[ ]:


quarterly = arroz['Price'].resample('Q')
quarterly_mean = quarterly.mean()


# In[ ]:


from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(arroz['Price'], model='additive')
fig = decomposition.plot()
plt.show()


# In[ ]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
            


# In[ ]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(arroz['Price'],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

            


# In[ ]:


mod = sm.tsa.statespace.SARIMAX(arroz['Price'],
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])


# In[ ]:


results.plot_diagnostics(figsize=(16, 8))
plt.show()


# In[ ]:


pred = results.get_prediction(start=pd.to_datetime(arroz.index[len(arroz)-12-1]), dynamic=False)
pred_ci = pred.conf_int()

ax = arroz.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Fedearroz Price')
plt.legend()

plt.show()


# In[ ]:


#Producing and visualizing forecasts

pred_uc = results.get_forecast(steps=12)
pred_ci = pred_uc.conf_int()

ax = arroz.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Fedearroz Price')

plt.legend()
plt.show()


# In[ ]:


Arroz_Forecast = pd.DataFrame(pred_uc.predicted_mean)
Arroz_Forecast.columns = ['Price']
Arroz_Forecast


# In[ ]:




