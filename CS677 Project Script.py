#!/usr/bin/env python
# coding: utf-8

# # Visual analysis with model prediction of factors affecting bitcoin market status and investment suggestion 
# # Course: CS677
# # Author: Pengfei Ma
# 

# ## Introduction

#   Global economics has experienced strong increasing over 10 years. Especially in US stock market, investors successfully raised the volume of market twice. The market created the number of millionaires over a decade; however, the global pandemic strongly punched the economics. As a result, the stock market fused several times in a very short period. With the continuous implementation of the Fed’s quantitative easing policy, the composite is not only backing to increase, but also created a record high. Among millions of products, several products became the super star, such as Tesla, Zoom, and GameStop. But compared with Bitcoin, these products look not that shiny. Bitcoin, as a virtual digital encryption currency, once became the gold of investors this year. The value of Bitcoin once exceeded 60,000 US dollars, which is more than 20 times more than 5 years ago. However, according to Bitcoin's own characteristics and algorithms, the number of Bitcoins will decrease, and the time and cost of mining a Bitcoin will increase exponentially. So, does the price of bitcoin have anything to do with the current mining cost of a bitcoin and the number of bitcoins in the market? Therefore, this project will focus on the historical data of Bitcoin in the past 10 years to analyze that What are the most important factors influencing Bitcoin price.

# ## I - Project Map

# In[164]:


from PIL import Image


# In[165]:


project_map = Image.open('project map.png')


# In[166]:


project_map


# ## II - Data Preparation
# ## 1. Data cleaning and collecting
# 
# ### Websites: Yahoo Finance, CoinMarketCap
# ### Datasets: BTC-USD_daily, BTC-USD_weekly, BTC-USD_weekly_detailed, VIX_daily, MarketCap_daily
# ### Values: AR, BR, MC, VMR

# In[4]:


from pandas_datareader import data as web
import os
import math
import numpy as np 
import pandas as pd


# ### Datasets
# #### > BTC-USD_daily

# In[5]:


# collect daliy info for BTC-USD from 2016-01-01 to 2021-04-23

def get_stock(ticker, start_date, end_date, s_window, l_window):
    try:
        df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
        df['Return'] = df['Adj Close'].pct_change()
        df['Return'].fillna(0, inplace = True)
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year 
        df['Day'] = df['Date'].dt.day
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            df[col] = df[col].round(2)
        df['Weekday'] = df['Date'].dt.weekday_name  
#        df['Weekday'] = df['Date'].dt.day_name       
        df['Week_Number'] = df['Date'].dt.strftime('%U')
        df['Year_Week'] = df['Date'].dt.strftime('%Y-%U')       
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday', 
                    'Week_Number', 'Year_Week', 'Open', 
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return']
        num_lines = len(df)
        df = df[col_list]
        print('read ', num_lines, ' lines of data for ticker: ' , ticker)
        return df
    except Exception as error:
        print(error)
        return None

try:
    ticker='BTC-USD'
    input_dir = r'/Users/mapengfei/Desktop/BU CS/Spring 2021/CS677/project'
    output_file = os.path.join(input_dir, ticker + '_daily.csv')
    df = get_stock(ticker, start_date='2016-01-01', end_date='2021-04-23', 
               s_window=14, l_window=50)
    df.to_csv(output_file, index=False)
    print('wrote ' + str(len(df)) + ' lines to file: ' + output_file)
except Exception as e:
    print(e)
    print('failed to get Yahoo stock data for ticker: ', ticker)


# In[5]:


BTC_daily = pd.read_csv('BTC-USD_daily.csv')


# In[6]:


BTC_daily


# #### >> BTC-USD_weekly & BTC-USD_weekly_detailed

# In[8]:


# collect weekly info for BTC-USD from 2016-01-01 to 2021-04-23

ticker='BTC-USD'
input_dir = r'/Users/mapengfei/Desktop/BU CS/Spring 2021/CS677/project'
output_file = os.path.join(input_dir, ticker + '_weekly_return_volatility.csv')

try:
    df = web.get_data_yahoo(ticker, start='2016-01-01',end='2021-04-23')
    df['Return'] = df['Adj Close'].pct_change()
    df['Return'].fillna(0, inplace = True)
    df['Return'] = 100.0 * df['Return']
    df['Return'] = df['Return'].round(3)        
    df['Date'] = df.index
    df['Date'] = pd.to_datetime(df['Date'])
    df['Week_Number'] = df['Date'].dt.strftime('%U')
    df['Year'] = df['Date'].dt.year 
    df_2 = df[['Year', 'Week_Number', 'Return']]
    df_2.index = range(len(df))
    df_grouped = df_2.groupby(['Year', 'Week_Number'])['Return'].agg([np.mean, np.std])
    df_grouped.reset_index(['Year', 'Week_Number'], inplace=True)
    df_grouped.rename(columns={'mean': 'mean_return', 'std':'volatility'}, inplace=True)
    df_grouped.fillna(0, inplace=True)
    df_grouped.to_csv(output_file, index=False)

#    df_grouped_2 = df_grouped.fillna(0)
#    df_grouped_2.to_csv(output_file, index=False)
    
except Exception as e:
    print(e)


output_file = os.path.join(input_dir, ticker + '_weekly_return_volatility_detailed.csv')
combined_df = df.merge(df_grouped, on=['Year', 'Week_Number'], how = 'inner')
combined_df.to_csv(output_file, index=False)
print("wrote ", len(combined_df), " file to ", output_file)


# In[7]:


BTC_weekly = pd.read_csv('BTC-USD_weekly.csv')
BTC_weekly_detail = pd.read_csv('BTC-USD_weekly_detailed.csv')


# In[8]:


BTC_weekly


# In[9]:


BTC_weekly_detail


# #### >>> VIX index

# In[90]:


# collect daliy info for VIX index from 2016-01-01 to 2021-04-23

def get_stock(ticker, start_date, end_date, s_window, l_window):
    try:
        df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
        df['Return'] = df['Adj Close'].pct_change()
        df['Return'].fillna(0, inplace = True)
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year 
        df['Day'] = df['Date'].dt.day
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            df[col] = df[col].round(2)
        df['Weekday'] = df['Date'].dt.weekday_name  
#        df['Weekday'] = df['Date'].dt.day_name       
        df['Week_Number'] = df['Date'].dt.strftime('%U')
        df['Year_Week'] = df['Date'].dt.strftime('%Y-%U')        
        col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday', 
                    'Week_Number', 'Year_Week', 'Open', 
                    'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Return']
        num_lines = len(df)
        df = df[col_list]
        print('read ', num_lines, ' lines of data for ticker: ' , ticker)
        return df
    except Exception as error:
        print(error)
        return None

try:
    ticker='^VIX'
    input_dir = r'/Users/mapengfei/Desktop/BU CS/Spring 2021/CS677/project'
    output_file = os.path.join(input_dir, ticker + '_daily.csv')
    df = get_stock(ticker, start_date='2015-12-31', end_date='2021-04-23', 
               s_window=14, l_window=50)
    df.to_csv(output_file, index=False)
    print('wrote ' + str(len(df)) + ' lines to file: ' + output_file)
except Exception as e:
    print(e)
    print('failed to get Yahoo stock data for ticker: ', ticker)


# In[11]:


vix = pd.read_csv('^VIX_daily.csv')


# In[12]:


vix


# Since only 1336 rows in vix, in order to fit the same length to BTC-USD_daily. The missing items are Saturday and Sunday because Bitcoin trading is 7/24; however, the VIX index is not followed, it is only computed 5/12. So I am going to repeat the Friday data to fill Saturday and Sunday. And for other missing dates, the data will be filled by the data of one day before.

# In[13]:


new = pd.merge(BTC_daily, vix, how="left", on=["Date"])


# In[14]:


new = new.fillna(0)


# In[15]:


vix2 = new[['Date', 'Adj Close_y']]


# In[16]:


vix2


# Filled out 2016-01-01, 2016-01-02, 2016-01-03 by the vix index of 2015-12-31

# In[17]:


vix2 = vix2.replace([0, 1, 2], [18.21, 18.21, 18.21])


# In[18]:


vix2 = vix2.rename(columns={"Adj Close_y": "Vix Index"})


# In[19]:


vix2


# #### >>>> Market Cap from CoinMarketCap

# In[20]:


marketcap = pd.read_csv('BitcoinMarketCap.csv')


# In[21]:


marketcap


# In[22]:


market_cap = []

for number in marketcap['Market Cap']:
    number = number.replace("$", '')
    number = number.replace(",", '')
    market_cap.append(int(number))


# In[23]:


# since the order is from present to the past, so we need to reverse the order. And pop out the last element
# because the last element is Apr.23, this date is not contained in other dataframes.

market_cap.reverse()
market_cap.pop(1)


# ### Values
# #### >>>>> AR and BR computation

# There are a huge element could influence the return of a stock. From the term of Market Sentiment in Finance, the market use AR and BR to scale the market sentiment which is also called Emotion Index for further analysis. 
# 
# AR: willingness indicator 
# BR: are indicators of the technology to analyze the historical stock price \
# I am going to compute the AR and BR as another two influence element to build the model. 
# 
# AR = $\frac{\sum_{n=1}^{N} H-O}{\sum_{n=1}^{N} O-L } * 100$ 
# 
# BR = $\frac{\sum_{n=1}^{N} H-YC}{\sum_{n=1}^{N} YC-L } * 100$ 
# 
# O is Open price 
# 
# H is the highest price in a day 
# 
# L is the lowest price in a day 
# 
# YC is Yesterday close price 
# 
# N is the number of trading dates 

# In[24]:


AR = []
BR = []

# The Denominator is possible to equal to 0, that will result in infinity of AR and BR. So, replace inf to 10,000

for i in range(len(BTC_daily)):
    
    if i == 0:
        
        upper_ar = BTC_daily["High"][i]-BTC_daily["Open"][i]
        lower_ar = BTC_daily["Open"][i]-BTC_daily["Low"][i]
        
        upper_br = BTC_daily["High"][i]-BTC_daily["Close"][i]
        lower_br = BTC_daily["Close"][i]-BTC_daily["Low"][i]
        
        if lower_ar == 0:
            upper_ar = 2000
            lower_ar = 1
        
        if lower_br == 0:
            upper_br = 2000
            lower_br = 1
            
        ar = upper_ar/lower_ar * 100
        br = upper_br/lower_br * 100
        
        AR.append(ar)
        BR.append(br)
    
    else:
        
        upper_ar = BTC_daily["High"][i]-BTC_daily["Open"][i]
        lower_ar = BTC_daily["Open"][i]-BTC_daily["Low"][i]
        
        upper_br = BTC_daily["High"][i]-BTC_daily["Close"][i]
        lower_br = BTC_daily["Close"][i-1]-BTC_daily["Low"][i-1]
                  
        if lower_ar == 0:
            upper_ar = 2000
            lower_ar = 1
        
        if lower_br == 0:
            upper_br = 2000
            lower_br = 1
            
#         print(upper_br)
#         print(lower_br)
        
#         print(upper_br/lower_br)
#         print(upper_br/lower_br * 100)
        
        ar = upper_ar/lower_ar * 100
        br = upper_br/lower_br * 100
        
        
        AR.append(ar)
        BR.append(br)


# #### >>>>>> Volume / MarketCap ratio (VMR)

# In[25]:


VMR = []

for i in range(len(BTC_daily)):
    vmr = BTC_daily['Volume'][i] / market_cap[i]
    
    VMR.append(vmr)


# ## 2. Create the dataframe that only contains necessary elements

# In[26]:


data = {'Date': BTC_daily['Date'], "Price":BTC_daily['Adj Close'], 'AR': AR, 'BR': BR, 'Vol':BTC_daily['Volume'], 
        'VIX':vix2['Vix Index'], 'MC': market_cap, 'VMR':VMR}
BTC = pd.DataFrame(data=data)


# In[27]:


BTC


# ## III - Overview of BTC

# ### 1. Line plot of Bitcoin natural number variables: Price, Volume, Market Cap

# In[28]:


import plotly.offline as py
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
pyo.init_notebook_mode()


# In[29]:


trace0 = go.Candlestick(x=BTC_daily['Date'],
                                     open=BTC_daily['Open'],
                                     high=BTC_daily['High'],
                                     low=BTC_daily['Low'],
                                     close=BTC_daily['Close'])

trace1 = go.Scatter(x= BTC['Date'], y=BTC['Price'],
                    mode='lines',marker_color='blue',name='Price')

data = [trace0, trace1]

layout = go.Layout(title='Bitcoin price envolution overview', xaxis_title="Date",yaxis_title="USD")

fig = go.Figure(data=data, layout = layout)

fig.update_layout(xaxis=dict(
    tickvals = ['2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01','2016-07-01',
               '2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01',
               '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01', '2017-06-01','2017-07-01',
               '2017-08-01','2017-09-01','2017-10-01','2017-11-01','2017-12-01',
               '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01','2018-07-01',
               '2018-08-01','2018-09-01','2018-10-01','2018-11-01','2018-12-01',
               '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01','2019-07-01',
               '2019-08-01','2019-09-01','2019-10-01','2019-11-01','2019-12-01',
               '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01','2020-07-01',
               '2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01',
               '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01'],
    ticktext = ["Jan 2016", "Feb 2016", "Mar 2016", "Apr 2016", "May 2016", "Jun 2016","Jul 2016", "Aug 2016",
               "Sep 2016", "Oct 2016", "Nov 2016", "Dec 2016",
               "Jan 2017", "Feb 2017", "Mar 2017", "Apr 2017", "May 2017", "Jun 2017","Jul 2017", "Aug 2017",
               "Sep 2017", "Oct 2017", "Nov 2017", "Dec 2017",
               "Jan 2018", "Feb 2018", "Mar 2018", "Apr 2018", "May 2018", "Jun 2018","Jul 2018", "Aug 2018",
               "Sep 2018", "Oct 2018", "Nov 2018", "Dec 2018",
               "Jan 2019", "Feb 2019", "Mar 2019", "Apr 2019", "May 2019", "Jun 2019","Jul 2019", "Aug 2019",
               "Sep 2019", "Oct 2019", "Nov 2019", "Dec 2019",
               "Jan 2020", "Feb 2020", "Mar 2020", "Apr 2020", "May 2020", "Jun 2020","Jul 2020", "Aug 2020",
               "Sep 2020", "Oct 2020", "Nov 2020", "Dec 2020",
               "Jan 2021", "Feb 2021", "Mar 2021", "Apr 2021"]))

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.2,
            showactive=True,
            buttons=list(
                [
                    dict(
                        label="Candlestick",
                        method="update",
                        args=[
                            {"visible": [True, False]},
                            {"yaxis.title.text": "USD",},
                        ],
                    ),
                    dict(
                        label="Line",
                        method="update",
                        args=[
                            {"visible": [False, True]},
                            {"yaxis.title.text": "USD"},
                        ],
                    ),
                ]
            ),
        )
    ]
)

pyo.plot(fig, filename = 'Bitcoin price envolution overview.html')

fig.show()


# In[30]:


x_values = BTC['Date']
y_values = BTC['Price']

trace0 = go.Scatter(x= x_values, y=y_values,
                    mode='lines',name='Bitcoin price')
trace1 = go.Scatter(x= x_values, y=BTC['Vol'],
                    mode='lines',marker_color='orange',name='Vol')
trace2 = go.Scatter(x= x_values, y=BTC['MC'],
                    mode='lines',marker_color='green',name='MC')

data = [trace0, trace1, trace2]

layout = go.Layout(title='Price, Volume, and Market Cap envolution', 
                   xaxis_title="Date",yaxis_title="USD", hovermode='x')

fig = go.Figure(data=data, layout=layout)

fig.update_layout(xaxis=dict(
    tickvals = ['2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01','2016-07-01',
               '2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01',
               '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01', '2017-06-01','2017-07-01',
               '2017-08-01','2017-09-01','2017-10-01','2017-11-01','2017-12-01',
               '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01','2018-07-01',
               '2018-08-01','2018-09-01','2018-10-01','2018-11-01','2018-12-01',
               '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01','2019-07-01',
               '2019-08-01','2019-09-01','2019-10-01','2019-11-01','2019-12-01',
               '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01','2020-07-01',
               '2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01',
               '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01'],
    ticktext = ["Jan 2016", "Feb 2016", "Mar 2016", "Apr 2016", "May 2016", "Jun 2016","Jul 2016", "Aug 2016",
               "Sep 2016", "Oct 2016", "Nov 2016", "Dec 2016",
               "Jan 2017", "Feb 2017", "Mar 2017", "Apr 2017", "May 2017", "Jun 2017","Jul 2017", "Aug 2017",
               "Sep 2017", "Oct 2017", "Nov 2017", "Dec 2017",
               "Jan 2018", "Feb 2018", "Mar 2018", "Apr 2018", "May 2018", "Jun 2018","Jul 2018", "Aug 2018",
               "Sep 2018", "Oct 2018", "Nov 2018", "Dec 2018",
               "Jan 2019", "Feb 2019", "Mar 2019", "Apr 2019", "May 2019", "Jun 2019","Jul 2019", "Aug 2019",
               "Sep 2019", "Oct 2019", "Nov 2019", "Dec 2019",
               "Jan 2020", "Feb 2020", "Mar 2020", "Apr 2020", "May 2020", "Jun 2020","Jul 2020", "Aug 2020",
               "Sep 2020", "Oct 2020", "Nov 2020", "Dec 2020",
               "Jan 2021", "Feb 2021", "Mar 2021", "Apr 2021"]))

fig.update_layout(legend_title_text='Natural number variables')

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.2,
            showactive=True,
            buttons=list(
                [
                    dict(
                        label="Price",
                        method="update",
                        args=[
                            {"visible": [True, False, False]},
                            {"yaxis.title.text": "Price in USD",},
                        ],
                    ),
                    dict(
                        label="Vol",
                        method="update",
                        args=[
                            {"visible": [False, True, False]},
                            {"yaxis.title.text": "Volume in USD"},
                        ],
                    ),
                    dict(
                        label="MC",
                        method="update",
                        args=[
                            {"visible": [False, False, True]},
                            {"yaxis.title.text": "Market Cap in USD"},
                        ],
                    ),
                ]
            ),
        )
    ]
)

pyo.plot(fig, filename = 'Price, Volume, and Market Cap envolution.html')

fig.show()


# In[33]:


fig = make_subplots(rows=1, cols=3, subplot_titles=('Price','Market Cap', 'Volume'))

fig.add_trace(
    go.Scatter(x= x_values, y=BTC['Price'],
                    mode='lines',marker_color='blue',name='Bitcoin price'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x= x_values, y=BTC['MC'],
                    mode='lines',marker_color='green',name='MC'),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x= x_values, y=BTC['Vol'],
                    mode='lines',marker_color='orange',name='Vol'),
    row=1, col=3
)

fig.update_layout(legend_title_text='Natural number variables',
                  title_text="Comparison of Price, Market Cap, and Volume")

pyo.plot(fig, filename = 'Comparison of Price, Market Cap, and Volume.html')

fig.show()


# ### 2. Line chart for index variables: AR, BR, VIX, VMR

# In[31]:


x_values = BTC['Date']
y_values = BTC['Price']

trace0 = go.Scatter(x= x_values, y=BTC['AR'],
                    mode='lines',marker_color='blue',name='AR')
trace1 = go.Scatter(x= x_values, y=BTC['BR'],
                    mode='lines',marker_color='red',name='BR')

trace2 = go.Scatter(x= x_values, y=BTC['VIX'],
                    mode='lines',marker_color='purple',name='VIX')

trace3 = go.Scatter(x= x_values, y=BTC['VMR'],
                    mode='lines',marker_color='pink',name='VMR')

data = [trace0, trace1, trace2, trace3]

layout = go.Layout(title='Bitcoin Market reflection index', xaxis_title="Date",yaxis_title="Index", hovermode='x')

fig = go.Figure(data=data, layout=layout)

fig.update_layout(xaxis=dict(
    tickvals = ['2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01','2016-07-01',
               '2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01',
               '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01', '2017-06-01','2017-07-01',
               '2017-08-01','2017-09-01','2017-10-01','2017-11-01','2017-12-01',
               '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01','2018-07-01',
               '2018-08-01','2018-09-01','2018-10-01','2018-11-01','2018-12-01',
               '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01','2019-07-01',
               '2019-08-01','2019-09-01','2019-10-01','2019-11-01','2019-12-01',
               '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01','2020-07-01',
               '2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01',
               '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01'],
    ticktext = ["Jan 2016", "Feb 2016", "Mar 2016", "Apr 2016", "May 2016", "Jun 2016","Jul 2016", "Aug 2016",
               "Sep 2016", "Oct 2016", "Nov 2016", "Dec 2016",
               "Jan 2017", "Feb 2017", "Mar 2017", "Apr 2017", "May 2017", "Jun 2017","Jul 2017", "Aug 2017",
               "Sep 2017", "Oct 2017", "Nov 2017", "Dec 2017",
               "Jan 2018", "Feb 2018", "Mar 2018", "Apr 2018", "May 2018", "Jun 2018","Jul 2018", "Aug 2018",
               "Sep 2018", "Oct 2018", "Nov 2018", "Dec 2018",
               "Jan 2019", "Feb 2019", "Mar 2019", "Apr 2019", "May 2019", "Jun 2019","Jul 2019", "Aug 2019",
               "Sep 2019", "Oct 2019", "Nov 2019", "Dec 2019",
               "Jan 2020", "Feb 2020", "Mar 2020", "Apr 2020", "May 2020", "Jun 2020","Jul 2020", "Aug 2020",
               "Sep 2020", "Oct 2020", "Nov 2020", "Dec 2020",
               "Jan 2021", "Feb 2021", "Mar 2021", "Apr 2021"]))

fig.update_layout(legend_title_text='Marketing reflection index')

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.2,
            showactive=True,
            buttons=list(
                [
                    dict(
                        label="AR",
                        method="update",
                        args=[
                            {"visible": [True, False, False, False]},
                            {"yaxis.title.text": "AR index",},
                        ],
                    ),
                    dict(
                        label="BR",
                        method="update",
                        args=[
                            {"visible": [False, True, False, False]},
                            {"yaxis.title.text": "BR index"},
                        ],
                    ),
                    dict(
                        label="VIX",
                        method="update",
                        args=[
                            {"visible": [False, False, True, False]},
                            {"yaxis.title.text": "VIX index"},
                        ],
                    ),
                    dict(
                        label="VMR",
                        method="update",
                        args=[
                            {"visible": [False, False, False, True]},
                            {"yaxis.title.text": "VMR"}
                        ],
                    ),
                ]
            ),
        )
    ]
)

pyo.plot(fig, filename = 'Bitcoin Market reflection index.html')

fig.show()


# In[32]:


fig = make_subplots(rows=2, cols=2, subplot_titles=('AR','BR', 'VIX', 'VMR'))

fig.add_trace(
    go.Scatter(x= x_values, y=BTC['AR'],
                    mode='lines',marker_color='blue',name='Bitcoin price'),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(x= x_values, y=BTC['BR'],
                    mode='lines',marker_color='red',name='MC'),
    row=1, col=2
)

fig.add_trace(
    go.Scatter(x= x_values, y=BTC['VIX'],
                    mode='lines',marker_color='purple',name='Vol'),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x= x_values, y=BTC['VMR'],
                    mode='lines',marker_color='pink',name='VMR'),
    row=2, col=2
)

fig.update_layout(legend_title_text='Marketing reflection index',
                  title_text="Comparison of AR,BR,VIX, and VMR")

pyo.plot(fig, filename = 'Comparison of AR,BR,VIX, and VMR.html')

fig.show()


# ### 3. Corresponding correlation matrix for BTC

# In[33]:


fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
M0 = sns.heatmap(BTC.corr(), annot = True).set_title('BTC corresponding correlation')

plt.savefig("BTC correlation martrix heatmap.png")


# ## IV - Label Assignment
# ### Rules
# There are 6 important factors in the dataset, the labels will be assigned based on marketing reflection index

# #### AR: 
# 1. AR value of 100 as the central zone, the ± 20, that AR value fluctuations between 80-120, the consolidation market, the stock price trend is relatively stable, does not appear violent fluctuations.
# 
# 2. AR value is high, said the market is active, strong, too high, said the stock into the high price, should choose the time to exit, AR value of the height of no specific standard, under normal circumstances, AR value rose to 150 or more, the stock may return at any time File down.
# 
# 3. AR value is lower that the popularity of recession, the need to enrich, too low implies that the stock may be stacked into a trough, can be considered opportunistic intervention, the general AR value fell below 50, the stock may rebound at any time rise.
# 
# #### BR:
# 1. BR value of the wave-sensitive than the AR value, when the BR value fluctuates between 70-150, the consolidation market, should remain on the sidelines.
# 
# 2. BR value is higher than 300 or more, the stock may fall back at any time, should choose the time to sell; BR value below 50 the following time may rebound at any time to rise, should choose the time to buy.
# 
# #### Features
# 1. AR trend and stock price are in the same direction, AR rises and the market also rises; AR falls and the market also falls.
# 
# 2. The AR indicator cannot show the strength and weakness of the stock price gap, and can only show the rangeof the stock price fluctuations on the day.
# 
# 3. AR often advances when the head or bottom is formed.
# 
# 4. From the BR indicator, we can see the formation of stock's upperor lowerprice zones.
# 
# 5. BR should be used in conjunction with AR to be effective.
# 
# 6. When using the BR indicator for individual stocks, when they encounter dividends, the closing price of the previous day needs to adjust the weight.
# 
# 7. When AR and BR drop sharply, it means that the stock price is close to the peak, it's sign of profit taking session; when BR rises sharply, but AR falls slightly or consolidates, trader should sell the rally; when BR is lower than AR, trader could buy the dip. (Usually BR value is higher than AR value)
# 
# Since Bitcoin has a huge fluctuations, we will determine the bitcoin market status by 50% more than normal standard.
# 
# Source: https://support.futunn.com/en-us/topic166#:~:text(AR)%20and%20willingness%20indicator%20(,the%20market%20The%20degree%20of

# ## Multiple Linear Regression

# In[39]:


from sklearn import linear_model


# In[40]:


X = BTC[['AR', 'BR', 'Vol','VIX', 'MC', 'VMR']]
y = BTC['Price']

regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.coef_)

weights = regr.coef_


# ##### The multiple linear regression equation of all factors respected to the price
# $ y =3.5442707e^{-4} \times {AR} - 9.4322964e^{-4}\times BR -7.5212122e^{-9}\times Vol -9.45219257\times VIX + 5.41147582e^{-8}\times MC+1464.05427\times{VMR}$
# 
# To scale it I am going to input all values of each factor and apply natural log to computed price as the scale index.

# ## Residual of multiple linear regression and real price

# In[41]:


p_price = []

for i in range(len(BTC)):
    
    y = (weights[0]*BTC['AR'][i] + weights[1]*BTC['BR'][i] + weights[2]*BTC['Vol'][i] + weights[3]*BTC['VIX'][i] + 
        weights[4]*BTC['MC'][i] + weights[5]*BTC['VMR'][i])

    p_price.append(abs(y))    


# In[42]:


x_values = BTC['Date']
y_values = BTC['Price']

trace0 = go.Scatter(x= x_values, y=y_values,
                    mode='lines',name='Real price')
trace1 = go.Scatter(x= x_values, y=p_price,
                    mode='lines',marker_color='orange',name='MLR price')

data = [trace0, trace1]

layout = go.Layout(title='Price, and MLR price envolution', xaxis_title="Date",yaxis_title="USD", hovermode='x')

fig = go.Figure(data=data, layout=layout)

fig.update_layout(xaxis=dict(
    tickvals = ['2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01','2016-07-01',
               '2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01',
               '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01', '2017-06-01','2017-07-01',
               '2017-08-01','2017-09-01','2017-10-01','2017-11-01','2017-12-01',
               '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01','2018-07-01',
               '2018-08-01','2018-09-01','2018-10-01','2018-11-01','2018-12-01',
               '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01','2019-07-01',
               '2019-08-01','2019-09-01','2019-10-01','2019-11-01','2019-12-01',
               '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01','2020-07-01',
               '2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01',
               '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01'],
    ticktext = ["Jan 2016", "Feb 2016", "Mar 2016", "Apr 2016", "May 2016", "Jun 2016","Jul 2016", "Aug 2016",
               "Sep 2016", "Oct 2016", "Nov 2016", "Dec 2016",
               "Jan 2017", "Feb 2017", "Mar 2017", "Apr 2017", "May 2017", "Jun 2017","Jul 2017", "Aug 2017",
               "Sep 2017", "Oct 2017", "Nov 2017", "Dec 2017",
               "Jan 2018", "Feb 2018", "Mar 2018", "Apr 2018", "May 2018", "Jun 2018","Jul 2018", "Aug 2018",
               "Sep 2018", "Oct 2018", "Nov 2018", "Dec 2018",
               "Jan 2019", "Feb 2019", "Mar 2019", "Apr 2019", "May 2019", "Jun 2019","Jul 2019", "Aug 2019",
               "Sep 2019", "Oct 2019", "Nov 2019", "Dec 2019",
               "Jan 2020", "Feb 2020", "Mar 2020", "Apr 2020", "May 2020", "Jun 2020","Jul 2020", "Aug 2020",
               "Sep 2020", "Oct 2020", "Nov 2020", "Dec 2020",
               "Jan 2021", "Feb 2021", "Mar 2021", "Apr 2021"]))

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.2,
            showactive=True,
            buttons=list(
                [
                    dict(
                        label="Real Price",
                        method="update",
                        args=[
                            {"visible": [True, False]},
                            {"yaxis.title.text": "USD",},
                        ],
                    ),
                    dict(
                        label="MLR Price",
                        method="update",
                        args=[
                            {"visible": [False, True]},
                            {"yaxis.title.text": "USD"},
                        ],
                    ),
                ]
            ),
        )
    ]
)

pyo.plot(fig, filename = 'Price, and MLR price envolution.html')

fig.show()


# In[43]:


residual = []

for i in range(len(BTC)):
    
    r = BTC["Price"][i] - p_price[i]
    
    residual.append(r)

np.mean(residual)


# The average residual of real price and MLR price is $\textbf{399.1540953443855}$ Based on the huge price of Bitcoin, this number of residual is acceptable.

# In[44]:


trace = go.Box(y=BTC['VIX'], name='VIX')

data = [trace]

layout = go.Layout(title='Boxplot of VIX', xaxis_title="Date",yaxis_title="index", hovermode='x')

fig = go.Figure(data=data, layout=layout)

fig.update_layout(xaxis=dict(
    tickvals = ['2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01','2016-07-01',
               '2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01',
               '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01', '2017-06-01','2017-07-01',
               '2017-08-01','2017-09-01','2017-10-01','2017-11-01','2017-12-01',
               '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01','2018-07-01',
               '2018-08-01','2018-09-01','2018-10-01','2018-11-01','2018-12-01',
               '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01','2019-07-01',
               '2019-08-01','2019-09-01','2019-10-01','2019-11-01','2019-12-01',
               '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01','2020-07-01',
               '2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01',
               '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01'],
    ticktext = ["Jan 2016", "Feb 2016", "Mar 2016", "Apr 2016", "May 2016", "Jun 2016","Jul 2016", "Aug 2016",
               "Sep 2016", "Oct 2016", "Nov 2016", "Dec 2016",
               "Jan 2017", "Feb 2017", "Mar 2017", "Apr 2017", "May 2017", "Jun 2017","Jul 2017", "Aug 2017",
               "Sep 2017", "Oct 2017", "Nov 2017", "Dec 2017",
               "Jan 2018", "Feb 2018", "Mar 2018", "Apr 2018", "May 2018", "Jun 2018","Jul 2018", "Aug 2018",
               "Sep 2018", "Oct 2018", "Nov 2018", "Dec 2018",
               "Jan 2019", "Feb 2019", "Mar 2019", "Apr 2019", "May 2019", "Jun 2019","Jul 2019", "Aug 2019",
               "Sep 2019", "Oct 2019", "Nov 2019", "Dec 2019",
               "Jan 2020", "Feb 2020", "Mar 2020", "Apr 2020", "May 2020", "Jun 2020","Jul 2020", "Aug 2020",
               "Sep 2020", "Oct 2020", "Nov 2020", "Dec 2020",
               "Jan 2021", "Feb 2021", "Mar 2021", "Apr 2021"]))

fig.show()


# ##### Market Status would be assigned by the following rules
# 
# $\color{red}{\text{Warning}}{: AR > 200, BR > 300}$
# 
# $\color{green}{\text{Fluctuation Up}}{: AR \in (120, 200], BR \in (150,300]}$
# 
# $\color{blue}{\text{Stable}}{: AR \in [80, 120], BR \in [70, 150]}$
# 
# $\color{grey}{\text{Fluctuation Down}}{: AR \in [50,80), BR \in [50,70)}$
# 
# $\color{orange}{\text{Downturn}}{: AR < 50, BR < 50}$
# 
# ##### if AR, BR do not fit above conditions, then consider about VIX:
# 
# $\color{red}{\text{Warning}}{: VIX > 50}$
# 
# $\color{green}{\text{Fluctuation Up}}{: VIX \in (13.37, 18.21]}$
# 
# $\color{blue}{\text{Stable}}{: VIX < 13.37}$
# 
# $\color{grey}{\text{Fluctuation Down}}{: VIX \in (25.47, 50]}$
# 
# $\color{orange}{\text{Downturn}}{: VIX \in (18.21, 25.47]}$
# 
# Stable market must have the lowest VIX
# 
# Increasing market will have the second lowerest VIX
# 
# Downturn has the middle range of VIX
# 
# Decreasing market has second highest VIX
# 
# Warned market must have the highest VIX
# 
# ##### Investment suggestion
# 
# $\color{red}{\text{Sell All}}:$ Warning status
# 
# $\color{darkgreen}{\text{75% Hedge}}:$ Flucuation Up status
# 
# $\color{grey}{\text{25% Hedge}}:$ Fluctuation Down
# 
# $\color{lightGreen}{\text{All in}}:$ Stable, Downturn

# In[45]:


MS = []
invest = []

for i in range(len(BTC)):
    
    if BTC['AR'][i] > 200 and BTC['BR'][i] > 300:
        MS.append("Warning")
        invest.append('Sell All')
        print(i, 'Append success')
        
    elif 120 < BTC['AR'][i] <= 200 and 150 < BTC['BR'][i] <= 300:
        MS.append("Fluctuation Up")
        invest.append('75% Hedge')
        print(i, 'Append success')
        
    elif 80 < BTC['AR'][i] <= 120 and 70 < BTC['BR'][i] <= 150:
        MS.append("Stable")
        invest.append('All in')
        print(i, 'Append success')
        
    elif 50 <= BTC['AR'][i] < 80 and 50 <= BTC['BR'][i] < 70:
        MS.append("Fluctuation Down")
        invest.append('25% Hedge')
        print(i, 'Append success')
        
    elif BTC['AR'][i] < 50 and BTC['BR'][i] < 50:
        MS.append("Downturn")
        invest.append('All in')
        print(i, 'Append success')
        
    else:        
        if BTC['VIX'][i] > 50:
            MS.append("Warning")
            invest.append('Sell All')
            print(i, 'Append success')
            
        elif 13.37 < BTC['VIX'][i] <= 18.21:
            MS.append("Fluctuation Up")
            invest.append('75% Hedge')
            print(i, 'Append success')
            
        elif BTC['VIX'][i] <= 13.37:
            MS.append("Stable")
            invest.append('All in')
            print(i, 'Append success')
            
        elif 25.47 < BTC['VIX'][i] <= 50:
            MS.append("Fluctuation Down")
            invest.append('25% Hedge')
            print(i, 'Append success')
        
        elif 18.21 < BTC['VIX'][i] <= 25.47:
            MS.append("Downturn")
            invest.append('All in')
            print(i, 'Append success')
        else:
            print(BTC['AR'][i],BTC['BR'][i],BTC['VIX'][i])


# In[46]:


BTC.insert(8,'Market Status', MS)
BTC.insert(9,'Invest Suggestion', invest)


# In[47]:


BTC


# # V. Models
# ## Classifiers: kNN, Logistic Regression, Naive Bayes, Gaussian SVM
# ## Clusters: K-means
# 
# This section will focus on several data science classifier algorithms to train the historical data and predict the data in 2021. Computed the accuracy of each classifiers for the time period of Jan.2021 to Apr.2021. Then, using the model to predict the whole 2021. 
# 
# ##### Market Status
# 
# $\color{red}{\text{Warning}}$
# 
# $\color{green}{\text{Fluctuation Up}}$
# 
# $\color{blue}{\text{Stable}}$
# 
# $\color{grey}{\text{Fluctuation Down}}$
# 
# $\color{orange}{\text{Downturn}}$
# 
# The varibles used for all classifiers and cluster are: $\textbf{Price, AR, BR, Vol, VIX, MC, VMR}$
# 
# The label colume used is $\textbf{Market Status}$

# In[48]:


BTC_train = BTC.head(1823)
BTC2021 = BTC.tail(114)


# In[49]:


BTC2021 = BTC2021.reset_index(drop=True)


# In[50]:


BTC_train


# In[51]:


BTC2021


# ## 1. kNN
# ### Finding best k

# In[53]:


from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[54]:


error_rates_knn = []

for k in range(2,50):

    X_train = BTC_train[['Price','AR', 'BR','Vol',"VIX","MC",'VMR']].values
    Y_train = BTC_train[["Market Status"]].values

    X_test = BTC2021[['Price','AR', 'BR','Vol',"VIX","MC",'VMR']].values
    Y_test = BTC2021[["Market Status"]].values

    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train,Y_train)

    prediction = knn_classifier.predict(X_test)
    error_rate = np.mean(prediction != Y_test)
    
    error_rates_knn.append(error_rate)


# In[55]:


trace = go.Scatter(x= np.array(range(2, 50)), y=error_rates_knn,
                    mode='lines',marker_color='orange',name='error rates')

trace1 = go.Scatter(x= np.array(range(2, 50)), mode = 'markers',y=error_rates_knn,marker_color='red',name='k')

data = [trace,trace1]

layout = go.Layout(title='error rates for different k value', xaxis_title="k",yaxis_title="index", hovermode='x')

fig = go.Figure(data=data, layout=layout)

fig.show()


# ##### Since the error rates for different k remains the same, which means the prediction is super stable. Therefore, k = 10 will be randomly selected. for next part.

# ### Model applicaton

# In[56]:


X_train = BTC_train[['Price','AR', 'BR','Vol',"VIX","MC",'VMR']].values
Y_train = BTC_train[["Market Status"]].values

X_test = BTC2021[['Price','AR', 'BR','Vol',"VIX","MC",'VMR']].values
Y_test = BTC2021[["Market Status"]].values

knn_classifier = KNeighborsClassifier(n_neighbors = 10)
knn_classifier.fit(X_train,Y_train)

prediction = knn_classifier.predict(X_test)
correct_rate = np.mean(prediction == Y_test)


# ### Market Status prediction by kNN

# In[57]:


prediction


# In[58]:


correct_rate


# kNN predicted the market status is all in Downturn in 2021 and gained 50.88% correct

# ## 2. Logistic Regression
# ### Model prediction

# In[59]:


from sklearn.linear_model import LogisticRegression


# In[60]:


X_train = BTC_train[['Price','AR', 'BR','Vol',"VIX","MC",'VMR']].values
Y_train = BTC_train[["Market Status"]].values

X_test = BTC2021[['Price','AR', 'BR','Vol',"VIX","MC",'VMR']].values
Y_test = BTC2021[["Market Status"]].values


# In[61]:


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train,Y_train)

prediction = log_reg_classifier.predict(X_test)
correct_rate = np.mean(prediction == Y_test)


# ### Market Status prediction by Logistic Regression

# In[62]:


prediction


# In[63]:


correct_rate


# Logistic regression prediction in 2021 gained 34.857% correct

# ## 3. Naive Bayes
# ### Model prediction

# In[64]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB


# In[65]:


NB_classifier = GaussianNB().fit(X_train,Y_train)
prediction = NB_classifier.predict(X_test)
correct_rate = np.mean(prediction== Y_test)


# ### Market Status prediction by Naive Bayes

# In[66]:


prediction


# In[286]:


correct_rate


# Naive Bayes prediction in 2021 gained 50.446% correct

# ## 4. Gaussian SVM
# ### Model application

# In[67]:


from sklearn import svm
from sklearn.preprocessing import StandardScaler


# In[68]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)


# In[69]:


svm_classifier = svm.SVC(kernel ='linear')
svm_classifier.fit(X_train,Y_train)

prediction = svm_classifier.predict(X_test)
correct_rate = svm_classifier.score(X_test, Y_test)


# ### Market Status prediction by Gaussian SVM

# In[70]:


prediction


# In[71]:


correct_rate


# Logistic regression prediction in 2021 gained 8.772% correct

# ### Summary table of the accuracy of four classifiers

# In[72]:


data = {'Classifier':['kNN', "Logistic Regression", "Naive Bayes","Gaussian SVM"], 
        'Accuracy':['50.877%',"34.857%","50.446%","8.772%"]}

table1 = pd.DataFrame(data=data)


# In[73]:


table1


# ##### As a result, kNN classifier predicted the best accuracy among four classifiers.

# ## 5. Cluster: K-means

# In[74]:


from sklearn.cluster import KMeans


# In[75]:


BTC_ele = BTC[['Price','AR', 'BR','Vol',"VIX","MC",'VMR']].values

inertia_list = []
for k in range(1,15):
    kmeans_classifier = KMeans(n_clusters = k)
    y_kmeans = kmeans_classifier.fit_predict(BTC_ele)
    inertia = kmeans_classifier.inertia_
    inertia_list.append(inertia)


# In[76]:


fig,ax = plt.subplots(1, figsize =(7,5))
plt.plot(range(1, 15),inertia_list,marker ='o',color ='green')
plt.xlabel('number of clusters : k')
plt.ylabel('inertia')
plt.tight_layout()
plt.show()


# The optimal number of clusters is k=2, since at k=2, there is no significant decrease in loss function.

# In[77]:


kmeans_classifier = KMeans(n_clusters=2)
y_kmeans = kmeans_classifier.fit_predict(BTC_ele)
centroids = kmeans_classifier.cluster_centers_


# In[306]:


centroids


# Two centroids were given by kmeans cluster. The Bitcoin price evolution would be separated to three stages by these two centroids.
# 
# Finding the closest two time slot for two centroids as the bound of each stage. 

# In[80]:


for i in range(len(BTC)):
    if 6080<= BTC['Price'][i] <= 6090:
        print(BTC.iloc[i])


# In[81]:


for i in range(len(BTC)):
    if 47474<= BTC['Price'][i] <= 47600:
        print(BTC.iloc[i])


# In[82]:


BTC[BTC["Date"]=='2018-06-21']


# In[83]:


BTC[BTC["Date"]=='2021-02-12']


# In[84]:


BTC_k1 = BTC.head(903)
BTC_k2 = BTC.iloc[903:1865]
BTC_k3 = BTC.tail(1937-1865)


# In[85]:


BTC_k2 = BTC_k2.reset_index(drop=True)
BTC_k3 = BTC_k3.reset_index(drop=True)


# In[87]:


trace0 = go.Scatter(x= BTC_k1['Date'], y=BTC_k1['Price'],
                    mode='lines',marker_color='blue',name='Stage 1')
trace1 = go.Scatter(x= BTC_k2['Date'], y=BTC_k2['Price'],
                    mode='lines',marker_color='red',name='Stage 2')

trace2 = go.Scatter(x= BTC_k3['Date'], y=BTC_k3['Price'],
                    mode='lines',marker_color='orange',name='Stage 3')

trace3 = go.Scatter(x= ["2018-06-21", '2021-02-12'], y=[6085.58160, 47474.7082],
                    mode='markers',marker_color='green',name='K mean centroids',
                    marker=dict(color='LightSkyBlue',size=15),marker_symbol='star')

data = [trace0, trace1, trace2, trace3]

layout = go.Layout(title='Bitcoin price envolution stages by K-means', 
                   xaxis_title="Date",yaxis_title="Index", hovermode='x')

fig = go.Figure(data=data, layout=layout)

fig.update_layout(xaxis=dict(
    tickvals = ['2016-01-01', '2016-02-01', '2016-03-01', '2016-04-01', '2016-05-01', '2016-06-01','2016-07-01',
               '2016-08-01','2016-09-01','2016-10-01','2016-11-01','2016-12-01',
               '2017-01-01', '2017-02-01', '2017-03-01', '2017-04-01', '2017-05-01', '2017-06-01','2017-07-01',
               '2017-08-01','2017-09-01','2017-10-01','2017-11-01','2017-12-01',
               '2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01','2018-07-01',
               '2018-08-01','2018-09-01','2018-10-01','2018-11-01','2018-12-01',
               '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01', '2019-05-01', '2019-06-01','2019-07-01',
               '2019-08-01','2019-09-01','2019-10-01','2019-11-01','2019-12-01',
               '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01', '2020-06-01','2020-07-01',
               '2020-08-01','2020-09-01','2020-10-01','2020-11-01','2020-12-01',
               '2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01','2021-05-01'],
    ticktext = ["Jan 2016", "Feb 2016", "Mar 2016", "Apr 2016", "May 2016", "Jun 2016","Jul 2016", "Aug 2016",
               "Sep 2016", "Oct 2016", "Nov 2016", "Dec 2016",
               "Jan 2017", "Feb 2017", "Mar 2017", "Apr 2017", "May 2017", "Jun 2017","Jul 2017", "Aug 2017",
               "Sep 2017", "Oct 2017", "Nov 2017", "Dec 2017",
               "Jan 2018", "Feb 2018", "Mar 2018", "Apr 2018", "May 2018", "Jun 2018","Jul 2018", "Aug 2018",
               "Sep 2018", "Oct 2018", "Nov 2018", "Dec 2018",
               "Jan 2019", "Feb 2019", "Mar 2019", "Apr 2019", "May 2019", "Jun 2019","Jul 2019", "Aug 2019",
               "Sep 2019", "Oct 2019", "Nov 2019", "Dec 2019",
               "Jan 2020", "Feb 2020", "Mar 2020", "Apr 2020", "May 2020", "Jun 2020","Jul 2020", "Aug 2020",
               "Sep 2020", "Oct 2020", "Nov 2020", "Dec 2020",
               "Jan 2021", "Feb 2021", "Mar 2021", "Apr 2021",'May 2021']))

pyo.plot(fig, filename = 'Bitcoin price envolution stages by K-means.html')

fig.show()


# # VI. Trading Demo
# ### Trading strategy: Real label, By k-NN, by Logistic Regression, by Naive Bayes, By Gaussian SVM
# This section will foucs on stimulating trading demo.
# 
# 
# $\textbf{Real world trading stimulation:}$ Applying real label trading strategy and other strategies by classifiers for the time period Jan.1st 2021 to Apr.23th 2021 
# 
# 
# $\textbf{Train set:}$ BTC from 2016-01-01 to 2020-12-31
# 
# 
# $\textbf{Test set:}$ BTC from 2021-01-01 to 2021-04-23
# 
# 
# The investment suggestion and trading strategy is stimulating the real market as much as possible. So there are four investment suggestion label generated before. Higher scale index means more risk to invest. So if the scale index is too high, it is time to sell all. Conversely, it is time to buy in. The reason why choose 75% hedge for Fluctuation Up is that no one know exactly how the market goes, so keep 25% in the pocket to lower the risk. The market is always surprised investors when they are chilled. On the other hand, keep 75% in the pocket when the label is Fluctuation Down to keep the money safer. Still holding 25% in the market to maintain the probability of the increasing. 
# 
# ##### Investment suggestion
# 
# $\color{red}{\text{Sell All}}$
# 
# $\color{darkgreen}{\text{75% Hedge}}$
# 
# $\color{grey}{\text{25% Hedge}}$
# 
# $\color{lightGreen}{\text{All in}}$
# 
# The varibles used for all classifiers and cluster are: $\textbf{Price, AR, BR, Vol, VIX, MC, VMR}$
# 
# The label colume used is $\textbf{Investment Suggestion}$

# ## 1. Real world trading stimulation
# ### Daily return rate for BTC Jan.1st 2021 to Apr.23rd 2021

# In[88]:


BTC2021


# In[89]:


return_rate = []

for i in range(len(BTC2021)):
    
    if i == 0:
        return_rate.append(0)
    else:
        rate = (BTC2021['Price'][i]-BTC2021['Price'][i-1])/BTC2021['Price'][i-1]
        return_rate.append(rate)


# In[90]:


BTC2021.insert(10, 'Return rate', return_rate)


# In[91]:


BTC2021


# ### > Real suggestion strategy
# 
# Following the assigned investment suggestions 

# In[92]:


cash_bah = 10000
balance_bah = []

for i in range(len(BTC2021)):
    
    if BTC2021['Invest Suggestion'][i] == "Sell All":
        
        balance_bah.append(cash_bah)
        
    elif BTC2021['Invest Suggestion'][i] == "All in":
        
        cash_bah = cash_bah * (1 + BTC2021["Return rate"][i])
        balance_bah.append(cash_bah)
        
    elif BTC2021['Invest Suggestion'][i] == "75% Hedge":
        
        invested = 0.75 * cash_bah
        cash_bah = 0.25*cash_bah + invested*(1 + BTC2021["Return rate"][i])
        balance_bah.append(cash_bah)
        
    elif BTC2021['Invest Suggestion'][i] == "25% Hedge":
        
        invested = 0.25 * cash_bah
        cash_bah = 0.75*cash_bah + invested*(1 + BTC2021["Return rate"][i])
        balance_bah.append(cash_bah)
        
    else:
        print(BTC2021.iloc[i])
        


# In[457]:


cash_bah


# The final return of real suggestion strategy is 16456.99 USD

# ### > kNN label strategy
# #### Finding best k for investment suggestion label

# In[126]:


error_rates_knn = []

for k in range(2,50):

    X_train = BTC_train[['Price','AR', 'BR','Vol',"VIX","MC",'VMR']].values
    Y_train = BTC_train[["Invest Suggestion"]].values

    X_test = BTC2021[['Price','AR', 'BR','Vol',"VIX","MC",'VMR']].values
    Y_test = BTC2021[["Invest Suggestion"]].values

    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train,Y_train)

    prediction = knn_classifier.predict(X_test)
    error_rate = np.mean(prediction != Y_test)
    
    error_rates_knn.append(error_rate)


# In[127]:


trace = go.Scatter(x= np.array(range(2, 50)), y=error_rates_knn,
                    mode='lines',marker_color='orange',name='error rates')

trace1 = go.Scatter(x= np.array(range(2, 50)), mode = 'markers',y=error_rates_knn,marker_color='red',name='k')

data = [trace,trace1]

layout = go.Layout(title='error rates for different k value', xaxis_title="k",yaxis_title="index", hovermode='x')

fig = go.Figure(data=data, layout=layout)

fig.show()


# ##### Since the error rates for different k remains the same, which means the prediction is super stable. Therefore, k = 10 will be randomly selected. for next part.

# In[128]:


X_train = BTC_train[['Price','AR', 'BR','Vol',"VIX","MC",'VMR']].values
Y_train = BTC_train[["Invest Suggestion"]].values

X_test = BTC2021[['Price','AR', 'BR','Vol',"VIX","MC",'VMR']].values
Y_test = BTC2021[["Invest Suggestion"]].values


# In[129]:


knn_classifier = KNeighborsClassifier(n_neighbors = 10)
knn_classifier.fit(X_train,Y_train)

prediction = knn_classifier.predict(X_test)
correct_rate_knn = np.mean(prediction == Y_test)


# In[130]:


prediction


# In[131]:


correct_rate_knn


# The accuracy of kNN for investment suggestion label is 54.385%

# In[132]:


cash_knn = 10000
balance_knn = []

for i in range(len(BTC2021)):
    
    if prediction[i] == "Sell All":
        
        balance_knn.append(cash_knn)
        
    elif prediction[i] == "All in":
        
        cash_knn = cash_knn * (1 + BTC2021["Return rate"][i])
        balance_knn.append(cash_knn)
        
    elif prediction[i] == "75% Hedge":
        
        invested = 0.75 * cash_knn
        cash_knn = 0.25*cash_knn + invested*(1 + BTC2021["Return rate"][i])
        balance_knn.append(cash_knn)
        
    elif prediction[i] == "25% Hedge":
        
        invested = 0.25 * cash_knn
        cash_knn = 0.75*cash_knn + invested*(1 + BTC2021["Return rate"][i])
        balance_knn.append(cash_knn)
        
    else:
        print(BTC2021.iloc[i])
        


# In[133]:


cash_knn


# The final return of kNN predicted investment suggestions is 17039.09 USD

# ### > Logistic Regression

# In[134]:


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train,Y_train)

prediction = log_reg_classifier.predict(X_test)
correct_rate_logr = np.mean(prediction == Y_test)


# In[135]:


prediction


# In[136]:


correct_rate_logr


# The accuracy of logistic regression for investment suggestion label is 49.077%

# In[137]:


cash_logr = 10000
balance_logr = []

for i in range(len(BTC2021)):
    
    if prediction[i] == "Sell All":
        
        balance_logr.append(cash_logr)
        
    elif prediction[i] == "All in":
        
        cash_logr = cash_logr * (1 + BTC2021["Return rate"][i])
        balance_logr.append(cash_logr)
        
    elif prediction[i] == "75% Hedge":
        
        invested = 0.75 * cash_logr
        cash_logr = 0.25*cash_logr + invested*(1 + BTC2021["Return rate"][i])
        balance_logr.append(cash_logr)
        
    elif prediction[i] == "25% Hedge":
        
        invested = 0.25 * cash_logr
        cash_logr = 0.75*cash_logr + invested*(1 + BTC2021["Return rate"][i])
        balance_logr.append(cash_logr)
        
    else:
        print(BTC2021.iloc[i])


# In[138]:


cash_logr


# The final return of logistic regression predicted investment suggestions is 15755.46 USD

# ### > Naive Bayes strategy

# In[139]:


NB_classifier = GaussianNB().fit(X_train,Y_train)
prediction = NB_classifier.predict(X_test)
correct_rate_nb = np.mean(prediction== Y_test)


# In[140]:


prediction


# In[141]:


correct_rate_nb


# The accuracy of Naive Bayes for investment suggestion label is 52.723%

# In[142]:


cash_nb = 10000
balance_nb = []

for i in range(len(BTC2021)):
    
    if prediction[i] == "Sell All":
        
        balance_nb.append(cash_nb)
        
    elif prediction[i] == "All in":
        
        cash_nb = cash_nb * (1 + BTC2021["Return rate"][i])
        balance_nb.append(cash_nb)
        
    elif prediction[i] == "75% Hedge":
        
        invested = 0.75 * cash_nb
        cash_nb = 0.25*cash_nb + invested*(1 + BTC2021["Return rate"][i])
        balance_nb.append(cash_nb)
        
    elif prediction[i] == "25% Hedge":
        
        invested = 0.25 * cash_nb
        cash_nb = 0.75*cash_nb + invested*(1 + BTC2021["Return rate"][i])
        balance_nb.append(cash_nb)
        
    else:
        print(BTC2021.iloc[i])


# In[143]:


cash_nb


# The final return of Naive Bayes predicted investment suggestions is 16768.45 USD

# ### > Gaussian SVM strategy

# In[144]:


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)


# In[145]:


svm_classifier = svm.SVC(kernel ='linear')
svm_classifier.fit(X_train,Y_train)

prediction = svm_classifier.predict(X_test)
correct_rate_svm = svm_classifier.score(X_test, Y_test)


# In[146]:


prediction


# In[147]:


correct_rate_svm


# The accuracy of Gaussian SVM for investment suggestion label is 8.772%

# In[148]:


cash_svm = 10000
balance_svm = []

for i in range(len(BTC2021)):
    
    if prediction[i] == "Sell All":
        
        balance_svm.append(cash_svm)
        
    elif prediction[i] == "All in":
        
        cash_svm = cash_svm * (1 + BTC2021["Return rate"][i])
        balance_svm.append(cash_svm)
        
    elif prediction[i] == "75% Hedge":
        
        invested = 0.75 * cash_svm
        cash_svm = 0.25*cash_svm + invested*(1 + BTC2021["Return rate"][i])
        balance_svm.append(cash_svm)
        
    elif prediction[i] == "25% Hedge":
        
        invested = 0.25 * cash_svm
        cash_svm = 0.75*cash_svm + invested*(1 + BTC2021["Return rate"][i])
        balance_svm.append(cash_svm)
        
    else:
        print(BTC2021.iloc[i])


# In[149]:


cash_svm


# The final return of Gaussian SVM predicted investment suggestions is 11676.02 USD

# ### Summary table of accuracies of four classifier prediction

# In[150]:


data1 = {'Trading Strategy models':['kNN', "Logistic Regression", "Naive Bayes","Gaussian SVM"], 
        'Accuracy':[correct_rate_knn, correct_rate_logr, correct_rate_nb, correct_rate_svm]}

table2 = pd.DataFrame(data=data1)


# In[151]:


table2


# It is clearly to see that, kNN still hold the highest accuracy for investment suggestion label

# ### Summary table of final returns of five trading strategies

# In[160]:


data2 = {'Trading Strategy':['Real label','kNN', "Logistic Regression", "Naive Bayes","Gaussian SVM"], 
        'Final return':[cash_bah, cash_knn, cash_logr, cash_nb, cash_svm]}

table3 = pd.DataFrame(data=data2)


# In[161]:


table3


# kNN finally gained the highest money in the first 4 months in 2021 in trading demo.

# ## 2. Line Chart of trading strategies return

# In[162]:


x_values = BTC2021['Date']

trace0 = go.Scatter(x= x_values, y=balance_bah,
                    mode='lines',marker_color='blue',name='Real label')

trace1 = go.Scatter(x= x_values, y=balance_knn,
                    mode='lines',marker_color='red',name='kNN')

trace2 = go.Scatter(x= x_values, y=balance_logr,
                    mode='lines',marker_color='purple',name='Logistic Regression')

trace3 = go.Scatter(x= x_values, y=balance_nb,
                    mode='lines',marker_color='orange',name='Naive Bayes')

trace4 = go.Scatter(x= x_values, y=balance_svm,
                    mode='lines',marker_color='green',name='Gaussian SVM')

data = [trace0, trace1, trace2, trace3, trace4]

layout = go.Layout(title='Trading Strategy stimulation', xaxis_title="Date",yaxis_title="USD", hovermode='x')

fig = go.Figure(data=data, layout=layout)

fig.update_layout(xaxis=dict(
    tickvals = ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01'],
    ticktext = ["Jan 2021", "Feb 2021", "Mar 2021", "Apr 2021","May 2021"]))

fig.update_layout(legend_title_text='Strategies and models')

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.2,
            showactive=True,
            buttons=list(
                [
                    dict(
                        label="Real label",
                        method="update",
                        args=[
                            {"visible": [True, False, False, False, False]},
                            {"yaxis.title.text": "Real label",},
                        ],
                    ),
                    dict(
                        label="kNN",
                        method="update",
                        args=[
                            {"visible": [False, True, False, False, False]},
                            {"yaxis.title.text": "kNN"},
                        ],
                    ),
                    dict(
                        label="Logistic Regression",
                        method="update",
                        args=[
                            {"visible": [False, False, True, False, False]},
                            {"yaxis.title.text": "Logistic Regression"},
                        ],
                    ),
                    dict(
                        label="Naive Bayes",
                        method="update",
                        args=[
                            {"visible": [False, False, False, True, False]},
                            {"yaxis.title.text": "Naive Bayes"}
                        ],
                    ),
                    dict(
                        label="SVM",
                        method="update",
                        args=[
                            {"visible": [False, False, False, False, True]},
                            {"yaxis.title.text": "SVM"}
                        ],
                    ),
                ]
            ),
        )
    ]
)

pyo.plot(fig, filename = 'Trading Strategy stimulation.html')

fig.show()


# In[163]:


trace0 = go.Box(y=balance_bah, name='Real label', marker_color = 'blue')

trace1 = go.Box(y=balance_knn, name='kNN', marker_color = 'red')

trace2 = go.Box(y=balance_logr, name='Logistic Regression', marker_color = 'green')

trace3 = go.Box(y=balance_nb, name='Naive Bayes', marker_color = 'purple')

trace4 = go.Box(y=balance_svm, name='SVM', marker_color = 'orange')

data = [trace0, trace1, trace2, trace3, trace4]

layout = go.Layout(title='Trading Strategy stimulation', 
                   xaxis_title="Trading strategy",yaxis_title="USD", hovermode='x')

fig = go.Figure(data=data, layout=layout)

fig.update_layout(legend_title_text='Strategies and models')

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.7,
            y=1.2,
            showactive=True,
            buttons=list(
                [
                    dict(
                        label="Real label",
                        method="update",
                        args=[
                            {"visible": [True, False, False, False, False]},
                            {"yaxis.title.text": "real label",},
                        ],
                    ),
                    dict(
                        label="kNN",
                        method="update",
                        args=[
                            {"visible": [False, True, False, False, False]},
                            {"yaxis.title.text": "kNN"},
                        ],
                    ),
                    dict(
                        label="Logistic Regression",
                        method="update",
                        args=[
                            {"visible": [False, False, True, False, False]},
                            {"yaxis.title.text": "Logistic Regression"},
                        ],
                    ),
                    dict(
                        label="Naive Bayes",
                        method="update",
                        args=[
                            {"visible": [False, False, False, True, False]},
                            {"yaxis.title.text": "Naive Bayes"}
                        ],
                    ),
                    dict(
                        label="SVM",
                        method="update",
                        args=[
                            {"visible": [False, False, False, False, True]},
                            {"yaxis.title.text": "SVM"}
                        ],
                    ),
                ]
            ),
        )
    ]
)

pyo.plot(fig, filename = 'Trading Strategy stimulation.html')

fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




