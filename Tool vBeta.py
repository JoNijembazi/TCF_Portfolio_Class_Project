# import the necessary packages
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import scipy.stats as stats
from scipy.stats import norm 
import scipy.optimize as sco
import scipy.interpolate as sci

# Import Data

##  REMEMBER TO CHANGE THIS URL TO THE NEW ONE LATER
url = f'https://raw.githubusercontent.com/JoNijembazi/TCF-Portfolio/main/U27776029-101.xlsx' 
prtu = pd.read_excel(url,engine='openpyxl')

# Clean Data
    # Drop rows with missing values
prtu.columns = prtu.iloc[8]
prtu = prtu[prtu.columns[[0,2,3,11,12]]]
prtu.drop(range(12),inplace=True)
prtu.reset_index(drop=True,inplace=True)

    # Regions
prtu['Country'] = 'n.a'
for i in range(len(prtu[:-2])):
    if 'US' in prtu['Security'][i]:
        prtu['Country'][i] = 'United States'
    else:
        prtu['Country'][i] = 'Canada'

# Portfolio Analysis

    # Prepare Equity List
eq_list = prtu['Security'][:-2].tolist()
eq_list = [x.replace('-U CN','-UN.TO') for x in eq_list]
eq_list = [x.replace(' CN','.TO') for x in eq_list]
eq_list = [x.replace(' US','') for x in eq_list]

    # Download finance data

df = pd.DataFrame()
for x in eq_list:  
    try:
        yf.Ticker(x).info['exchange']
        stock = yf.download(x, period='5y', interval='1d',repair=True,progress=False)['Close'];
        df = pd.concat([df,stock],axis=1)
    except Exception as e:
        try: 
            yf.Ticker(x.replace('.TO','.V')).info['exchange']
            stock = yf.download(x.replace('.TO','.V'), period='5y', interval='1d',repair=True,progress=False)['Close'];
            df = pd.concat([df,stock],axis=1)
        except Exception as e:
            print(f'Error ticker {x} may be incorrect or delisted')
            df[x] = np.nan
    
    # Remove missing/delisted stocks
df.columns = prtu['Security'][:-2].tolist()
drop_cols = df.columns[df.isna().all()]
df.drop(drop_cols,axis=1,inplace=True)
prtu = prtu[~prtu['Security'].isin(drop_cols)]

    # Calculate Market Value & Weight

        # # Last Price
last_price = df.iloc[-1]
prtu['Price'] = last_price.reindex(prtu['Security'],fill_value=0).values    

        # # Add USD/CAD
usd_cad = yf.Ticker('USDCAD=X').info['regularMarketPrice']    
prtu[prtu['Security']=='USD']['Price'] = usd_cad

        # # Market Value CAD and no CAD
prtu['Market Value'] = prtu['Price'] * prtu['Position']     
prtu['Market Value (CAD)'] = prtu['Market Value']

for i in prtu.index:
    if prtu['Country'][i] == 'United States':
        prtu['Market Value (CAD)'][i] = prtu['Price'][i] * prtu['Position'][i] * usd_cad
    else:
        continue 

        # # Weight
prtu['Weight'] = prtu['Market Value (CAD)'] / prtu['Market Value (CAD)'].sum()

    # Calculate Returns
daily_returns = df.pct_change()

    # Calculate Expected Returns
        # # Returns
mean_returns = (daily_returns.mean() * 252)*100
annualized_returns = (((1 + daily_returns).cumprod().iloc[-1]) ** (252/len(df)) - 1)*100
        
        # # Reindex & match with prtu
annualized_returns =annualized_returns.reindex(prtu['Security'], fill_value=0)
mean_returns = mean_returns.reindex(prtu['Security'], fill_value=0)

prtu['Annualized Returns'] = annualized_returns.values
prtu['Mean Returns'] = mean_returns.values

    # Calculate sigma & variance
ann_var = daily_returns.var() * 253 *100
ann_std = pd.Series(daily_returns.std() * np.sqrt(253)) *100

        # # Reindex & match with prtu
ann_std= ann_std.reindex(prtu['Security'], fill_value=0) 
ann_var= ann_var.reindex(prtu['Security'], fill_value=0)

prtu['Annualized Volatility'] = ann_std.values
prtu['Annualized Variance'] = ann_var.values

    # Chart Portfolio Return to Volatility
fig = px.scatter(prtu,x='Annualized Volatility',y='Annualized Returns',color='Country',hover_name='Security')
fig.update_layout(title='Portfolio Return to Volatility',xaxis_title='Annualized Volatility (%)',yaxis_title='Annualized Returns (%)')
fig.show()

    # Calculate Covariance & Correlation Matrix
cov_matrix = daily_returns.cov() * 253
corr_matrix = daily_returns.corr()
    
    # Chart Correlation Matrix
matrix = px.imshow(corr_matrix.round(2),
            title='Correlation Matrix', 
            labels=dict(color='Correlation'),
            text_auto=True,
            color_continuous_scale='RdBu',
            width=800,
            height=800
            )
matrix.show()

# Portfolio Optimization
    # Assets  
