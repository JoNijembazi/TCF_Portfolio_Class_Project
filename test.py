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
from joblib import Parallel, delayed
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Import Data

##  REMEMBER TO CHANGE THIS URL TO THE NEW ONE LATER
url = 'https://raw.githubusercontent.com/JoNijembazi/TCF-Portfolio/main/TCF20250131.xlsx'
Stock_Master_list = pd.read_excel(url, engine='openpyxl')

# Clean Data
    # Drop rows with missing values
Stock_Master_list.columns = Stock_Master_list.iloc[8]

Stock_Master_list = Stock_Master_list[Stock_Master_list.columns[[0,2,3,11,12]]]
Stock_Master_list.drop(range(12),inplace=True)
Stock_Master_list.dropna(axis=0,how='all',inplace=True)
Stock_Master_list.reset_index(drop=True,inplace=True)


    # Regions
Stock_Master_list['Country'] = 'n.a'

for i in range(len(Stock_Master_list[:-2])):
    if 'US' in Stock_Master_list['Security'][i]:
        Stock_Master_list.loc[i,'Country'] = 'United States'
    else:
        Stock_Master_list.loc[i,'Country'] = 'Canada'

# Portfolio Analysis
    # Prepare Equity List
eq_list = Stock_Master_list['Security'].iloc[:-2].tolist()
eq_list = [x.replace('-U CN','-UN.TO') for x in eq_list]
eq_list = [x.replace(' CN','.TO') for x in eq_list]
eq_list = [x.replace(' US','') for x in eq_list]

    # Download finance data
df = pd.DataFrame()
final_eq_list = []
for x in eq_list:  
    try:
        # Check if the ticker is valid
        yf.Ticker(x).info['exchange']

        # download the stock data
        stock = yf.download(x, period='5y', interval='1d',repair=True,progress=False)['Close'];
        df = pd.concat([df,stock],axis=1)

    except Exception as e:
        try:
            # Check if the ticker is valid 
            yf.Ticker(x.replace('.TO','.V')).info['exchange']    

            # Download the stock data
            stock = yf.download(x.replace('.TO','.V'), period='5y', interval='1d',repair=True,progress=False)['Close'];
            df = pd.concat([df,stock],axis=1)
        except Exception as e:
            print(f'Error ticker {x} may be incorrect or delisted')
            df[x] = np.nan
    
    # Remove missing/delisted stocks
eq_list = df.columns[~df.isna().all()]  
df.columns = Stock_Master_list['Security'].iloc[:-2].tolist()
drop_cols = df.columns[df.isna().all()]
df.drop(drop_cols,axis=1,inplace=True)
Stock_Master_list = Stock_Master_list[~Stock_Master_list['Security'].isin(drop_cols)]


    # Add Descriptive Data
Stock_Master_list[['Sector','Trailing P/E','1Y Forward P/E','Consensus Target']] = 'n.a'
Stock_Master_list['Type'] = 'Cash'
for x,y in zip(eq_list,Stock_Master_list['Security'].iloc[:-2]):        
    try:
        # Check type 
        ticker_info = yf.Ticker(x).info
        Stock_Master_list.loc[Stock_Master_list['Security']==y,'Type'] = 'Stock'
        try:
            etf_sector = pd.Series(yf.Ticker(x).funds_data.sector_weightings).idxmax()
            Stock_Master_list.loc[Stock_Master_list['Security']==y,'Type'] = 'ETF'
            
        except:
            pass 
        # Sector
        Stock_Master_list.loc[Stock_Master_list['Security']==y,'Sector'] = ticker_info.get('sectorDisp', etf_sector)
        print(x,y,Stock_Master_list.loc[Stock_Master_list['Security']==y,'Sector'])
        # Price to Earnings
        Stock_Master_list.loc[Stock_Master_list['Security']==y,'Trailing P/E'] = ticker_info.get('trailingPE', 'n.a')
        Stock_Master_list.loc[Stock_Master_list['Security']==y,'1Y Forward P/E'] = ticker_info.get('forwardPE', 'n.a')
        
        # Consensus Target
        Stock_Master_list.loc[Stock_Master_list['Security']==y,'Consensus Target'] = ticker_info.get('targetMeanPrice', 'n.a')
    except:
        continue

sector_codes = {'realestate': 'Real Estate', 
                'consumer_cyclical': 'Consumer Cyclical', 
                'basic_materials': 'Basic Materials', 
                'consumer_defensive': 'Consumer Defensive', 
                'technology': 'Technology', 
                'communication_services': 'Communication Services', 
                'financial_services': 'Financial Services', 
                'utilities': 'Utilities', 
                'industrials': 'Industrials', 
                'energy': 'Energy', 
                'healthcare': 'Healthcare'}

normalized_codes = { 
                'Consumer Cyclical': 'Consumer Discretionary', 
                'Basic Materials': 'Materials', 
                'Consumer Defensive': 'Consumer Staples', 
                'Technology': 'Information Technology', 
                }

for i in sector_codes.keys():
    Stock_Master_list.loc[Stock_Master_list['Sector']==i,'Sector'] = sector_codes[i]

for i in normalized_codes.keys():
    Stock_Master_list.loc[Stock_Master_list['Sector']==i,'Sector'] = normalized_codes[i]

Stock_Master_list