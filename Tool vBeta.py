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
        prtu.loc[i,'Country'] = 'United States'
    else:
        prtu.loc[i,'Country'] = 'Canada'

# Portfolio Analysis
    # Prepare Equity List
eq_list = prtu['Security'].iloc[:-2].tolist()
eq_list = [x.replace('-U CN','-UN.TO') for x in eq_list]
eq_list = [x.replace(' CN','.TO') for x in eq_list]
eq_list = [x.replace(' US','') for x in eq_list]

    # Download finance data
prtu[['Sector','Trailing P/E','1Y Forward P/E','Consensus Target']] = 'n.a'
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
df.columns = prtu['Security'].iloc[:-2].tolist()
drop_cols = df.columns[df.isna().all()]
df.drop(drop_cols,axis=1,inplace=True)
prtu = prtu[~prtu['Security'].isin(drop_cols)]


    # Add Descriptive Data
    # Reprepare Equity List 


for x,y in zip(eq_list,prtu['Security'].iloc[:-2]):        
    try:
        
        ticker_info = yf.Ticker(x).info
        print('Security:', x, ticker_info.get('sectorDisp', 'n.a'))
        try:
        # print('Security:', pd.Series(yf.Ticker(x).funds_data.sector_weightings).idxmax())
                                            #  
                                            #  ))
            etf_sector = pd.Series(yf.Ticker(x).funds_data.sector_weightings).idxmax()
        except:
            pass 
        # Sector
        prtu.loc[prtu['Security']==y,'Sector'] = ticker_info.get('sectorDisp', etf_sector)
        prtu.loc[prtu['Security']==y,'Sector']
        # Price to Earnings
        prtu.loc[prtu['Security']==y,'Trailing P/E'] = ticker_info.get('trailingPE', 'n.a')
        prtu.loc[prtu['Security']==y,'1Y Forward P/E'] = ticker_info.get('forwardPE', 'n.a')
        
        # Consensus Target
        prtu.loc[prtu['Security']==y,'Consensus Target'] = ticker_info.get('targetMeanPrice', 'n.a')
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

for i in sector_codes.keys():
    prtu.loc[prtu['Sector']==i,'Sector'] = sector_codes[i]


        # # Last Price
last_price = df.iloc[-1]
prtu['Price'] = last_price.reindex(prtu['Security'],fill_value=0).values    

        # # Add USD/CAD
usd_cad = yf.Ticker('USDCAD=X').info['regularMarketPrice']    
prtu.loc[prtu['Security']=='USD','Price'] = usd_cad

        # # Market Value CAD and no CAD
prtu['Market Value'] = prtu['Price'] * prtu['Position']     
prtu['Market Value (CAD)'] = prtu['Market Value']

for i in prtu.index:
    if prtu.loc[i, 'Country'] == 'United States':
        prtu.loc[i, 'Market Value (CAD)'] = prtu.loc[i, 'Price'] * prtu.loc[i, 'Position'] * usd_cad 
    if prtu.loc[i, 'Security'] == 'CAD':
        prtu.loc[i, 'Market Value (CAD)'] = prtu.loc[i, 'Position'] * 1000
    else:
        continue 

        # # Weight
prtu['Weight'] = prtu['Market Value (CAD)'] / prtu['Market Value (CAD)'].sum()



# Portfolio Optimization

    # Calculate Expected Returns
        # # Returns
daily_returns = df.pct_change()
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

# Add logs (As per log returns)


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

    # Asset Weights
weights = np.array(prtu['Weight'].iloc[:-2])

    # Functions

        # Portfolio Returns 
def portfolio_mean_return(weights):
    return np.sum(weights * mean_returns)

def portfolio_ann_return(weights):
    return np.sum(weights * annualized_returns)


        # Portfolio Volatility 
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Sector Constraints
IPS_Sector_constraint = {'Communication Services': (0, 0.165),
                      'Consumer Discretionary': (0, 0.173), 
                      'Consumer Staples': (0, 0.154), 
                      'Energy': (0, 0.154), 
                      'Financials Services': (0.163, 0.363), 
                      'Health Care': (0, 0.16), 
                      'Industrials': (0.08, 0.208), 
                      'Information Technology': (0.113, 0.313), 
                      'Materials': (0, 0.166), 
                      'Real Estate': (0, 0.116), 
                      'Utilities': (0, 0.131)
                      }

Sector_weights = [prtu.loc[prtu['Sector']==i,'Weight'].sum() for i in IPS_Sector_constraint.keys()]
print(Sector_weights)

def check_sum(weights):
    return ()

list_portfolio_returns = []
list_portfolio_sd = []

# simulate 5000 random weight vectors (numpy array objects)
for p in range(10000):
  # Return random floats in the half-open interval [0.0, 1.0)
  
  weights = np.random.random(size = len(prtu['Security'].iloc[:-2]))
  
  # Normalize to unity
  
  # The /= operator divides the array by the sum of the array and rebinds "weights" to the new object
  weights /= np.sum(weights)
  
  # Lists are mutable so growing will not be memory inefficient
  list_portfolio_returns.append(portfolio_ann_return(weights, mean_returns.iloc[:-2]))
  list_portfolio_sd.append(portfolio_volatility(weights,cov_matrix))
  
  # Convert list to numpy arrays
  port_returns = np.array(list_portfolio_returns)
  port_sd = np.array(list_portfolio_sd)

# Scatter Plot of Portfolio Returns vs. Volatility
fig = px.scatter(x=port_sd, y=port_returns, title='Portfolio Returns vs. Volatility', trendline='ols')
fig.update_layout(xaxis_title='Portfolio Volatility (%)',yaxis_title='Portfolio Returns (%)')
fig.show()

# Histogram of Portfolio Returns
fig = px.histogram(x=port_returns, marginal='box',nbins=50, title='Portfolio Returns')
fig.update_layout(xaxis_title='Portfolio Returns (%)',yaxis_title='Frequency')
fig.show()

# Histogram of Portfolio Volatility
fig = px.histogram(x=port_sd, marginal='box',nbins=50, title='Portfolio Volatility')
fig.update_layout(xaxis_title='Portfolio Volatility (%)',yaxis_title='Frequency')
fig.show()

risk_free_rate = yf.download('^TNX',period='1d')['Close'].iloc[-1] / 100

def sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret = portfolio_return(weights, mean_returns)
    p_vol = portfolio_volatility(weights, cov_matrix)
    return (p_ret - risk_free_rate) / p_vol
