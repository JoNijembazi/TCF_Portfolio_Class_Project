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
prtu[['Sector','Trailing P/E','1Y Forward P/E','Consensus Target']] = 'n.a'
prtu['Type'] = 'Cash'
for x,y in zip(eq_list,prtu['Security'].iloc[:-2]):        
    try:
        # Check type 
        ticker_info = yf.Ticker(x).info
        prtu.loc[prtu['Security']==y,'Type'] = 'Stock'
        try:
            etf_sector = pd.Series(yf.Ticker(x).funds_data.sector_weightings).idxmax()
            prtu.loc[prtu['Security']==y,'Type'] = 'ETF'
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

normalized_codes = { 
                'Consumer Cyclical': 'Consumer Discretionary', 
                'Basic Materials': 'Materials', 
                'Consumer Defensive': 'Consumer Staples', 
                'Technology': 'Information Technology', 
                }

for i in sector_codes.keys():
    prtu.loc[prtu['Sector']==i,'Sector'] = sector_codes[i]

for i in normalized_codes.keys():
    prtu.loc[prtu['Sector']==i,'Sector'] = normalized_codes[i]

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
        prtu.loc[i, 'Market Value'] = prtu.loc[i, 'Position'] * 1000
        prtu.loc[i, 'Market Value (CAD)'] = prtu.loc[i, 'Position'] * 1000
    if prtu.loc[i, 'Security'] == 'USD':
        prtu.loc[i, 'Market Value'] = prtu.loc[i, 'Position'] * 1000
        prtu.loc[i, 'Market Value (CAD)'] = prtu.loc[i, 'Market Value'] * usd_cad
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

    # ETF Data
etfs = prtu.loc[prtu['Type']=='ETF']
    # Functions

        # Portfolio Returns 
def portfolio_mean_return(weights):
    return np.sum(weights * mean_returns[:-2])

def portfolio_ann_return(weights):
    return np.sum(weights * annualized_returns[:-2])


        # Portfolio Volatility 
def portfolio_volatility(weights):
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



GICs = prtu['Sector'].iloc[:-2].unique()
Sector_weights = pd.Series([float(prtu.loc[prtu['Sector']==i,'Weight'].sum()) for i in GICs],index= GICs,dtype='float64').to_numpy()
portfolio_weights = np.array(prtu['Weight'].iloc[:-2])[:, np.newaxis]
ETF_weights = pd.Series([float(etfs.loc[etfs['Sector']==i,'Weight'].sum()) for i in GICs],index= GICs,dtype='float64').to_numpy()

# Risk Free Rate (10-Year US Treasury)
risk_free_rate = yf.download('^TNX',period='1d')['Close'].iloc[-1] / 100

def check_sum(weights):
    return ()


    # Generate random weights
weights = np.random.random(size=(10000, len(portfolio_weights)))
# Normalize weights to unity
weights /= np.sum(weights, axis=1)[:, np.newaxis]

# Calculate portfolio returns and standard deviations 
    # (Utilized Element wise operations to speed up operation processs)
port_returns = np.sum(weights * np.array(annualized_returns[:-2].values), axis=1)
port_sd = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights))


#Visualize Target Portfolios & Frontier 
target =  np.linspace(
    start=port_sd.min(),
    stop=port_sd.max(),
    num=100
            )

size_constraints = tuple((0,0.1) for w in portfolio_weights)

constraints = (
  {'type': 'eq', 'fun': lambda x: portfolio_mean_return(x) - target},
  {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
)

    # instantiate empty container for the objective values to be minimized
obj_sd = []
obj_weight =[]
    # For loop to minimize objective function
print(np.array(portfolio_weights))
for target in target:
  min_result_object = sco.minimize(
    # Objective function
    fun = portfolio_volatility,
    # Initial guess, which is the equal weight array
    x0 = np.array(portfolio_weights).flatten(),
    method = 'SLSQP',
    bounds = size_constraints,
    constraints = constraints
    ) 
  # Extract the objective value and append it to the output container
  obj_sd.append(min_result_object['fun'])
  obj_weight.append(min_result_object['x'])


# Scatter Plot of Mean-Variance Line & Portfolios

fig = px.scatter(pd.DataFrame({'Volatility': obj_sd, 'Returns': target}), x='Volatility', y='Returns', title='Portfolio Returns vs. Volatility')
fig.update_layout(xaxis_title='Portfolio Volatility (%)',
                  yaxis_title='Portfolio Returns (%)')
# fig.update_traces(marker=dict(size=2),
#                   selector=dict(mode='markers'),
#                   opacity=0.5)

# fig.add_scatter(x=port_sd,
#                 y=port_returns,
#                 )

fig.show()

target =  np.linspace(
    start=port_sd.min(),
    stop=port_sd.max(),
    num=100
            )
# Histogram of Portfolio Returns
# fig = px.histogram(x=port_returns, marginal='box',nbins=50, title='Portfolio Returns')
# fig.update_layout(xaxis_title='Portfolio Returns (%)',yaxis_title='Frequency')
# fig.show()

# Histogram of Portfolio Volatility
# fig = px.histogram(x=port_sd, marginal='box',nbins=50, title='Portfolio Volatility')
# fig.update_layout(xaxis_title='Portfolio Volatility (%)',yaxis_title='Frequency')
# fig.show()



def sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_ret = portfolio_ann_return(weights, mean_returns)
    p_vol = portfolio_volatility(weights, cov_matrix)
    return (p_ret - risk_free_rate) / p_vol
