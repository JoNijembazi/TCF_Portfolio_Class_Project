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
Stock_Master_list[['Sector','Trailing P/E','1Y Forward P/E','Consensus Target','Beta']] = 'n.a'
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
        Stock_Master_list.loc[Stock_Master_list['Security']==y,'Sector']
        # Price to Earnings
        Stock_Master_list.loc[Stock_Master_list['Security']==y,'Trailing P/E'] = ticker_info.get('trailingPE', 'n.a')
        Stock_Master_list.loc[Stock_Master_list['Security']==y,'1Y Forward P/E'] = ticker_info.get('forwardPE', 'n.a')
        
        # Consensus Target
        Stock_Master_list.loc[Stock_Master_list['Security']==y,'Consensus Target'] = ticker_info.get('targetMeanPrice', 'n.a')

        # Beta
        Stock_Master_list.loc[Stock_Master_list['Security']==y,'Beta'] = ticker_info.get('beta', 'n.a')

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

# Quantitative Data
        # # Last Price
last_price = df.apply(lambda x: x.dropna().iloc[-1] if not x.dropna().empty else np.nan)
Stock_Master_list['Price'] = last_price.reindex(Stock_Master_list['Security'],fill_value=0).values    

        # # Add USD/CAD
usd_cad = yf.Ticker('USDCAD=X').info['regularMarketPrice']    
Stock_Master_list.loc[Stock_Master_list['Security']=='USD','Price'] = usd_cad

        # # Market Value CAD and no CAD
Stock_Master_list['Market Value'] = Stock_Master_list['Price'] * Stock_Master_list['Position']     
Stock_Master_list['Market Value (CAD)'] = Stock_Master_list['Market Value']

for i in Stock_Master_list.index:
    if Stock_Master_list.loc[i, 'Country'] == 'United States':
        Stock_Master_list.loc[i, 'Market Value (CAD)'] = Stock_Master_list.loc[i, 'Price'] * Stock_Master_list.loc[i, 'Position'] * usd_cad 
    if Stock_Master_list.loc[i, 'Security'] == 'CAD':
        Stock_Master_list.loc[i, 'Market Value'] = Stock_Master_list.loc[i, 'Position'] * 1000
        Stock_Master_list.loc[i, 'Market Value (CAD)'] = Stock_Master_list.loc[i, 'Position'] * 1000
    if Stock_Master_list.loc[i, 'Security'] == 'USD':
        Stock_Master_list.loc[i, 'Market Value'] = Stock_Master_list.loc[i, 'Position'] * 1000
        Stock_Master_list.loc[i, 'Market Value (CAD)'] = Stock_Master_list.loc[i, 'Market Value'] * usd_cad
    else:
        continue 

        # # Weight
Stock_Master_list['Weight'] = Stock_Master_list['Market Value (CAD)'] / Stock_Master_list['Market Value (CAD)'].sum()

# Portfolio Quant Metrics
    # Calculate Expected Returns
        # # Returns
daily_returns = df.pct_change()
mean_returns = (daily_returns.mean() * 252)*100
annualized_returns = (((1 + daily_returns).cumprod().iloc[-1]) ** (252/len(df)) - 1)*100
        
        # # Reindex & match with masterlist
annualized_returns = annualized_returns.reindex(Stock_Master_list['Security'], fill_value=0)
mean_returns = mean_returns.reindex(Stock_Master_list['Security'], fill_value=0)

Stock_Master_list['Annualized Returns (%)'] = annualized_returns.values
Stock_Master_list['Mean Returns (%)'] = mean_returns.values

    # Calculate sigma & variance
ann_var = daily_returns.var() * 253 *100
ann_std = pd.Series(daily_returns.std() * np.sqrt(253)) *100

        # # Reindex & match with masterlist
ann_std= ann_std.reindex(Stock_Master_list['Security'], fill_value=0) 
ann_var= ann_var.reindex(Stock_Master_list['Security'], fill_value=0)

Stock_Master_list['Annualized Volatility (%)'] = ann_std.values
Stock_Master_list['Annualized Variance (%)'] = ann_var.values

    # Calculate Covariance & Correlation Matrix
cov_matrix = daily_returns.cov() * 253
corr_matrix = daily_returns.corr()    

    # Asset Weights
weights = np.array(Stock_Master_list['Weight'].iloc[:-2])
    
    # Sector Weights
GICs = Stock_Master_list['Sector'].iloc[:-2].unique()
Sector_weights = pd.Series([float(Stock_Master_list.loc[Stock_Master_list['Sector']==i,'Weight'].sum()) for i in GICs], index=GICs, dtype='float64')

    # ETF Data
# Filter ETF weights
ETF_Weights = Stock_Master_list[:-2].apply(lambda row: row['Weight'] if row['Type'] == 'ETF' else 0, axis=1).values
ETF_Weights = ETF_Weights / ETF_Weights.sum()
ETF_Weights = pd.DataFrame({
    'Weight': ETF_Weights,
    'Sector': Stock_Master_list['Sector'].iloc[:-2].values,
    'Country': Stock_Master_list['Country'].iloc[:-2].values,
    'Ticker': Stock_Master_list['Security'].iloc[:-2].values})
ETF_Weights = np.array(ETF_Weights)

# Filter Active weights
Actives_Weights = Stock_Master_list[:-2].apply(lambda row: row['Weight'] if row['Type'] == 'Stock' else 0, axis=1).values
Actives_Weights = Actives_Weights / Actives_Weights.sum()
Actives_Weights= pd.DataFrame({
    'Weight': Actives_Weights,
    'Sector': Stock_Master_list['Sector'].iloc[:-2],
    'Country': Stock_Master_list['Country'].iloc[:-2],
    'Ticker': Stock_Master_list['Security'].iloc[:-2]})
Actives_Weights = np.array(Actives_Weights)

    # Total Weights (ETF + Active)

portfolio_weights = np.array(Stock_Master_list[['Weight', 'Type', 'Sector','Country']].iloc[:-2])

# Portfolio Optimization
    # Risk Free Rate (10-Year US Treasury)
risk_free_rate = yf.download('^TNX',period='1d',progress=False)['Close'].iloc[-1] / 100

    # Generate random weights
weights = np.random.random(size=(10000, len(portfolio_weights)))
# Normalize weights to unity
weights /= np.sum(weights, axis=1)[:, np.newaxis]

# Calculate portfolio returns and standard deviations 
    # (Utilized Element-wise operations and einsum to speed up operation processs)
port_returns = np.sum(weights * np.array(annualized_returns[:-2].values), axis=1)
port_sd = np.sqrt(np.einsum('ij,jk,ik->i', weights, cov_matrix, weights))

# Create constraints
    # Define the constraints & Return function

        # Sector Constraints
IPS_Sector_constraint = {'Communication Services': (0, 0.165),
                      'Consumer Discretionary': (0, 0.173), 
                      'Consumer Staples': (0, 0.154), 
                      'Energy': (0, 0.154), 
                      'Financial Services': (0.163, 0.363), 
                      'Health Care': (0, 0.16), 
                      'Industrials': (0.08, 0.208), 
                      'Information Technology': (0.113, 0.313), 
                      'Materials': (0, 0.166), 
                      'Real Estate': (0, 0.116), 
                      'Utilities': (0, 0.131)
                      }

def sum_by_country(weights, country):
    country_mask = np.array(Stock_Master_list['Country'].iloc[:-2] == country)
    return np.sum(weights[country_mask])

def sum_by_type(weights, type):
    type_mask = np.array(Stock_Master_list['Type'].iloc[:-2] == type)
    return np.sum(weights[type_mask])

    # Max returns constraints

# Define the constraints function
def simple_constraints_func(target):
    return (
        {'type': 'eq', 'fun': lambda x: portfolio_ann_return(x) - target},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: 0.60 - sum_by_country(x, 'Canada')},
        {'type': 'ineq', 'fun': lambda x: sum_by_country(x, 'Canada') - 0.40 },
        {'type': 'eq', 'fun': lambda x: sum_by_type(x, 'Stock') - 1}
    )

def etf_constraints_func(target):
    constraints = [
        {'type': 'eq', 'fun': lambda x: portfolio_ann_return(x) - target},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: 0.60 - sum_by_country(x, 'Canada')},
        {'type': 'ineq', 'fun': lambda x: sum_by_country(x, 'Canada') - 0.40 },
        {'type': 'eq', 'fun': lambda x: sum_by_type(x, 'ETF') - 1}
    ]
    # for sector, (min_w, max_w) in IPS_Sector_constraint.items():
    #     sector_mask = np.array(Stock_Master_list['Sector'].iloc[:-2] == sector)
    #     constraints.append({'type': 'ineq', 'fun': lambda x, sector_mask=sector_mask, max_w=max_w: max_w - np.sum(x[sector_mask])})
    #     constraints.append({'type': 'ineq', 'fun': lambda x, sector_mask=sector_mask, max_w=max_w: np.sum(x[sector_mask]) - min_w })
    return constraints

def full_constraints_func(target):
    constraints = [
        {'type': 'eq', 'fun': lambda x: portfolio_ann_return(x) - target},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: 0.60 - sum_by_country(x, 'Canada')},
        {'type': 'ineq', 'fun': lambda x: sum_by_country(x, 'Canada') - 0.40 }
    ]
    for sector, (min_w, max_w) in IPS_Sector_constraint.items():
        sector_mask = np.array(Stock_Master_list['Sector'].iloc[:-2] == sector)
        constraints.append({'type': 'ineq', 'fun': lambda x, sector_mask=sector_mask, max_w=max_w: max_w - np.sum(x[sector_mask])})
        constraints.append({'type': 'ineq', 'fun': lambda x, sector_mask=sector_mask, max_w=max_w: np.sum(x[sector_mask]) - min_w})
    return constraints

# Portfolio Returns  & Volatility

def portfolio_ann_return(weights):
    return np.sum(weights * annualized_returns[:-2]) 
def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Produce Sharpe Ratio 
def sharpe_ratio(input):
    p_ret = portfolio_ann_return(input[1]/100)
    p_vol = input[0]
    sharpe = (p_ret - risk_free_rate * 100) / p_vol
    return sharpe, p_vol

target =  np.linspace(
        start=10,
        stop=20,
        num=100
            )

# Define the minimization function for all Scenarios
def simple_minimize_for_target(target, initial_w):
    min_result_object = sco.minimize(
        fun=portfolio_volatility,
        x0=initial_w[:, 0],
        method='SLSQP',
        bounds=tuple((0,1) for _ in range(len(initial_w))),
        constraints=simple_constraints_func(target=target)
    )
    return min_result_object['fun']*100, min_result_object['x']*100

def etf_minimize_for_target(target, initial_w):
    min_result_object = sco.minimize(
        fun=portfolio_volatility,
        x0=initial_w[:, 0],
        method='SLSQP',
        bounds= tuple((0,1) for _ in range(len(initial_w))),
        constraints=etf_constraints_func(target=target)
    )
    return min_result_object['fun']*100, min_result_object['x']*100

def full_minimize_for_target(target, initial_w):
    min_result_object = sco.minimize(
        fun=portfolio_volatility,
        x0=initial_w[:, 0],
        method='SLSQP',
        bounds= tuple((0,0.075) for _ in range(len(initial_w))),
        constraints=full_constraints_func(target=target)
    )
    return min_result_object['fun']*100, min_result_object['x']*100

# MINIMIZE
minimized_results = Parallel(n_jobs=-1)(delayed(simple_minimize_for_target)(t, np.array(Actives_Weights)) for t in target)
obj_sd, obj_weight = zip(*minimized_results)

sharpe_results = Parallel(n_jobs=-1)(delayed(sharpe_ratio)(i)for i in minimized_results)
    # Extract the results
sharpe,vol = zip(*sharpe_results)
port_returns_actives = pd.DataFrame(np.array([portfolio_ann_return(i) for i in obj_weight]))
# Combine the weights and names of Active Weights securities


# Minimize function for etfs securities, all constraints

minimized_results_etfs = Parallel(n_jobs=-1)(delayed(etf_minimize_for_target)(t, np.array(ETF_Weights)) for t in target)
sharpe_results = Parallel(n_jobs=-1)(delayed(sharpe_ratio)((sd, weights)) for sd, weights in minimized_results_etfs)
    # Extract the results
obj_sd_etfs,obj_weight_etfs = zip(*minimized_results_etfs)
sharpe_etfs, vol_etfs = zip(*sharpe_results)

etf_weights_returned = pd.DataFrame(obj_weight_etfs)
etf_weights_returned.columns = Stock_Master_list['Security'].iloc[:-2].values

# Minimize function for all securities, all constraints

minimized_results_full = Parallel(n_jobs=-1)(delayed(full_minimize_for_target)(t, np.array(portfolio_weights)) for t in target)
sharpe_results = Parallel(n_jobs=-1)(delayed(sharpe_ratio)(i)for i in minimized_results_full)
    # Extract the results
obj_sd_full,obj_weight_full = zip(*minimized_results_full)
sharpe_full, vol_full = zip(*sharpe_results)
full_weights_returned = pd.DataFrame(obj_weight_full)
full_weights_returned.columns = Stock_Master_list['Security'].iloc[:-2].values

def minimum_variance_full(target_range,weights):
    minimized_results_full = Parallel(n_jobs=-1)(delayed(full_minimize_for_target)(t, np.array(weights)) for t in target_range)
    sharpe_results = Parallel(n_jobs=-1)(delayed(sharpe_ratio)(i)for i in minimized_results_full)
    obj_sd_full,obj_weight = zip(*minimized_results_full)
    sharpe, vol = zip(*sharpe_results)
    return obj_sd_full, sharpe, obj_weight

def minimum_variance_etf(target_range,weights):
    minimized_results = Parallel(n_jobs=-1)(delayed(etf_minimize_for_target)(t, np.array(weights)) for t in target_range)
    sharpe_results = Parallel(n_jobs=-1)(delayed(sharpe_ratio)(i)for i in minimized_results)
    obj_sd,obj_weight = zip(*minimized_results_full)
    sharpe, vol = zip(*sharpe_results)
    return obj_sd, sharpe, obj_weight

def minimum_variance_actives(target_range,weights):
    minimized_results = Parallel(n_jobs=-1)(delayed(simple_minimize_for_target)(t, np.array(weights)) for t in target_range)
    sharpe_results = Parallel(n_jobs=-1)(delayed(sharpe_ratio)(i)for i in minimized_results)
    obj_sd,obj_weight = zip(*minimized_results_full)
    sharpe, vol = zip(*sharpe_results)
    return obj_sd, sharpe, obj_weight

# ------------------

# Graph & Table Section

def create_frontier_graph(obj_sds, obj_weight, target, sharpe, title):
    frontiergraph = go.Figure()

    frontiergraph.add_trace(go.Scatter(x=obj_sds, 
                             y=target, 
                             mode='lines',
                             name='Efficient Frontier',
                             customdata=sharpe,
                             hovertemplate="Return: %{y}%<br>" +
                             "Standard Deviation: %{x}%"+
                             "<br>Sharpe Ratio: %{customdata}",
                             )
                             )

    frontiergraph.update_layout(xaxis_title='Portfolio Volatility (%)',
                      yaxis_title='Portfolio Returns (%)',
                      title = title,
                      hovermode ='closest',
                      plot_bgcolor='white',
                      paper_bgcolor='WhiteSmoke',
                      font=dict(color='#8F001A'),
                      title_font=dict(size=20, color='#8F001A'),
                      xaxis=dict(showline=True, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                      yaxis=dict(showline=False, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      )
    frontiergraph.add_trace(go.Scatter(
                        x=[obj_sds[np.argmax(sharpe)]],
                        y=[target[np.argmax(sharpe)]],
                        mode='markers',
                        name='Optimal Portfolio',                        
                        marker=dict(color='gold', size=9, symbol='star'),
                        hovertemplate="Sharpe Ratio: %{customdata}",
                        customdata=[sharpe[np.argmax(sharpe)]]
                    )) 
    # Add a table for Optimal Portfolio Composition
    optimal_portfolio_table = go.Figure(data=[go.Table(
        header=dict(values=['Security', 'Weight (%)'],
                    fill_color='#8F001A',
                    align='left',
                    font=dict(color='white', size=14),
                    line_color='white'),
        cells=dict(values=[
            Stock_Master_list['Security'].iloc[:-2].values,
            [f'{weight:.2f}' for weight in obj_weight[np.argmax(sharpe)]]  
        ],
        fill_color='white',
        align='left',
        font=dict(color='black', size=12),
        line_color='white'),
    )])

    optimal_portfolio_table.update_layout(title='<i><b>Optimal Composition</b></i>',
                                          plot_bgcolor='white',
                                          paper_bgcolor='WhiteSmoke',
                                          font=dict(color='#8F001A'),
                                          title_font=dict(size=20, color='#8F001A'),
                                          height=30 * (len(Stock_Master_list) + 1))  
    
    return frontiergraph, optimal_portfolio_table

# Create graphs for different portfolios

# ------------------

# Correlation Matrix
corr_matrix_graph = px.imshow(corr_matrix.round(2),
            title='<i><b> 5 Year Correlation Matrix</b></i>', 
            labels=dict(color='Correlation'),
            text_auto=True,
            color_continuous_scale='RdBu',
            width=1600,
            height=800
            )
corr_matrix_graph.update_layout(plot_bgcolor='white',
                  paper_bgcolor='WhiteSmoke',
                  font=dict(color='#8F001A'),
                  title_font=dict(size=20, color='#8F001A'),
                  xaxis=dict(showline=True, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                  yaxis=dict(showline=False, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                  )

# ------------------

# Portfolio Return to Volatility
Return_Vol_graph = px.scatter(Stock_Master_list,x='Annualized Volatility (%)',y='Annualized Returns (%)',color='Country',hover_name='Security',)
Return_Vol_graph.update_layout(title='<i><b>TCF Holdings Return to Volatility</b></i>',
                  xaxis_title='Annualized Volatility(%)',
                  yaxis_title='Annualized Returns(%)',
                  plot_bgcolor='white',
                  paper_bgcolor='white',
                  font=dict(color='#8F001A'),
                  hovermode='closest',
                  hoverlabel=dict(
                      bgcolor="white",
                      font_size=12,
                      font_family="Tahoma"
                  ),
                  title_font=dict(size=20, color='#8F001A'),
                  xaxis=dict(showline=True, linewidth=1, linecolor='#8F001A', showgrid=False, zeroline=False),
                  yaxis=dict(showline=True, linewidth=1, linecolor='#8F001A', showgrid=False, zeroline=False),
                  )
Return_Vol_graph.update_traces(marker=dict(size=10))

# ------------------

# Price_performance
date = len(df)
 
# Table of Stock Master List
securities_list = go.Figure(data=[go.Table(
    header=dict(values=list(Stock_Master_list.columns),
                fill_color='#8F001A',
                align='left',
                font=dict(color='white', size=14),
                line_color='white'),
    cells=dict(values=[Stock_Master_list[col].apply(lambda x: f'{x:.2f}' if isinstance(x, (int, float)) else x) for col in Stock_Master_list.columns],
               fill_color='white',
               align='left',
               font=dict(color='black', size=12),
               line_color='white'),
               )  # Adjust the height of the cells
])

securities_list.update_layout(title='<i><b>Securities List</b></i>',
                    plot_bgcolor='white',
                    paper_bgcolor='WhiteSmoke',
                    font=dict(color='#8F001A'),
                    title_font=dict(size=20, color='#8F001A'),
                    height=30 * (len(Stock_Master_list) + 1))  # Adjust the height of the table

# Histogram of Daily Returns
histogram = go.Figure()

for stock in daily_returns.columns:
    histogram.add_trace(go.Histogram(
        x=daily_returns[stock]*100,
        name=stock,
        opacity=0.75,
        nbinsx=50  # Increase the number of bins
    ))

histogram.update_layout(
    title='<i><b>Histogram of Daily Returns (%)</b></i>',
    xaxis_title='Daily Returns',
    yaxis_title='Frequency',
    barmode='overlay',
    plot_bgcolor='white',
    paper_bgcolor='WhiteSmoke',
    font=dict(color='#8F001A'),
    title_font=dict(size=20, color='#8F001A'),
    xaxis=dict(showline=True, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
    yaxis=dict(showline=True, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
)

# Display Graphs
# Build a Dash app with tabs to separate the graphs
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Efficient Frontier', children=[
            html.Div([
                html.Div([
                    dcc.Slider(
                        id='active_slider',
                        min=10,
                        max=30,
                        value=20,
                        marks=None,
                        step=0.1,
                        tooltip={
                            "placement": "bottom", 
                            "always_visible": True,
                            "style": {'font-family': 'Tahoma', 'color': 'white'},
                            "template": '{value}%'
                        },
                    ),
                    dcc.Graph(id='frontiergraph_actives', style={'width': '100%', 'margin': '0 auto'}),
                    dcc.Graph(id='table_actives', style={'width': '100%', 'margin': '0 auto', 'border-top': '0px solid #8F001A'}),
                ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top','backgroundColor': 'WhiteSmoke'}),
                
                html.Div([
                    dcc.Slider(
                        id='etf_slider',
                        min=10,
                        max=30,
                        value=20,
                        marks=None,
                        step=0.1,
                        tooltip={
                            "placement": "bottom", 
                            "always_visible": True,
                            "style": {'font-family': 'Tahoma', 'color': 'white'},
                            "template": '{value}%'
                        }, 
                    ),
                    dcc.Graph(id='frontiergraph_etfs', style={'width': '100%', 'margin': '0 auto'}),
                    dcc.Graph(id='table_etfs', style={'width': '100%', 'margin': '0 auto'}),
                ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top','backgroundColor': 'WhiteSmoke'}),
                html.Div([
                    dcc.Slider(
                        id='full_slider',
                        min=10,
                        max=30,
                        value=20,
                        marks=None,
                        step=0.1,
                        tooltip={
                            "placement": "bottom", 
                            "always_visible": True,
                            "style": {'font-family': 'Tahoma', 'color': 'white'},
                            "template": '{value}%'
                        },
                    ),
                    dcc.Graph(id='frontiergraph_full', style={'width': '100%', 'margin': '0 auto', 'border-top': '0px solid #8F001A'}),
                    dcc.Graph(id='table_full', style={'width': '100%', 'margin': '0 auto', 'border-top': '0px solid #8F001A'}),
                ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'flex-start','backgroundColor': 'WhiteSmoke'}),
        ]),
        
        dcc.Tab(label='Securities List', children=[
            dcc.Graph(figure=securities_list)
        ]),
        dcc.Tab(label='Performance Holdings', children=[
            html.Div([
                dcc.Dropdown(
                    id='sector_selector',
                    options=[{'label': sector, 'value': sector} for sector in Stock_Master_list['Sector'].unique()],
                    value=Stock_Master_list['Sector'].unique()[0],
                    clearable=False,
                    style={
                        'width': '50%',
                        'margin': '0 auto',
                        'display': 'inline-block',
                        'font-family': 'Tahoma',
                    },
                ),
                dcc.Dropdown(
                    id='stock_selector',
                    options=[{'label': stocks, 'value': stocks} for stocks in Stock_Master_list['Security'].iloc[:-2]],
                    value=Stock_Master_list['Security'].iloc[:-2][0],
                    clearable=False,
                    style={
                        'width': '50%',
                        'margin': '0 auto',
                        'display': 'inline-block',
                        'font-family': 'Tahoma',
                    }
                ),
                dcc.Graph(id='Price_chart_Sec', style={'width': '50%', 'margin': '0 auto', 'display': 'inline-block'}),
                dcc.Graph(id='Price_chart_stock', style={'width': '50%', 'margin': '0 auto', 'display': 'inline-block'}),
                dcc.Graph(figure=Return_Vol_graph,style={'width': '100%', 'margin': '0 auto',})
            ], style={'textAlign': 'center', 'margin-bottom': '20px', 'backgroundColor': 'WhiteSmoke', 'font-family': 'Tahoma'}),
        ]),
        dcc.Tab(label='Histogram of Daily Returns', children=[
            dcc.Graph(figure=histogram)
        ]),
        dcc.Tab(label='Correlation of Portfolio Stocks', children=[
            dcc.Graph(figure=corr_matrix_graph, style={'width': '100%', 'margin': '0 auto'})
        ]),
    ], style={'font-family': 'Tahoma'})
])

# Price Charts
@app.callback( Output('Price_chart_Sec','figure'),
               Output('Price_chart_stock','figure'),
               Input('sector_selector', 'value'),
               Input('stock_selector', 'value'),
               )
def update_graphs(sector_selector, stock_selector):
    # 1. Chart Sector performance
    Sector = Stock_Master_list.loc[Stock_Master_list['Sector']==sector_selector]['Security']
    chart_performance = df[Sector].iloc[-date:].pct_change(fill_method=None)

    # Create a dataframe containing relative performance (first row equals 0)
    relative_performance = ((chart_performance + 1).cumprod() - 1)*100
    relative_performance.iloc[0] = 0
        
    # Chart
    Price_chart_Sec = px.line(relative_performance, 
                          y=relative_performance.columns, 
                          x=relative_performance.index, 
                          title=f'<i><b>Performance Over Time, {sector_selector}</b></i>',
                          )

    Price_chart_Sec.update_layout(xaxis_title='Date', 
                      yaxis_title='Performance', 
                      legend_title='Securities',
                      plot_bgcolor='white',
                      paper_bgcolor='WhiteSmoke',
                      font=dict(color='#8F001A'),
                      title_font=dict(size=20, color='#8F001A'),
                      xaxis=dict(showline=True, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                      yaxis=dict(showline=True, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                      margin=dict(t=50)  # Reduce the top margin
                      )
    
    Price_chart_Sec.update_traces(hovertemplate='%{y:.2f}',connectgaps=True)
    Price_chart_Sec['layout'].pop('updatemenus')

    # 2. Chart Stock Price
    chart_performance = df[[stock_selector]].iloc[-date:].pct_change(fill_method=None)

    # Create a dataframe containing relative performance (first row equals 0)
    relative_performance = ((chart_performance + 1).cumprod() - 1)*100
    relative_performance.iloc[0] = 0
    
    # Chart
    Price_chart_stock = px.line( relative_performance,
                            y=relative_performance.columns, 
                            x=relative_performance.index,
                            title=f'<b>{stock_selector}</b>' 
                            )
    Price_chart_stock.update_layout(xaxis_title='Date', 
                  yaxis_title='Performance', 
                  legend_title='Securities',
                  plot_bgcolor='white',
                  paper_bgcolor='WhiteSmoke',
                  font=dict(color='#8F001A'),
                  title_font=dict(size=20, color='#8F001A'),
                  xaxis=dict(showline=True, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                  yaxis=dict(showline=True, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                  )
    Price_chart_stock.update_traces(hovertemplate='%{y:.2f}',connectgaps=True)
    Price_chart_stock['layout'].pop('updatemenus')    
    return Price_chart_Sec, Price_chart_stock

# Optimal Portfolio Charts             
@app.callback(
    Output('frontiergraph_full', 'figure'),
    Output('table_full', 'figure'),
    Input('full_slider','value'),
    )
def update_full(full_slider):
    target =  np.linspace(start=10, stop=full_slider, num=100)
    obj_sd, sharpe, obj_weight = minimum_variance_full(target, portfolio_weights)
    frontiergraph_full, table_full = create_frontier_graph(obj_sd, obj_weight,target, sharpe, '<i><b>Efficient Frontier (Full Portfolio)</b></i>')
    return frontiergraph_full, table_full

@app.callback(
    Output('frontiergraph_actives', 'figure'),
    Output('table_actives', 'figure'),
    Input('active_slider','value'),
)
def update_active(active_slider):
    target =  np.linspace(start=10, stop=active_slider, num=100)
    obj_sd, sharpe, obj_weight = minimum_variance_actives(target, Actives_Weights)
    frontiergraph_actives, table_actives = create_frontier_graph(obj_sd, obj_weight,target, sharpe, '<i><b>Efficient Frontier (Active Positions)</b></i>')
    return frontiergraph_actives, table_actives

@app.callback(
    Output('frontiergraph_etfs', 'figure'),
    Output('table_etfs', 'figure'),
    Input('etf_slider','value'),
)
def update_etf(etf_slider):
        target =  np.linspace(start=10, stop=etf_slider, num=100)
        obj_sd, sharpe, obj_weight = minimum_variance_etf(target, ETF_Weights)
        frontiergraph_etfs, table_etfs = create_frontier_graph(obj_sd, obj_weight,target, sharpe, '<i><b>Efficient Frontier (ETFs Only)</b></i>')
        return frontiergraph_etfs, table_etfs

if __name__ == '__main__':
    app.run(debug=True)