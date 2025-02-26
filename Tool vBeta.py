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
url = f'https://raw.githubusercontent.com/JoNijembazi/TCF-Portfolio/main/TCF20250131.xlsx' 
Stock_Master_list = pd.read_excel(url,engine='openpyxl')

# Clean Data
    # Drop rows with missing values
Stock_Master_list.columns = Stock_Master_list.iloc[8]
Stock_Master_list = Stock_Master_list[Stock_Master_list.columns[[0,2,3,11,12]]]
Stock_Master_list.drop(range(12),inplace=True)
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
        Stock_Master_list.loc[Stock_Master_list['Security']==y,'Sector']
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
etfs = Stock_Master_list.loc[Stock_Master_list['Type']=='ETF']
ETF_weights = np.array(etfs['Weight']/etfs['Weight'].sum())

    # Actives Weights
Actives = Stock_Master_list.loc[Stock_Master_list['Type']=='Stock']
Actives_Weights = np.array(Actives['Weight']/Actives['Weight'].sum())
    # Total Weights (ETF + Active)
portfolio_weights = np.array(Stock_Master_list[['Weight', 'Type', 'Sector']].iloc[:-2])


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
        
        # Size
size_constraints = tuple((0,0.1) for _ in range(len(portfolio_weights)))

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

    # Max returns constraints
def constraints_func(target):
    return (
        {'type': 'eq', 'fun': lambda x: portfolio_ann_return(x) - target},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    )

# Define the constraints function
def constraints_func(target):
    constraints = [
        {'type': 'eq', 'fun': lambda x: portfolio_ann_return(x) - target},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
    ]
    for sector, (min_w, max_w) in IPS_Sector_constraint.items():
        sector_mask = np.array(Stock_Master_list['Sector'].iloc[:-2] == sector)
        constraints.append(
            {'type': 'ineq', 'fun': lambda x, sector_mask=sector_mask, max_w=max_w: max_w - np.sum(x[sector_mask])},
        )   
    return constraints

# Portfolio Returns 

def portfolio_ann_return(weights):
    return np.sum(weights * annualized_returns[:-2])


    # Portfolio Volatility 
def portfolio_volatility(weights):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Produce Sharpe Ratio 
def sharpe_ratio(input):
    p_ret = portfolio_ann_return(input[1]/100)
    p_vol = input[0]
    sharpe = (p_ret - risk_free_rate*100) / p_vol
    return sharpe, p_vol


target =  np.linspace(
    start=10,
    stop=20,
    num=200
            )

# Define the minimization function for all Securities
def minimize_for_target(target, initial_w):
    min_result_object = sco.minimize(
        fun=portfolio_volatility,
        x0=initial_w[:, 0],
        method='SLSQP',
        bounds=size_constraints,
        constraints=constraints_func(target=target)
    )
    return min_result_object['fun']*100, min_result_object['x']*100

minimized_results = Parallel(n_jobs=-1)(delayed(minimize_for_target)(t, np.array(portfolio_weights)) for t in target)

    # Extract the results
obj_sd, obj_weight = zip(*minimized_results)

sharpe_results = Parallel(n_jobs=-1)(delayed(sharpe_ratio)(i)for i in minimized_results)

sharpe,vol = zip(*sharpe_results)

frontiergraph = go.Figure()

frontiergraph.add_trace(go.Scatter(x=obj_sd, 
                         y=target, 
                         mode='lines',
                         name='Efficient Frontier',
                         customdata=sharpe,
                         hovertemplate="Return: %{x}%<br>" +
                         "Standard Deviation: %{y}%"+
                         "<br>Sharpe Ratio: %{customdata}",)
                         )

frontiergraph.update_layout(xaxis_title='Portfolio Volatility (%)',
                  yaxis_title='Portfolio Returns (%)',
                  title = '<i><b>Efficient Frontier vs Assets</b></i>',
                  hovermode ='closest',
                  plot_bgcolor='white',
                  paper_bgcolor='WhiteSmoke',
                  font=dict(color='#8F001A'),
                  title_font=dict(size=20, color='#8F001A'),
                  xaxis=dict(showline=True, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                  yaxis=dict(showline=False, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                  )


frontiergraph.add_trace(go.Scatter(x=Stock_Master_list['Annualized Volatility (%)'],
                y=Stock_Master_list['Annualized Returns (%)'],
                mode='markers',
                name='Assets',
                customdata=Stock_Master_list[['Security', 'Type', 'Country']].values,
                hovertemplate= "<b>%{customdata[0]}</b><br>"+
                "<b>%{customdata[1]}</b><br>"+
                "<b>%{customdata[2]}</b><br><br>"+
                "Return:%{y}%<br>"+
                "Standard Deviation: %{x}%", 
                marker=dict(color='#8F001A', size=8, line=dict(width=0, color='DarkSlateGrey'))
                ))


frontiergraph.show()

frontiergraph_etfs = go.Figure()

frontiergraph_etfs.add_trace(go.Scatter(x=obj_sd, 
                         y=target, 
                         mode='lines',
                         name='Efficient Frontier',
                         customdata=sharpe,
                         hovertemplate="Return: %{x}%<br>" +
                         "Standard Deviation: %{y}%"+
                         "<br>Sharpe Ratio: %{customdata}",)
                         )

frontiergraph_etfs.update_layout(xaxis_title='Portfolio Volatility (%)',
                  yaxis_title='Portfolio Returns (%)',
                  title = '<i><b>Efficient Frontier vs Assets</b></i>',
                  hovermode ='closest',
                  plot_bgcolor='white',
                  paper_bgcolor='WhiteSmoke',
                  font=dict(color='#8F001A'),
                  title_font=dict(size=20, color='#8F001A'),
                  xaxis=dict(showline=True, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                  yaxis=dict(showline=False, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                  )


frontiergraph_etfs.add_trace(go.Scatter(x=Stock_Master_list['Annualized Volatility (%)'],
                y=Stock_Master_list['Annualized Returns (%)'],
                mode='markers',
                name='Assets',
                customdata=Stock_Master_list[['Security', 'Type', 'Country']].values,
                hovertemplate= "<b>%{customdata[0]}</b><br>"+
                "<b>%{customdata[1]}</b><br>"+
                "<b>%{customdata[2]}</b><br><br>"+
                "Return:%{y}%<br>"+
                "Standard Deviation: %{x}%", 
                marker=dict(color='#8F001A', size=8, line=dict(width=0, color='DarkSlateGrey'))
                ))

frontiergraph_etfs.show()


# Chart Correlation Matrix
corr_matrix_graph = px.imshow(corr_matrix.round(2),
            title='<i><b> 5 Year Correlation Matrix</b></i>', 
            labels=dict(color='Correlation'),
            text_auto=True,
            color_continuous_scale='RdBu',
            width=800,
            height=800
            )
corr_matrix_graph.update_layout(plot_bgcolor='white',
                  paper_bgcolor='WhiteSmoke',
                  font=dict(color='#8F001A'),
                  title_font=dict(size=20, color='#8F001A'),
                  xaxis=dict(showline=True, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                  yaxis=dict(showline=False, linewidth=2, linecolor='#8F001A', showgrid=False, zeroline=False),
                  )
corr_matrix_graph.show()

# Chart Portfolio Return to Volatility
Return_Vol_graph = px.scatter(Stock_Master_list,x='Annualized Volatility (%)',y='Annualized Returns (%)',color='Country',hover_name='Security')
Return_Vol_graph.update_layout(title='<i><b>Portfolio Return to Volatility</b></i>',
                  xaxis_title='Annualized Volatility (%)',
                  yaxis_title='Annualized Returns (%)',
                  plot_bgcolor='white',
                  paper_bgcolor='white',
                  font=dict(color='#8F001A'),
                  title_font=dict(size=20, color='#8F001A'),
                  xaxis=dict(showline=True, linewidth=1, linecolor='#8F001A', showgrid=False, zeroline=False),
                  yaxis=dict(showline=True, linewidth=1, linecolor='#8F001A', showgrid=False, zeroline=False),
                  )
Return_Vol_graph.show()

# Chart Price_performance
date = len(df)

    # Calculate ETF performance
Type = Stock_Master_list[Stock_Master_list['Type']=='ETF']['Security']    
chart_performance = df[Type].iloc[-date:].pct_change(fill_method=None)

    # Create a dataframe containing relative performance (first row equals 0)
relative_performance = ((chart_performance + 1).cumprod() - 1)*100
relative_performance.iloc[0] = 0
    
    # Chart
Price_chart_stock = px.line(relative_performance, y=relative_performance.columns, x=relative_performance.index, title='<i><b>Performance Over Time, by Security</b></i>')
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
Price_chart_stock.update_traces(hovertemplate='%{y:.2f}')


# Chart Sector performance

    # Calculate ETF performance
Sector = Stock_Master_list.loc[Stock_Master_list['Sector']=='Information Technology']['Security']
chart_performance = df[Sector].iloc[-date:].pct_change(fill_method=None)

    # Create a dataframe containing relative performance (first row equals 0)
relative_performance = ((chart_performance + 1).cumprod() - 1)*100
relative_performance.iloc[0] = 0
    
    # Chart
Price_chart_Sec = px.line(relative_performance, 
                      y=relative_performance.columns, 
                      x=relative_performance.index, 
                      title='<i><b>Performance Over Time, by Sector</b></i>',
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
Price_chart_Sec.update_traces(hovertemplate='%{y:.2f}')
Price_chart_Sec['layout'].pop('updatemenus')

# Histogram of Portfolio
# Table of Stock Master List
table = go.Figure(data=[go.Table(
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

table.update_layout(title='<i><b>Securities List</b></i>',
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
    title='<i><b>Histogram of Daily Returns</b></i>',
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

histogram.show()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Portfolio Analysis Dashboard", style={'textAlign': 'center', 'color': '#8F001A'}),
    
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload File', style={'backgroundColor': '#8F001A', 'color': 'white'}),
        multiple=False
    ),
    
    dcc.Tabs([
        dcc.Tab(label='Efficient Frontier', children=[
            dcc.Graph(id='efficient-frontier', figure=frontiergraph),
            dcc.Graph(id='efficient-frontier-etfs', figure=frontiergraph_etfs)
        ]),
        dcc.Tab(label='Correlation Matrix', children=[
            dcc.Graph(id='correlation-matrix', figure=corr_matrix_graph)
        ]),
        dcc.Tab(label='Return vs Volatility', children=[
            dcc.Graph(id='return-volatility', figure=Return_Vol_graph)
        ]),
        dcc.Tab(label='Performance Over Time', children=[
            dcc.Graph(id='price-performance', figure=Price_chart_stock)
        ]),
        dcc.Tab(label='Sector Performance Over Time', children=[
            dcc.Graph(id='sector-performance', figure=Price_chart_Sec)
        ]),
        dcc.Tab(label='Securities List', children=[
            dcc.Graph(id='securities-list', figure=table)
        ]),
        dcc.Tab(label='Histogram of Daily Returns', children=[
            dcc.Graph(id='daily-returns-histogram', figure=histogram)
        ])
    ])
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)