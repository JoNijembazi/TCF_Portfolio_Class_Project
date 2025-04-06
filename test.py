import yfinance as yf
import pandas as pd


Ticker = yf.Ticker('AAPL')

Ticker.get_info('beta')


