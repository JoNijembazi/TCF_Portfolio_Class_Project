import yfinance as yf
import pandas as pd


Ticker = yf.Ticker('APO')

print(Ticker.info)