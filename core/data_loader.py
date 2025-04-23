# core/data_loader.py

import pandas_datareader.data as web
import yfinance as yf
import pandas as pd
from datetime import datetime

def download_macro_data(start_date, end_date):
    cpi = web.DataReader('CPIAUCSL', 'fred', start_date, end_date)
    gdp = web.DataReader('GDPC1',   'fred', start_date, end_date)

    # Resample GDP to monthly (it’s quarterly by default)
    gdp_monthly = gdp.resample('MS').ffill()
    
    return cpi, gdp_monthly


def download_price_data(tickers, start_date, cache_file="cached_prices.csv", use_cache=False):
    import os

    if use_cache and os.path.exists(cache_file):
        print(f"📦 Loading cached price data from {cache_file}")
        return pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)

    print("🌐 Downloading fresh price data from Yahoo Finance...")
    price_data = (
        yf.download(tickers, start=start_date, interval='1mo', auto_adjust=False, progress=False)['Adj Close']
        .dropna()
    )

    price_data.to_csv(cache_file)
    print(f"💾 Saved fresh data to {cache_file}")
    return price_data
