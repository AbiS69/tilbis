# core/feature_engineering.py

import pandas as pd

def compute_macro_features(cpi_df, gdp_df, method='classic'):
    """Compute YoY and Momentum features for CPI and GDP."""
    # Year-over-year % change
    cpi_df['CPI_YoY'] = cpi_df['CPIAUCSL'].pct_change(12)
    gdp_df['GDP_YoY'] = gdp_df['GDPC1'].pct_change(12)

    if method == "classic":
        cpi_df['CPI_Momentum'] = cpi_df['CPI_YoY'] - cpi_df['CPI_YoY'].shift(3)
        gdp_df['GDP_Momentum'] = gdp_df['GDP_YoY'] - gdp_df['GDP_YoY'].shift(3)

    elif method == "macd":
        cpi_df['CPI_Momentum'] = (
            cpi_df['CPI_YoY'].rolling(3).mean() - cpi_df['CPI_YoY'].rolling(6).mean()
        )
        gdp_df['GDP_Momentum'] = (
            gdp_df['GDP_YoY'].rolling(3).mean() - gdp_df['GDP_YoY'].rolling(6).mean()
        )

    elif method == "zscore":
        cpi_df['CPI_Momentum'] = (
            (cpi_df['CPI_YoY'] - cpi_df['CPI_YoY'].rolling(12).mean()) /
            cpi_df['CPI_YoY'].rolling(12).std()
        )
        gdp_df['GDP_Momentum'] = (
            (gdp_df['GDP_YoY'] - gdp_df['GDP_YoY'].rolling(12).mean()) /
            gdp_df['GDP_YoY'].rolling(12).std()
        )

    elif method == "fast3m":
        cpi_df['CPI_Momentum'] = cpi_df['CPIAUCSL'].pct_change(3)
        gdp_df['GDP_Momentum'] = gdp_df['GDPC1'].pct_change(3)

    elif method == "diff3m":
        cpi_df['CPI_Momentum'] = cpi_df['CPIAUCSL'].diff(3)
        gdp_df['GDP_Momentum'] = gdp_df['GDPC1'].diff(3)

    else:
        raise ValueError(f"Invalid momentum method: {method}")

    return cpi_df.dropna(), gdp_df.dropna()
