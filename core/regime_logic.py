# core/regime_logic.py

import pandas as pd

def lag_macro_features(cpi_df, gdp_df):
    """Lag CPI by 1 month, GDP by 2 months to simulate release delays."""
    cpi_df['CPI_Momentum_Lagged'] = cpi_df['CPI_Momentum'].shift(1)
    gdp_df['GDP_Momentum_Lagged'] = gdp_df['GDP_Momentum'].shift(2)
    return cpi_df, gdp_df


def merge_macro_features(cpi_df, gdp_df):
    """Merge CPI and GDP features into one macro DataFrame."""
    df = pd.merge(
        cpi_df[['CPI_YoY', 'CPI_Momentum_Lagged']],
        gdp_df[['GDP_YoY', 'GDP_Momentum_Lagged']],
        left_index=True, right_index=True, how='inner'
    ).dropna()

    df.rename(columns={
        'CPI_Momentum_Lagged': 'CPI_Momentum',
        'GDP_Momentum_Lagged': 'GDP_Momentum'
    }, inplace=True)

    return df


def classify_regime(row):
    """Classify macro regime based on CPI and GDP momentum."""
    if   (row['CPI_Momentum'] > 0) and (row['GDP_Momentum'] > 0):  return 'Overheat'
    elif (row['CPI_Momentum'] < 0) and (row['GDP_Momentum'] > 0):  return 'Recovery'
    elif (row['CPI_Momentum'] > 0) and (row['GDP_Momentum'] < 0):  return 'Stagflation'
    elif (row['CPI_Momentum'] < 0) and (row['GDP_Momentum'] < 0):  return 'Reflation'
    else:                                                          return 'Unclassified'


def assign_regimes(macro_df):
    """Apply regime classification row-wise."""
    macro_df['Regime'] = macro_df.apply(classify_regime, axis=1)
    return macro_df
