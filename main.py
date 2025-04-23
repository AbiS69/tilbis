# main.py

import pandas as pd
from core.data_loader import download_macro_data, download_price_data
from core.feature_engineering import compute_macro_features
from core.regime_logic import lag_macro_features, merge_macro_features, assign_regimes
from core.strategy import compute_strategy_returns
from core.plot_tools import plot_all_in_one

# ---------------- CONFIG ----------------
start_date = '1999-01-01'
end_date = '2025-04-21'
momentum_method = 'classic'
tickers = [
    'SPY', 'QQQ', 'TLT', 'IEF', 'SHY',
    'GLD', 'GSG', 'DX-Y.NYB', 'VNQ', 'EFA', 'XLF', 'XLU'
]
use_cache = False
cache_file = "cached_prices.csv"
trading_cost_bps = 10

# ---------------- DATA ----------------
print("📥 Downloading data...")
cpi, gdp = download_macro_data(start_date, end_date)
price_data = download_price_data(tickers, start_date, cache_file, use_cache)

# Flatten multiindex columns if cached
if isinstance(price_data.columns, pd.MultiIndex):
    price_data.columns = price_data.columns.get_level_values(0)

# ---------------- FEATURES ----------------
print("🧠 Engineering features...")
cpi_feat, gdp_feat = compute_macro_features(cpi, gdp, method=momentum_method)
cpi_feat, gdp_feat = lag_macro_features(cpi_feat, gdp_feat)
macro = merge_macro_features(cpi_feat, gdp_feat)
macro = assign_regimes(macro)

# ---------------- RETURNS ----------------
print("📈 Computing returns and strategy...")
returns = price_data.pct_change().dropna()
strategy_data = pd.merge(macro[['Regime']], returns, left_index=True, right_index=True, how='inner')
strategy_data = compute_strategy_returns(strategy_data, tickers, trading_cost_bps=trading_cost_bps)

# ---------------- RESULTS ----------------
def annualized_sharpe(rets, risk_free=0.0):
    excess = rets - risk_free
    return (excess.mean() / excess.std()) * (12 ** 0.5)

def annualized_return(returns, periods_per_year=12):
    total_growth = (1 + returns).prod()
    years = len(returns) / periods_per_year
    return total_growth**(1 / years) - 1

strat_ret = annualized_return(strategy_data['Strategy_Net'])
spy_ret = annualized_return(strategy_data['SPY'])
sharpe_strat = annualized_sharpe(strategy_data['Strategy_Net'])
sharpe_spy = annualized_sharpe(strategy_data['SPY'])

# ---------------- OUTPUT ----------------
print("\n📊 Results:")
print(f"🔹 Strategy CAGR         : {strat_ret:.2%}")
print(f"🔹 Strategy Sharpe Ratio : {sharpe_strat:.2f}")
print(f"🔸 SPY Buy & Hold CAGR  : {spy_ret:.2%}")
print(f"🔸 SPY Buy & Hold Sharpe: {sharpe_spy:.2f}")

# Call after strategy_data and macro are ready:
plot_all_in_one(macro, price_data['SPY'], strategy_data, tickers)
print("✅ All done!")
# ---------------- END ----------------
