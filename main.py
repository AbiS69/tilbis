# main.py

import pandas as pd
from core.data_loader import download_macro_data, download_price_data
from core.feature_engineering import compute_macro_features
from core.regime_logic import lag_macro_features, merge_macro_features, assign_regimes
from core.strategy import compute_strategy_returns
from core.plot_tools import plot_all_in_one
from core.regime_scoring import (
    get_macro_features_matrix,
    define_regime_centroids,
    compute_regime_scores,
    blend_weights
)
from core.plot_tools import plot_regimes_in_feature_space
from core.plot_tools import plot_regime_scores_over_time




# ---------------- CONFIG ----------------
start_date = '1999-01-01'
end_date = '2025-04-21'
momentum_method = 'zscore'
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
# ----- OPTIONAL: Regime scoring logic -----
X_macro = get_macro_features_matrix(macro)
centroids = define_regime_centroids(macro)
regime_scores = compute_regime_scores(X_macro, centroids)
macro["Regime"] = regime_scores.idxmax(axis=1)


# ---------------- RETURNS ----------------
print("📈 Computing returns and strategy...")
returns = price_data.pct_change().dropna()

# Merge regime scores and returns
strategy_data = pd.merge(regime_scores, returns, left_index=True, right_index=True, how='inner')

# Build your regime weight map using get_weights(regime) from strategy.py
from core.strategy import get_weights
regime_weight_map = {regime: get_weights(regime, tickers) for regime in regime_scores.columns}

# Blend weights + compute strategy returns
strategy_returns = []
prev_weights = {t: 0 for t in tickers}

for date, row in strategy_data.iterrows():
    scores = row[regime_scores.columns].to_dict()
    blended = blend_weights(scores, regime_weight_map, tickers)

    gross_return = sum(row[t] * blended.get(t, 0.0) for t in tickers)
    turnover = sum(abs(blended[t] - prev_weights.get(t, 0.0)) for t in tickers)
    cost = turnover * (trading_cost_bps / 10000)
    net_return = gross_return - cost

    strategy_returns.append(net_return)
    prev_weights = blended

# Add this after computing net returns
total_turnover = 0
for date, row in strategy_data.iterrows():
    scores = row[regime_scores.columns].to_dict()
    blended = blend_weights(scores, regime_weight_map, tickers)
    turnover = sum(abs(blended[t] - prev_weights.get(t, 0.0)) for t in tickers)
    total_turnover += turnover
    prev_weights = blended

print(f"\n🔍 Total turnover over period: {total_turnover:.2f}")
print(f"📉 Total cost paid: {total_turnover * trading_cost_bps/10000:.2%}")
strategy_data['Strategy_Net'] = strategy_returns
strategy_data['Cumulative_Strategy_Net'] = (1 + strategy_data['Strategy_Net']).cumprod()


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
strategy_data["Regime"] = macro["Regime"]
plot_regime_scores_over_time(regime_scores)

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
