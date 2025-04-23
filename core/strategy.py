# core/strategy.py

import pandas as pd

def get_weights(regime, tickers):
    """Return asset weights based on the regime, normalized to gross exposure = 1."""
    weights = {
        # 📈 Recovery: Risk-on, favoring growth, financials, real assets
        'Recovery': {
            'QQQ': 0.30,
            'XLF': 0.20,
            'VNQ': 0.20,
            'SPY': 0.15,
            'EFA': 0.15
        },

        # 🌅 Reflation: Early-cycle beta, tech, gold, real estate, utilities
        'Reflation': {
            'QQQ': 0.30,
            'GLD': 0.20,
            'VNQ': 0.20,
            'EFA': 0.15,
            'XLU': 0.15
        },

        # 🌡️ Overheat: Inflation hedge tilt — gold, commodities, defensives
        'Overheat': {
            'GLD': 0.25,
            'GSG': 0.20,
            'QQQ': 0.15,
            'XLF': 0.15,
            'XLU': 0.25
        },

        # 🔥 Stagflation: Defensive long dollar + bonds, short equities and cyclicals
        'Stagflation': {
            'DDX-Y.NYB': 0.30,
            'TLT': 0.10,
            'GSG': -0.10,
            'SPY': -0.10,
            'QQQ': -0.10,
            'XLF': -0.10,
            'EFA': -0.15
        }
    }

    # Fallback if regime is unknown: equal-weight
    fallback = {k: 1 / len(tickers) for k in tickers}
    regime_weights = weights.get(regime, fallback)

    # Normalize to gross exposure = 1.0
    gross = sum(abs(v) for v in regime_weights.values())
    return {k: v / gross for k, v in regime_weights.items()}


def compute_strategy_returns(strategy_data, tickers, trading_cost_bps=10):
    """Apply weights to returns and compute net strategy returns."""
    trading_cost = trading_cost_bps / 10000
    prev_weights = get_weights(strategy_data['Regime'].iloc[0], tickers)
    net_returns = []

    for _, row in strategy_data.iterrows():
        current_weights = get_weights(row['Regime'], tickers)

        # Turnover = sum of absolute weight changes
        turnover = sum(
            abs(current_weights.get(t, 0) - prev_weights.get(t, 0)) for t in tickers
        )

        gross_return = sum(row[t] * current_weights.get(t, 0) for t in tickers)
        cost = turnover * trading_cost
        net_return = gross_return - cost
        net_returns.append(net_return)

        prev_weights = current_weights

    strategy_data['Strategy_Net'] = net_returns
    strategy_data['Cumulative_Strategy_Net'] = (1 + strategy_data['Strategy_Net']).cumprod()
    return strategy_data
