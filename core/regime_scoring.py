# core/regime_scoring.py

import numpy as np
import pandas as pd

# ----------------------------------------
# 1. Get feature matrix from macro data
# ----------------------------------------

def get_macro_features_matrix(macro_df):
    """Extract feature matrix for scoring."""
    return macro_df[['CPI_Momentum', 'GDP_Momentum']].copy()

# ----------------------------------------
# 2. Define regime "prototypes"
# ----------------------------------------

def define_regime_centroids(macro_df):
    """
    Compute centroids (average macro features) from historical regime assignments.
    Requires `macro_df` to have: ['CPI_Momentum', 'GDP_Momentum', 'Regime']
    """
    centroids = {}
    regimes = macro_df['Regime'].dropna().unique()

    for regime in regimes:
        subset = macro_df[macro_df['Regime'] == regime][['CPI_Momentum', 'GDP_Momentum']]
        if not subset.empty:
            centroids[regime] = subset.mean().values


    centroid_df = pd.DataFrame(centroids, index=['CPI_Momentum', 'GDP_Momentum']).T
    print("\n📍 Computed Regime Centroids:")
    print(centroid_df.round(4))
    return centroids


# ----------------------------------------
# 3. Score regimes based on distance
# ----------------------------------------

def compute_regime_scores(features_df, centroids):
    """
    Compute inverse-distance scores for each regime per time step.
    Returns a DataFrame of scores (rows = time, cols = regimes).
    """
    scores = []

    for _, row in features_df.iterrows():
        row_vec = row.values
        row_scores = {}

        for regime, centroid in centroids.items():
            distance = np.linalg.norm(row_vec - centroid)
            score = 1 / (1 + distance) ** 3
            row_scores[regime] = score

        total = sum(row_scores.values())
        row_scores = {k: v / total for k, v in row_scores.items()}  # Normalize
        scores.append(row_scores)

    return pd.DataFrame(scores, index=features_df.index)

# ----------------------------------------
# 4. Blend weights from regime scores
# ----------------------------------------

def blend_weights(regime_scores_row, regime_weight_map, all_assets):
    """
    Given a row of regime scores and a dict of regime weights,
    return a single blended weight vector (dict of asset: weight).
    """
    blended = {asset: 0.0 for asset in all_assets}

    for regime, score in regime_scores_row.items():
        weights = regime_weight_map.get(regime, {})
        for asset in all_assets:
            blended[asset] += score * weights.get(asset, 0.0)

    # Optional: Normalize gross exposure to 1 (abs weights)
    gross = sum(abs(w) for w in blended.values())
    if gross > 0:
        blended = {k: v / gross for k, v in blended.items()}

    return blended
