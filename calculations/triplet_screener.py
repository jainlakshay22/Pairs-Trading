"""
Step 3: Triplet Screener
Given each anchor pair (A→B) with its known lag L, find a third stock C
such that r_C(t+k) is predictable from r_A(t) and r_B(t+L).

Fits: r_C(t+k) = alpha + beta1*r_A(t) + beta2*r_B(t+L) + epsilon
Records beta1, beta2, p-values, and R² for all (pair, C, horizon k) combos.

Outputs: triplet_results.csv — filtered to statistically significant triplets.
"""

from project_paths import project_file

import pandas as pd
import numpy as np
import statsmodels.api as sm
from itertools import product

# --- Config ---
ANCHOR_PAIRS = [
    ("ONGC.NS",      "LT.NS"),
    ("TCS.NS",       "INFY.NS"),
    ("TCS.NS",       "RELIANCE.NS"),
    ("ICICIBANK.NS", "RELIANCE.NS"),
    ("INFY.NS",      "RELIANCE.NS"),
]

# Horizons to test (days ahead we try to predict C)
K_HORIZONS = [1, 2, 3, 5]

# Significance threshold
P_THRESHOLD = 0.05
R2_THRESHOLD = 0.01  # 1% — daily returns are noisy, don't expect high R²

# --- Load data ---
returns_df = pd.read_csv(project_file("nse_log_returns.csv"), index_col=0, parse_dates=True)
lag_df     = pd.read_csv(project_file("anchor_pair_lags.csv"))

ALL_STOCKS = list(returns_df.columns)

# Build lag lookup: pair string → optimal lag
lag_lookup = {}
for _, row in lag_df.iterrows():
    key = (row["Pair"].split(" - ")[0].strip(), row["Pair"].split(" - ")[1].strip())
    lag_lookup[key] = int(row["Optimal_Lag"])

# --- Helper: align three series for regression ---
def build_regression_data(r_a, r_b, r_c, lag_L, horizon_k, min_obs=100):
    """
    Aligns:
        X1 = r_A(t)        — leader return at t
        X2 = r_B(t + L)    — follower B return at t+L (B follows A by L days)
        y  = r_C(t + k)    — target C return at t+k

    We need t such that t, t+L, and t+k are all valid indices.
    Returns (X, y) as numpy arrays, or (None, None) if too few observations.
    """
    n = len(r_a)
    assert len(r_b) == n and len(r_c) == n, "Series length mismatch"

    # Valid range: t must satisfy:
    #   t >= 0
    #   t + lag_L >= 0  and  t + lag_L < n  → t in [-lag_L, n-lag_L-1]
    #   t + horizon_k < n                   → t < n - horizon_k
    max_shift = max(lag_L, horizon_k, 0)
    min_shift = min(lag_L, horizon_k, 0)

    t_start = max(0, -min_shift)
    t_end   = n - max_shift - 1  # inclusive

    if t_end - t_start + 1 < min_obs:
        return None, None

    t_idx   = np.arange(t_start, t_end + 1)
    x1      = r_a[t_idx]               # r_A(t)
    x2      = r_b[t_idx + lag_L]       # r_B(t + L)
    y       = r_c[t_idx + horizon_k]   # r_C(t + k)

    # Drop NaNs
    mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    if mask.sum() < min_obs:
        return None, None

    X = np.column_stack([x1[mask], x2[mask]])
    return X, y[mask]


# --- Main loop ---
records = []
total = len(ANCHOR_PAIRS) * len(ALL_STOCKS) * len(K_HORIZONS)
done  = 0

print(f"Screening {total} (pair, C, horizon) combinations...\n")

for (stock_a, stock_b) in ANCHOR_PAIRS:
    lag_L = lag_lookup.get((stock_a, stock_b), 0)

    r_a = returns_df[stock_a].to_numpy()
    r_b = returns_df[stock_b].to_numpy()

    candidate_C = [s for s in ALL_STOCKS if s not in (stock_a, stock_b)]

    for stock_c in candidate_C:
        r_c = returns_df[stock_c].to_numpy()

        for k in K_HORIZONS:
            done += 1
            X_raw, y = build_regression_data(r_a, r_b, r_c, lag_L, k)

            if X_raw is None:
                continue  # not enough data

            # OLS with constant
            X = sm.add_constant(X_raw)
            try:
                model  = sm.OLS(y, X).fit()
            except Exception:
                continue

            alpha   = model.params[0]
            beta1   = model.params[1]   # coefficient on r_A(t)
            beta2   = model.params[2]   # coefficient on r_B(t+L)
            p_alpha = model.pvalues[0]
            p_beta1 = model.pvalues[1]
            p_beta2 = model.pvalues[2]
            r2      = model.rsquared
            n_obs   = int(model.nobs)

            records.append({
                "Pair_A":    stock_a,
                "Pair_B":    stock_b,
                "Lag_L":     lag_L,
                "Stock_C":   stock_c,
                "Horizon_k": k,
                "Alpha":     round(alpha,  6),
                "Beta1_A":   round(beta1,  6),
                "Beta2_B":   round(beta2,  6),
                "p_Beta1":   round(p_beta1, 4),
                "p_Beta2":   round(p_beta2, 4),
                "R2":        round(r2,      4),
                "N_obs":     n_obs,
            })

    print(f"  Done: {stock_a} - {stock_b} (lag={lag_L})")

# --- Save full results ---
all_df = pd.DataFrame(records)
all_df.to_csv(project_file("triplet_results_full.csv"), index=False)
print(f"\nFull results: {len(all_df)} rows → triplet_results_full.csv")

# --- Filter: both betas significant AND R² above threshold ---
sig_df = all_df[
    (all_df["p_Beta1"] < P_THRESHOLD) &
    (all_df["p_Beta2"] < P_THRESHOLD) &
    (all_df["R2"]      > R2_THRESHOLD)
].copy()

# Sort by R² descending
sig_df = sig_df.sort_values("R2", ascending=False).reset_index(drop=True)
sig_df.to_csv(project_file("triplet_results_significant.csv"), index=False)

print(f"Significant triplets (p<{P_THRESHOLD} for both betas, R²>{R2_THRESHOLD}): {len(sig_df)}")
print(f"Saved to triplet_results_significant.csv\n")

if len(sig_df) > 0:
    print("=== TOP 20 SIGNIFICANT TRIPLETS ===")
    display_cols = ["Pair_A", "Pair_B", "Lag_L", "Stock_C", "Horizon_k",
                    "Beta1_A", "Beta2_B", "p_Beta1", "p_Beta2", "R2"]
    print(sig_df[display_cols].head(20).to_string(index=False))
else:
    print("No significant triplets found with current thresholds.")
    print("Try relaxing: increase P_THRESHOLD to 0.10 or drop R2_THRESHOLD to 0.005")
    # Show top 10 by R² regardless of significance
    print("\n=== TOP 10 BY R² (regardless of significance) ===")
    top10 = all_df.sort_values("R2", ascending=False).head(10)
    print(top10[["Pair_A","Pair_B","Stock_C","Horizon_k","Beta1_A","Beta2_B","p_Beta1","p_Beta2","R2"]].to_string(index=False))
