"""
Step 3 (Full Universe): Triplet Screener — All Pairs
Given every possible anchor pair (A, B) from the full 25-stock universe,
find a third stock C such that r_C(t+k) is predictable from r_A(t) and r_B(t).

Since all optimal lags were 0, we use lag_L=0 for all pairs (same-day co-movement).
The conditional expectation is: E[r_C(t+k) | r_A(t), r_B(t)]

Fits: r_C(t+k) = alpha + beta1*r_A(t) + beta2*r_B(t) + epsilon
Records beta1, beta2, p-values, R² for all (pair, C, horizon k) combos.

Outputs:
  - triplet_full_significant.csv  : passes both p-value and R² filter
  - triplet_full_all.csv          : every result (large file)
"""

from project_paths import project_file

import pandas as pd
import numpy as np
import statsmodels.api as sm
import itertools

# --- Config ---
K_HORIZONS    = [1, 2, 3, 5]   # prediction horizons in trading days
P_THRESHOLD   = 0.05            # significance level for both betas
R2_THRESHOLD  = 0.01            # minimum R² to consider (1%)
MIN_OBS       = 200             # minimum observations per regression
LAG_L         = 0               # all lags were 0 — using same-day anchor

# --- Load returns ---
returns_df = pd.read_csv(project_file("nse_log_returns.csv"), index_col=0, parse_dates=True)
ALL_STOCKS = list(returns_df.columns)
returns    = {s: returns_df[s].to_numpy() for s in ALL_STOCKS}
n          = len(returns_df)

print(f"Universe : {len(ALL_STOCKS)} stocks")
print(f"Obs      : {n} trading days")

# Generate all (A, B) pairs — A != B, unordered
ALL_PAIRS = list(itertools.combinations(ALL_STOCKS, 2))
print(f"Anchor pairs: {len(ALL_PAIRS)}")
total = len(ALL_PAIRS) * len(ALL_STOCKS) * len(K_HORIZONS)
print(f"Total regressions to run: ~{total:,} (skipping C in {{A,B}})\n")

# --- Regression helper ---
def run_regression(r_a, r_b, r_c, k, min_obs=MIN_OBS):
    """
    Regress r_C(t+k) on [1, r_A(t), r_B(t)].
    Returns dict of results or None if insufficient data.
    """
    max_k = max(k, 0)
    t_end = n - max_k - 1  # last valid t (inclusive)
    if t_end < min_obs:
        return None

    t_idx = np.arange(0, t_end + 1)
    x1    = r_a[t_idx]          # r_A(t)
    x2    = r_b[t_idx]          # r_B(t)  [lag=0]
    y     = r_c[t_idx + k]      # r_C(t+k)

    mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    if mask.sum() < min_obs:
        return None

    X = sm.add_constant(np.column_stack([x1[mask], x2[mask]]))
    try:
        res = sm.OLS(y[mask], X).fit()
    except Exception:
        return None

    return {
        "alpha":   res.params[0],
        "beta1":   res.params[1],
        "beta2":   res.params[2],
        "p_beta1": res.pvalues[1],
        "p_beta2": res.pvalues[2],
        "r2":      res.rsquared,
        "n_obs":   int(res.nobs),
    }

# --- Main loop ---
records = []
done    = 0

for (stock_a, stock_b) in ALL_PAIRS:
    r_a = returns[stock_a]
    r_b = returns[stock_b]

    candidate_C = [s for s in ALL_STOCKS if s not in (stock_a, stock_b)]

    for stock_c in candidate_C:
        r_c = returns[stock_c]

        for k in K_HORIZONS:
            done += 1
            res = run_regression(r_a, r_b, r_c, k)
            if res is None:
                continue

            records.append({
                "Stock_A":   stock_a,
                "Stock_B":   stock_b,
                "Stock_C":   stock_c,
                "Horizon_k": k,
                "Beta1_A":   round(res["beta1"],  6),
                "Beta2_B":   round(res["beta2"],  6),
                "p_Beta1":   round(res["p_beta1"], 4),
                "p_Beta2":   round(res["p_beta2"], 4),
                "R2":        round(res["r2"],      4),
                "Alpha":     round(res["alpha"],   6),
                "N_obs":     res["n_obs"],
            })

    # Progress
    pairs_done = ALL_PAIRS.index((stock_a, stock_b)) + 1
    if pairs_done % 50 == 0 or pairs_done == len(ALL_PAIRS):
        print(f"  {pairs_done}/{len(ALL_PAIRS)} pairs done — {len(records):,} results so far")

# --- Save full results ---
all_df = pd.DataFrame(records)
all_df.to_csv(project_file("triplet_full_all.csv"), index=False)
print(f"\nFull results: {len(all_df):,} rows → triplet_full_all.csv")

# --- Filter significant ---
sig_df = all_df[
    (all_df["p_Beta1"] < P_THRESHOLD) &
    (all_df["p_Beta2"] < P_THRESHOLD) &
    (all_df["R2"]      > R2_THRESHOLD)
].copy().sort_values("R2", ascending=False).reset_index(drop=True)

sig_df.to_csv(project_file("triplet_full_significant.csv"), index=False)
print(f"Significant triplets (p<{P_THRESHOLD}, R²>{R2_THRESHOLD}): {len(sig_df)}")

# --- Summary ---
if len(sig_df) > 0:
    print("\n=== TOP 30 SIGNIFICANT TRIPLETS BY R² ===")
    cols = ["Stock_A", "Stock_B", "Stock_C", "Horizon_k",
            "Beta1_A", "Beta2_B", "p_Beta1", "p_Beta2", "R2"]
    print(sig_df[cols].head(30).to_string(index=False))

    # Count by horizon
    print("\n=== SIGNIFICANT COUNT BY HORIZON ===")
    print(sig_df.groupby("Horizon_k").size().rename("Count"))

    # Count by Stock_C — which stock is most predictable?
    print("\n=== MOST FREQUENTLY PREDICTED STOCK_C ===")
    print(sig_df["Stock_C"].value_counts().head(10))

    # Count by anchor pair — which pairs are most useful?
    print("\n=== MOST USEFUL ANCHOR PAIRS ===")
    sig_df["Anchor"] = sig_df["Stock_A"] + " + " + sig_df["Stock_B"]
    print(sig_df["Anchor"].value_counts().head(10))

else:
    print("\nNo triplets passed both filters.")
    print("Showing top 20 by R² regardless of significance:\n")
    cols = ["Stock_A", "Stock_B", "Stock_C", "Horizon_k",
            "Beta1_A", "Beta2_B", "p_Beta1", "p_Beta2", "R2"]
    print(all_df.sort_values("R2", ascending=False).head(20)[cols].to_string(index=False))

    # Also show relaxed filter
    relaxed = all_df[
        (all_df["p_Beta1"] < 0.10) &
        (all_df["p_Beta2"] < 0.10) &
        (all_df["R2"]      > 0.005)
    ]
    print(f"\nWith relaxed thresholds (p<0.10, R²>0.005): {len(relaxed)} triplets")