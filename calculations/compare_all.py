"""
Master Execution Script:
Loads NSE sector data, evaluates candidate pairs across all individual methods,
computes dual-objective rankings, and displays the comparative leaderboard.
"""

from project_paths import project_file
import itertools
import pandas as pd
import numpy as np

# Import all individual method modules
from methods.ssd_method import compute_ssd
from methods.correlation_method import compute_pearson_corr, compute_max_lagged_corr
from methods.cointegration_method import compute_cointegration
from methods.dcct_method import compute_dcct
from methods.ac_method import compute_ac

# 1. Load Preprocessed Data
prices_df = pd.read_csv(project_file("nse_cleaned_prices.csv"), index_col=0, parse_dates=True)
norm_df = pd.read_csv(project_file("nse_normalized_prices.csv"), index_col=0, parse_dates=True)
log_p_df = pd.read_csv(project_file("nse_log_prices.csv"), index_col=0, parse_dates=True)
returns_df = pd.read_csv(project_file("nse_log_returns.csv"), index_col=0, parse_dates=True)

# Select a sub-universe of representative stocks across sectors for quick pairwise screening
candidate_tickers = [
    "TCS.NS", "INFY.NS",          # IT
    "HDFCBANK.NS", "ICICIBANK.NS", # Finance
    "SUNPHARMA.NS", "CIPLA.NS",   # Healthcare
    "RELIANCE.NS", "ONGC.NS",     # Energy
    "LT.NS", "SIEMENS.NS"         # Industrials
]

pairs = list(itertools.combinations(candidate_tickers, 2))
results = []

print(f"Running multi-method evaluation on {len(pairs)} candidate pairs...")

for stock_a, stock_b in pairs:
    # Series inputs
    p_norm_a, p_norm_b = norm_df[stock_a].to_numpy(), norm_df[stock_b].to_numpy()
    lp_a, lp_b = log_p_df[stock_a].to_numpy(), log_p_df[stock_b].to_numpy()
    r_a, r_b = returns_df[stock_a].to_numpy(), returns_df[stock_b].to_numpy()
    
    # 1. SSD Measure (Lower = Better)
    ssd_val = compute_ssd(p_norm_a, p_norm_b)
    
    # 2. Pearson Correlation (Higher = Better)
    pearson_val = compute_pearson_corr(r_a, r_b)
    
    # 3. Fixed Lagged Correlation (Higher = Better)
    lag_corr, opt_lag = compute_max_lagged_corr(r_a, r_b, max_lag=10)
    
    # 4. Engle-Granger Cointegration p-value (Lower = Better)
    _, p_val, gamma = compute_cointegration(lp_a, lp_b)
    
    # 5. DCCT Measure (psi=25, p=25) (Higher = Better)
    dcct_val = compute_dcct(r_a, r_b, p=25, psi=25)
    
    # 6. Aligned Correlation AC Distance (Lower = Better)
    ac_dist, best_k, _, _ = compute_ac(r_a, r_b, k_values=[5, 12, 25])
    
    results.append({
        "Pair": f"{stock_a} - {stock_b}",
        "SSD": ssd_val,
        "Pearson_Corr": pearson_val,
        "Lagged_Corr": lag_corr,
        "Optimal_Lag": opt_lag,
        "Coint_pVal": p_val,
        "DCCT_psi25": dcct_val,
        "AC_Distance": ac_dist,
        "AC_Best_k": best_k
    })

df = pd.DataFrame(results)

# -------------------------------------------------------------
# DUAL COMBINATION RANKINGS (Gupta & Chatterjee 2020)
# -------------------------------------------------------------
# Rank each column (1 = Best)
df["Rank_SSD"] = df["SSD"].rank(ascending=True)                    # Low SSD is best
df["Rank_Pearson"] = df["Pearson_Corr"].rank(ascending=False)      # High Corr is best
df["Rank_DCCT"] = df["DCCT_psi25"].rank(ascending=False)           # High DCCT is best
df["Rank_AC"] = df["AC_Distance"].rank(ascending=True)             # Low AC distance is best
df["Rank_Coint"] = df["Coint_pVal"].rank(ascending=True)           # Low p-value is best

# Combined Selection Objectives
df["DCCT_and_SSD_Score"] = (df["Rank_DCCT"] + df["Rank_SSD"]) / 2.0
df["AC_and_Coint_Score"] = (df["Rank_AC"] + df["Rank_Coint"]) / 2.0
df["CORR_and_SSD_Score"] = (df["Rank_Pearson"] + df["Rank_SSD"]) / 2.0

# -------------------------------------------------------------
# Output Leaderboard
# -------------------------------------------------------------
print("\n" + "="*90)
print("TOP 5 PAIRS IDENTIFIED BY DCCT & SSD COMBINATION (Gupta & Chatterjee 2020)")
print("="*90)
top_dcct_ssd = df.sort_values("DCCT_and_SSD_Score").head(5)
print(top_dcct_ssd[["Pair", "DCCT_psi25", "SSD", "Rank_DCCT", "Rank_SSD", "DCCT_and_SSD_Score"]])

print("\n" + "="*90)
print("TOP 5 PAIRS IDENTIFIED BY ALIGNED CORRELATION (AC) & COINTEGRATION")
print("="*90)
top_ac_coint = df.sort_values("AC_and_Coint_Score").head(5)
print(top_ac_coint[["Pair", "AC_Distance", "Coint_pVal", "AC_Best_k", "Rank_AC", "Rank_Coint", "AC_and_Coint_Score"]])

# Save full comparison table to CSV
df.to_csv(project_file("pairs_method_comparison_leaderboard.csv"), index=False)
print("\nComplete multi-method comparison table saved to 'pairs_method_comparison_leaderboard.csv'.")