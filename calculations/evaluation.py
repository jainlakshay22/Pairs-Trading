
from project_paths import project_file
import itertools
import pandas as pd
import numpy as np
from methods.ssd_method import compute_ssd
from methods.correlation_method import compute_pearson_corr, compute_max_lagged_corr
from methods.cointegration_method import compute_cointegration
from methods.dcct_method import compute_dcct
from methods.ac_method import compute_ac

def generate_multi_method_leaderboard():
    norm_df = pd.read_csv(project_file("nse_normalized_prices.csv"), index_col=0, parse_dates=True)
    log_p_df = pd.read_csv(project_file("nse_log_prices.csv"), index_col=0, parse_dates=True)
    returns_df = pd.read_csv(project_file("nse_log_returns.csv"), index_col=0, parse_dates=True)

    candidate_tickers = [
        "TCS.NS", "INFY.NS",
        "HDFCBANK.NS", "ICICIBANK.NS",
        "SUNPHARMA.NS", "CIPLA.NS",
        "RELIANCE.NS", "ONGC.NS",
        "LT.NS", "SIEMENS.NS"
    ]

    pairs = list(itertools.combinations(candidate_tickers, 2))
    records = []

    for stock_a, stock_b in pairs:
        p_norm_a, p_norm_b = norm_df[stock_a].to_numpy(), norm_df[stock_b].to_numpy()
        lp_a, lp_b = log_p_df[stock_a].to_numpy(), log_p_df[stock_b].to_numpy()
        r_a, r_b = returns_df[stock_a].to_numpy(), returns_df[stock_b].to_numpy()

        ssd_val = compute_ssd(p_norm_a, p_norm_b)
        pearson_val = compute_pearson_corr(r_a, r_b)
        lag_corr, opt_lag = compute_max_lagged_corr(r_a, r_b, max_lag=10)
        _, p_val, _ = compute_cointegration(lp_a, lp_b)
        dcct_val = compute_dcct(r_a, r_b, p=25, psi=25)
        ac_dist, best_k, _, _ = compute_ac(r_a, r_b, k_values=[5, 12, 25])

        records.append({
            "Pair": f"{stock_a} - {stock_b}",
            "SSD": ssd_val,
            "Pearson": pearson_val,
            "Lagged": lag_corr,
            "Optimal_Lag": opt_lag,
            "Coint_pVal": p_val,
            "DCCT": dcct_val,
            "AC_Distance": ac_dist,
            "AC_Best_k": best_k
        })

    df = pd.DataFrame(records)

    # Standardized Rankings (1 = Best)
    df["Rank_SSD"] = df["SSD"].rank(ascending=True)
    df["Rank_Pearson"] = df["Pearson"].rank(ascending=False)
    df["Rank_Lagged"] = df["Lagged"].abs().rank(ascending=False)
    df["Rank_Coint"] = df["Coint_pVal"].rank(ascending=True)
    df["Rank_DCCT"] = df["DCCT"].rank(ascending=False)
    df["Rank_AC"] = df["AC_Distance"].rank(ascending=True)

    base_methods = ["SSD", "Pearson", "Lagged", "Coint", "DCCT", "AC"]

    # Generate all 15 pairwise combinations
    for m1, m2 in itertools.combinations(base_methods, 2):
        df[f"{m1}+{m2}"] = (df[f"Rank_{m1}"] + df[f"Rank_{m2}"]) / 2.0

    df.to_csv(project_file("all_method_combinations_leaderboard.csv"), index=False)
    print("Multi-method evaluation complete. Saved to 'all_method_combinations_leaderboard.csv'.")
    return df

if __name__ == "__main__":
    generate_multi_method_leaderboard()