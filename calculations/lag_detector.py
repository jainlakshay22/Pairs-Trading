"""
Step 2: Determine optimal lag for each anchor pair.
Outputs a table: Pair | Optimal Lag (days) | Lagged Correlation | Direction (A leads B or B leads A)
"""

from project_paths import project_file

import pandas as pd
import numpy as np
from methods.correlation_method import compute_max_lagged_corr

# --- Config ---
ANCHOR_PAIRS = [
    ("ONGC.NS",      "LT.NS"),
    ("TCS.NS",       "INFY.NS"),
    ("TCS.NS",       "RELIANCE.NS"),
    ("ICICIBANK.NS", "RELIANCE.NS"),
    ("INFY.NS",      "RELIANCE.NS"),
]
MAX_LAG = 10  # trading days, i.e. 2 weeks

# --- Load returns ---
returns_df = pd.read_csv(project_file("nse_log_returns.csv"), index_col=0, parse_dates=True)

records = []

for stock_a, stock_b in ANCHOR_PAIRS:
    r_a = returns_df[stock_a].to_numpy()
    r_b = returns_df[stock_b].to_numpy()

    lag_corr, opt_lag = compute_max_lagged_corr(r_a, r_b, max_lag=MAX_LAG)

    # Positive lag means A leads B (r_a[:-lag] vs r_b[lag:])
    # Negative lag means B leads A
    if opt_lag > 0:
        direction = f"{stock_a} leads {stock_b} by {opt_lag} day(s)"
        leader, follower = stock_a, stock_b
    elif opt_lag < 0:
        direction = f"{stock_b} leads {stock_a} by {abs(opt_lag)} day(s)"
        leader, follower = stock_b, stock_a
    else:
        direction = "No clear lead-lag (zero lag)"
        leader, follower = stock_a, stock_b

    records.append({
        "Pair":         f"{stock_a} - {stock_b}",
        "Leader":       leader,
        "Follower":     follower,
        "Optimal_Lag":  opt_lag,
        "Lagged_Corr":  round(lag_corr, 4),
        "Direction":    direction,
    })

    print(f"{stock_a} - {stock_b}: lag={opt_lag}, corr={lag_corr:.4f} | {direction}")

lag_df = pd.DataFrame(records)
lag_df.to_csv(project_file("anchor_pair_lags.csv"), index=False)
print("\nSaved to anchor_pair_lags.csv")
print(lag_df[["Pair", "Leader", "Follower", "Optimal_Lag", "Lagged_Corr"]].to_string(index=False))
