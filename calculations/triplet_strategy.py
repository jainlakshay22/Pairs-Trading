"""
Step 6: Trading Strategy Backtest on Validated Triplets
========================================================
Signal: when |r_A(t)| > 1 sigma AND |r_B(t)| > 0.5 sigma,
        take a position in C in the predicted direction.
        Hold for k days. Exit after k days.

Also computes IS vs OOS performance separately.

Transaction cost: 10 bps per side (20 bps round trip).
"""

from project_paths import project_file

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
TC_BPS        = 0.0010   # 10 bps per trade (one side); 20 bps round-trip
SIGNAL_THRESH = 1.0      # |z-score of r_A| threshold to enter trade
SIGNAL_THRESH_B = 0.5    # |z-score of r_B| threshold (softer for second anchor)
SPLIT_DATE    = "2023-01-01"

# ── Load data ──────────────────────────────────────────────────────────────────
returns_df = pd.read_csv(project_file("nse_log_returns.csv"), index_col=0, parse_dates=True)
validated  = pd.read_csv(project_file("validated_triplets.csv"))
validated  = validated[validated["Validated"] == True].reset_index(drop=True)

dates  = returns_df.index
n      = len(returns_df)
returns = {s: returns_df[s].to_numpy() for s in returns_df.columns}
dates_arr = dates.to_numpy()

split_idx = np.searchsorted(dates_arr, np.datetime64(SPLIT_DATE))

print(f"Universe : {returns_df.shape[1]} stocks, {n} days")
print(f"Validated triplets to backtest: {len(validated)}")
print(f"IS: days 0-{split_idx-1} | OOS: days {split_idx}-{n-1}")
print(f"Transaction cost: {TC_BPS*10000:.0f} bps per side\n")


def compute_metrics(pnl_series):
    """Daily PnL series → Sharpe, MaxDD, WinRate, TotalReturn."""
    if len(pnl_series) == 0 or pnl_series.std() == 0:
        return dict(sharpe=np.nan, max_dd=np.nan, win_rate=np.nan,
                    total_ret=np.nan, n_trades=0)
    sharpe    = pnl_series.mean() / pnl_series.std() * np.sqrt(252)
    cum       = pnl_series.cumsum()
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    max_dd    = drawdowns.min()
    trades    = pnl_series[pnl_series != 0]
    win_rate  = (trades > 0).mean() if len(trades) > 0 else np.nan
    total_ret = pnl_series.sum()
    return dict(sharpe=sharpe, max_dd=max_dd, win_rate=win_rate,
                total_ret=total_ret, n_trades=len(trades))


def backtest_triplet(row, idx_start=0, idx_end=None):
    """
    Backtest one triplet over [idx_start, idx_end).
    Returns dict of performance metrics.
    """
    A = row["Stock_A"]
    B = row["Stock_B"]
    C = row["Stock_C"]
    k = int(row["Horizon_k"])
    b1 = row["Beta1_A"]
    b2 = row["Beta2_B"]

    if idx_end is None:
        idx_end = n

    r_a = returns[A]
    r_b = returns[B]
    r_c = returns[C]

    # Rolling z-score parameters (use IS data to calibrate)
    sigma_a = np.nanstd(r_a[:split_idx])
    sigma_b = np.nanstd(r_b[:split_idx])
    if sigma_a == 0 or sigma_b == 0:
        return None

    daily_pnl = []
    in_trade  = False
    entry_day = -999
    position  = 0.0   # +1 long C, -1 short C

    for t in range(idx_start, idx_end - k):
        za = r_a[t] / sigma_a
        zb = r_b[t] / sigma_b

        if np.isnan(za) or np.isnan(zb):
            daily_pnl.append(0.0)
            continue

        # Signal: both anchors need to move meaningfully
        a_signal = abs(za) >= SIGNAL_THRESH
        b_signal = abs(zb) >= SIGNAL_THRESH_B

        if a_signal and b_signal and not in_trade:
            # Predicted direction of C at t+k
            # E[r_C(t+k)] ~ alpha + b1*r_A(t) + b2*r_B(t)
            predicted = b1 * r_a[t] + b2 * r_b[t]
            if abs(predicted) < 1e-8:
                daily_pnl.append(0.0)
                continue

            position  = np.sign(predicted)   # +1 or -1
            in_trade  = True
            entry_day = t
            # Entry cost
            entry_cost = TC_BPS
            daily_pnl.append(-entry_cost)
        elif in_trade and (t - entry_day) >= k:
            # Exit: realize the return over k days
            r_c_kday = r_c[entry_day + k] if (entry_day + k) < n else 0.0
            if np.isnan(r_c_kday):
                r_c_kday = 0.0
            realized = position * r_c_kday - TC_BPS  # exit cost
            daily_pnl.append(realized)
            in_trade = False
            position = 0.0
        else:
            daily_pnl.append(0.0)

    return np.array(daily_pnl)


# ── Run backtests ──────────────────────────────────────────────────────────────
results = []

for i, row in validated.iterrows():
    A, B, C = row["Stock_A"], row["Stock_B"], row["Stock_C"]
    k = int(row["Horizon_k"])

    # Full backtest
    pnl_full = backtest_triplet(row, idx_start=0)
    if pnl_full is None:
        continue

    # IS only
    pnl_is  = backtest_triplet(row, idx_start=0,          idx_end=split_idx)
    # OOS only
    pnl_oos = backtest_triplet(row, idx_start=split_idx)

    m_full = compute_metrics(pnl_full)
    m_is   = compute_metrics(pnl_is)  if pnl_is  is not None else {}
    m_oos  = compute_metrics(pnl_oos) if pnl_oos is not None else {}

    results.append({
        "Stock_A":        A,
        "Stock_B":        B,
        "Stock_C":        C,
        "Horizon_k":      k,
        "Beta1_A":        row["Beta1_A"],
        "Beta2_B":        row["Beta2_B"],
        "IS_R2":          row["IS_R2"],
        "OOS_R2":         row["OOS_R2"],
        "Shuffle_pval":   row["Shuffle_pval"],
        "Rolling_Pct":    row["Rolling_Sig_Pct"],
        # Full
        "Sharpe_Full":    round(m_full["sharpe"],   3),
        "MaxDD_Full":     round(m_full["max_dd"],   4),
        "WinRate_Full":   round(m_full["win_rate"], 3),
        "NTrades_Full":   m_full["n_trades"],
        "TotRet_Full":    round(m_full["total_ret"],4),
        # IS
        "Sharpe_IS":      round(m_is.get("sharpe",  np.nan), 3),
        "MaxDD_IS":       round(m_is.get("max_dd",  np.nan), 4),
        "WinRate_IS":     round(m_is.get("win_rate",np.nan), 3),
        "NTrades_IS":     m_is.get("n_trades", 0),
        # OOS
        "Sharpe_OOS":     round(m_oos.get("sharpe",  np.nan), 3),
        "MaxDD_OOS":      round(m_oos.get("max_dd",  np.nan), 4),
        "WinRate_OOS":    round(m_oos.get("win_rate",np.nan), 3),
        "NTrades_OOS":    m_oos.get("n_trades", 0),
    })

strat_df = pd.DataFrame(results)
strat_df = strat_df.sort_values("Sharpe_OOS", ascending=False).reset_index(drop=True)
strat_df.to_csv(project_file("triplet_strategy_results.csv"), index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 70)
print("STRATEGY PERFORMANCE SUMMARY")
print("=" * 70)
print(f"\nTriplets backtested: {len(strat_df)}")
print(f"Sharpe_OOS > 0     : {(strat_df['Sharpe_OOS'] > 0).sum()}")
print(f"Sharpe_OOS > 0.5   : {(strat_df['Sharpe_OOS'] > 0.5).sum()}")
print(f"Sharpe_OOS > 1.0   : {(strat_df['Sharpe_OOS'] > 1.0).sum()}")

print("\n=== TOP 15 TRIPLETS BY OOS SHARPE ===")
cols = ["Stock_A","Stock_B","Stock_C","Horizon_k",
        "IS_R2","OOS_R2","Sharpe_IS","Sharpe_OOS","MaxDD_OOS","WinRate_OOS","NTrades_OOS"]
print(strat_df[cols].head(15).to_string(index=False))

print("\n=== FULL RESULTS ===")
print(strat_df[cols].to_string(index=False))
print(f"\nSaved → triplet_strategy_results.csv")
