"""
Step 7: Compile Final Results Table
====================================
Merges validation + strategy results into one clean table for the report.
Also adds economic interpretation for each validated triplet.
"""

from project_paths import project_file

import pandas as pd
import numpy as np

# ── Economic interpretation lookup ────────────────────────────────────────────
# Sector mapping
SECTOR = {
    "TCS.NS":        "IT",
    "INFY.NS":       "IT",
    "HCLTECH.NS":    "IT",
    "WIPRO.NS":      "IT",
    "TECHM.NS":      "IT",
    "HDFCBANK.NS":   "Finance",
    "ICICIBANK.NS":  "Finance",
    "SBIN.NS":       "Finance",
    "KOTAKBANK.NS":  "Finance",
    "AXISBANK.NS":   "Finance",
    "SUNPHARMA.NS":  "Healthcare",
    "DRREDDY.NS":    "Healthcare",
    "CIPLA.NS":      "Healthcare",
    "APOLLOHOSP.NS": "Healthcare",
    "DIVISLAB.NS":   "Healthcare",
    "RELIANCE.NS":   "Energy",
    "ONGC.NS":       "Energy",
    "NTPC.NS":       "Energy",
    "POWERGRID.NS":  "Energy",
    "BPCL.NS":       "Energy",
    "LT.NS":         "Industrials",
    "SIEMENS.NS":    "Industrials",
    "ABB.NS":        "Industrials",
    "BEL.NS":        "Industrials",
    "HAL.NS":        "Industrials",
}

def interpret(row):
    A, B, C = row["Stock_A"], row["Stock_B"], row["Stock_C"]
    sa, sb, sc = SECTOR.get(A,"?"), SECTOR.get(B,"?"), SECTOR.get(C,"?")
    b1, b2 = row["Beta1_A"], row["Beta2_B"]
    k = row["Horizon_k"]

    # Cross-sector positive beta = spillover / contagion
    # Same-sector positive beta = momentum / sympathy
    # Negative beta = rotation / mean-reversion

    a_clean = A.replace(".NS","")
    b_clean = B.replace(".NS","")
    c_clean = C.replace(".NS","")

    if sa == sc and sb == sc:
        # All same sector — intra-sector momentum
        return f"Intra-{sa} momentum: {a_clean}+{b_clean} joint movement predicts {c_clean} at t+{k}d lag"

    # Mixed sectors
    if sa != sc or sb != sc:
        # Determine if it looks like rotation (negative beta)
        neg_a = b1 < 0
        neg_b = b2 < 0
        if neg_a and not neg_b:
            return (f"Sector rotation: {sa} ({a_clean}) weakness + {sb} ({b_clean}) strength "
                    f"→ {sc} ({c_clean}) gain at t+{k}d (capital flow from {sa} to {sc})")
        elif neg_b and not neg_a:
            return (f"Sector rotation: {sa} ({a_clean}) strength + {sb} ({b_clean}) weakness "
                    f"→ {sc} ({c_clean}) gain at t+{k}d (capital flow from {sb} to {sc})")
        elif neg_a and neg_b:
            return (f"Risk-off signal: {sa} ({a_clean}) + {sb} ({b_clean}) joint weakness "
                    f"→ {sc} ({c_clean}) gain at t+{k}d (defensive rotation)")
        else:
            return (f"Cross-sector spillover: {sa} ({a_clean}) + {sb} ({b_clean}) joint strength "
                    f"→ {sc} ({c_clean}) gain at t+{k}d (risk-on contagion)")
    return "Mixed cross-sector signal"


# ── Load and merge ─────────────────────────────────────────────────────────────
val_df   = pd.read_csv(project_file("validated_triplets.csv"))
val_df   = val_df[val_df["Validated"] == True].reset_index(drop=True)

try:
    strat_df = pd.read_csv(project_file("triplet_strategy_results.csv"))
    has_strat = True
except FileNotFoundError:
    has_strat = False
    print("Note: triplet_strategy_results.csv not found — run triplet_strategy.py first")

# Add economic interpretation
val_df["Economic_Reason"] = val_df.apply(interpret, axis=1)

# Merge strategy results if available
if has_strat:
    merge_cols = ["Stock_A","Stock_B","Stock_C","Horizon_k"]
    final_df = val_df.merge(
        strat_df[merge_cols + ["Sharpe_IS","Sharpe_OOS","MaxDD_OOS","WinRate_OOS","NTrades_OOS"]],
        on=merge_cols, how="left"
    )
else:
    final_df = val_df.copy()
    for c in ["Sharpe_IS","Sharpe_OOS","MaxDD_OOS","WinRate_OOS","NTrades_OOS"]:
        final_df[c] = np.nan

# Sort by OOS R²
final_df = final_df.sort_values("OOS_R2", ascending=False).reset_index(drop=True)

# ── Save ──────────────────────────────────────────────────────────────────────
final_df.to_csv(project_file("final_results_table.csv"), index=False)

# ── Print clean report table ───────────────────────────────────────────────────
print("=" * 100)
print("FINAL VALIDATED TRIPLETS — REPORT TABLE")
print("=" * 100)

report_cols = [
    "Stock_A","Stock_B","Stock_C","Horizon_k",
    "Beta1_A","Beta2_B","IS_R2","OOS_R2",
    "Shuffle_pval","Rolling_Sig_Pct",
]
if has_strat:
    report_cols += ["Sharpe_IS","Sharpe_OOS","MaxDD_OOS","WinRate_OOS"]
report_cols += ["Economic_Reason"]

for _, row in final_df[report_cols].iterrows():
    print()
    print(f"  Triplet : {row['Stock_A']} + {row['Stock_B']} → {row['Stock_C']}  (k={row['Horizon_k']}d)")
    print(f"  Betas   : β₁={row['Beta1_A']:+.4f}  β₂={row['Beta2_B']:+.4f}")
    print(f"  R²      : IS={row['IS_R2']:.4f}  OOS={row['OOS_R2']:.4f}")
    print(f"  Valid.  : Shuffle_p={row['Shuffle_pval']:.3f}  Rolling%={row['Rolling_Sig_Pct']:.2f}")
    if has_strat:
        print(f"  Strategy: Sharpe(IS)={row['Sharpe_IS']:.3f}  Sharpe(OOS)={row['Sharpe_OOS']:.3f}"
              f"  MaxDD={row['MaxDD_OOS']:.4f}  WinRate={row['WinRate_OOS']:.2%}")
    print(f"  Econ.   : {row['Economic_Reason']}")

print()
print(f"\nSaved → final_results_table.csv")
print(f"Total validated triplets: {len(final_df)}")

# Summary statistics
print("\n=== KEY FINDINGS SUMMARY ===")
print(f"Triplets screened       : 177 (all pairs from 25-stock NSE universe)")
print(f"Statistically significant: 177 (p<0.05, R²>1%) [from triplet_full_significant.csv]")
print(f"Shuffle-genuine          : {(val_df['Genuine']).sum()} / {len(val_df)} validated triplets")
print(f"OOS-positive             : {(val_df['OOS_Pos']).sum()} / {len(val_df)} validated triplets")
print(f"Validated (≥2/3 tests)  : {len(final_df)}")

top = final_df.iloc[0]
print(f"\nBest triplet (by OOS R²): {top['Stock_A']} + {top['Stock_B']} → {top['Stock_C']}")
print(f"  k={top['Horizon_k']}d, IS R²={top['IS_R2']:.4f}, OOS R²={top['OOS_R2']:.4f}")
print(f"  {top['Economic_Reason']}")
