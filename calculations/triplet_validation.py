"""
Step 4: Triplet Validation
Three tests on the 178 significant triplets from triplet_full_significant.csv:

1. Rolling Window Stability  — 252-day windows, check if beta is significant
                               in ≥60% of windows
2. Shuffle / Permutation Test — permute r_A 1000 times, check if real R²
                                is in top 5% of shuffle distribution
3. Out-of-Sample (OOS)        — train 2019-2022, test 2023-2026, record IS/OOS R²

Outputs:
  validated_triplets.csv — all triplets with validation columns appended
"""

from project_paths import project_file

import pandas as pd
import numpy as np
import statsmodels.api as sm

# ── Config ─────────────────────────────────────────────────────────────────────
ROLLING_WINDOW  = 252        # trading days per window
MIN_SIG_PCT     = 0.60       # ≥60% of windows must show p<0.05 for BOTH betas
N_SHUFFLES      = 1000       # permutation test iterations
SHUFFLE_ALPHA   = 0.05       # real R² must beat 95th pct of shuffle distribution
SPLIT_DATE      = "2023-01-01"   # IS: before, OOS: on or after
P_THRESHOLD     = 0.05

# ── Load data ──────────────────────────────────────────────────────────────────
returns_df = pd.read_csv(project_file("nse_log_returns.csv"), index_col=0, parse_dates=True)
sig_df     = pd.read_csv(project_file("triplet_full_significant.csv"))

ALL_STOCKS = list(returns_df.columns)
n          = len(returns_df)
dates      = returns_df.index

print(f"Returns shape : {returns_df.shape}")
print(f"Triplets to validate: {len(sig_df)}")
print(f"IS / OOS split : before / on-or-after {SPLIT_DATE}\n")

returns = {s: returns_df[s].to_numpy() for s in ALL_STOCKS}
dates_arr = dates.to_numpy()

# ── Helpers ────────────────────────────────────────────────────────────────────

def ols_r2_pvals(y, x1, x2):
    """Return (r2, p1, p2) from OLS y ~ 1 + x1 + x2. NaN on failure."""
    mask = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    if mask.sum() < 30:
        return np.nan, np.nan, np.nan
    X = sm.add_constant(np.column_stack([x1[mask], x2[mask]]))
    try:
        res = sm.OLS(y[mask], X).fit()
        return res.rsquared, res.pvalues[1], res.pvalues[2]
    except Exception:
        return np.nan, np.nan, np.nan


def rolling_stability(r_a, r_b, r_c, k, window=ROLLING_WINDOW):
    """
    Slide a window of `window` days across the full series.
    At each step: regress r_C(t+k) ~ r_A(t) + r_B(t).
    Return fraction of windows where BOTH p_beta1 < 0.05 and p_beta2 < 0.05.
    """
    max_start = n - window - k
    if max_start < 1:
        return np.nan

    sig_count = 0
    total     = 0
    for start in range(0, max_start):
        idx = np.arange(start, start + window)
        y   = r_c[idx + k]
        x1  = r_a[idx]
        x2  = r_b[idx]
        _, p1, p2 = ols_r2_pvals(y, x1, x2)
        if np.isnan(p1):
            continue
        total += 1
        if p1 < P_THRESHOLD and p2 < P_THRESHOLD:
            sig_count += 1

    return sig_count / total if total > 0 else np.nan


def shuffle_test(r_a, r_b, r_c, k, n_shuffles=N_SHUFFLES, rng=None):
    """
    Permute r_A randomly n_shuffles times.
    Return p-value = fraction of shuffles with R² ≥ real R².
    Also return the real R².
    """
    if rng is None:
        rng = np.random.default_rng(42)

    t_end = n - k - 1
    idx   = np.arange(0, t_end + 1)
    y     = r_c[idx + k]
    x1    = r_a[idx]
    x2    = r_b[idx]

    real_r2, _, _ = ols_r2_pvals(y, x1, x2)
    if np.isnan(real_r2):
        return np.nan, np.nan

    shuffle_r2s = np.empty(n_shuffles)
    for i in range(n_shuffles):
        x1_shuf = rng.permutation(x1)
        r2_s, _, _ = ols_r2_pvals(y, x1_shuf, x2)
        shuffle_r2s[i] = r2_s if not np.isnan(r2_s) else 0.0

    pval = np.mean(shuffle_r2s >= real_r2)
    return pval, real_r2


def oos_validation(r_a, r_b, r_c, k, split_date=SPLIT_DATE):
    """
    IS: dates < split_date   → fit OLS
    OOS: dates >= split_date → compute R² on held-out data using IS betas.
    Returns (is_r2, oos_r2).
    """
    split_idx = np.searchsorted(dates_arr, np.datetime64(split_date))

    # IS indices
    is_end = split_idx - k - 1
    if is_end < 30:
        return np.nan, np.nan

    is_idx = np.arange(0, is_end + 1)
    y_is   = r_c[is_idx + k]
    x1_is  = r_a[is_idx]
    x2_is  = r_b[is_idx]
    mask_is = np.isfinite(x1_is) & np.isfinite(x2_is) & np.isfinite(y_is)
    if mask_is.sum() < 30:
        return np.nan, np.nan

    X_is = sm.add_constant(np.column_stack([x1_is[mask_is], x2_is[mask_is]]))
    try:
        res_is = sm.OLS(y_is[mask_is], X_is).fit()
    except Exception:
        return np.nan, np.nan

    is_r2  = res_is.rsquared
    params = res_is.params   # [alpha, beta1, beta2]

    # OOS indices
    oos_start = split_idx
    oos_end   = n - k - 1
    if oos_end <= oos_start + 30:
        return is_r2, np.nan

    oos_idx = np.arange(oos_start, oos_end + 1)
    y_oos   = r_c[oos_idx + k]
    x1_oos  = r_a[oos_idx]
    x2_oos  = r_b[oos_idx]
    mask_oos = np.isfinite(x1_oos) & np.isfinite(x2_oos) & np.isfinite(y_oos)
    if mask_oos.sum() < 30:
        return is_r2, np.nan

    y_pred = (params[0]
              + params[1] * x1_oos[mask_oos]
              + params[2] * x2_oos[mask_oos])
    y_act  = y_oos[mask_oos]

    ss_res = np.sum((y_act - y_pred) ** 2)
    ss_tot = np.sum((y_act - y_act.mean()) ** 2)
    oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return is_r2, oos_r2


# ── Main loop ─────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

roll_pcts   = []
shuf_pvals  = []
is_r2s      = []
oos_r2s     = []

total = len(sig_df)
print("Running validation (this takes a few minutes)...")
print(f"{'Row':>5}  {'A':>12}  {'B':>12}  {'C':>12}  {'k':>3}  "
      f"{'Roll%':>6}  {'Shuf_p':>7}  {'IS_R²':>7}  {'OOS_R²':>8}")
print("-" * 80)

for i, row in sig_df.iterrows():
    A = row["Stock_A"]
    B = row["Stock_B"]
    C = row["Stock_C"]
    k = int(row["Horizon_k"])

    r_a = returns[A]
    r_b = returns[B]
    r_c = returns[C]

    # 1. Rolling stability
    roll_pct = rolling_stability(r_a, r_b, r_c, k)

    # 2. Shuffle test
    shuf_p, _ = shuffle_test(r_a, r_b, r_c, k, rng=rng)

    # 3. OOS
    is_r2, oos_r2 = oos_validation(r_a, r_b, r_c, k)

    roll_pcts.append(roll_pct)
    shuf_pvals.append(shuf_p)
    is_r2s.append(is_r2)
    oos_r2s.append(oos_r2)

    if (i + 1) % 20 == 0 or (i + 1) == total:
        rp   = f"{roll_pct:.2f}" if not np.isnan(roll_pct) else "NaN"
        sp   = f"{shuf_p:.3f}"   if not np.isnan(shuf_p)   else "NaN"
        isr  = f"{is_r2:.4f}"   if not np.isnan(is_r2)     else "NaN"
        oosr = f"{oos_r2:.4f}"  if not np.isnan(oos_r2)    else "NaN"
        print(f"{i+1:>5}  {A:>12}  {B:>12}  {C:>12}  {k:>3}  "
              f"{rp:>6}  {sp:>7}  {isr:>7}  {oosr:>8}")

# ── Assemble results ──────────────────────────────────────────────────────────
out_df = sig_df.copy()
out_df["Rolling_Sig_Pct"] = np.round(roll_pcts, 3)
out_df["Shuffle_pval"]    = np.round(shuf_pvals, 4)
out_df["IS_R2"]           = np.round(is_r2s, 4)
out_df["OOS_R2"]          = np.round(oos_r2s, 4)

# ── Verdict ───────────────────────────────────────────────────────────────────
out_df["Stable"]  = out_df["Rolling_Sig_Pct"] >= MIN_SIG_PCT
out_df["Genuine"] = out_df["Shuffle_pval"]    <  SHUFFLE_ALPHA
out_df["OOS_Pos"] = out_df["OOS_R2"]          >  0

# A triplet "passes" if it clears at least 2 of 3 tests
out_df["Tests_Passed"] = (
    out_df["Stable"].astype(int) +
    out_df["Genuine"].astype(int) +
    out_df["OOS_Pos"].astype(int)
)
out_df["Validated"] = out_df["Tests_Passed"] >= 2

out_df = out_df.sort_values(["Validated", "OOS_R2"], ascending=[False, False])
out_df.to_csv(project_file("validated_triplets.csv"), index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"Total triplets tested     : {len(out_df)}")
print(f"Rolling stable (≥60%)     : {out_df['Stable'].sum()}")
print(f"Genuine (shuffle p<0.05)  : {out_df['Genuine'].sum()}")
print(f"OOS R² > 0                : {out_df['OOS_Pos'].sum()}")
print(f"Validated (≥2 of 3 tests) : {out_df['Validated'].sum()}")

validated = out_df[out_df["Validated"]]
if len(validated) > 0:
    print(f"\n=== TOP VALIDATED TRIPLETS BY OOS R² ===")
    cols = ["Stock_A", "Stock_B", "Stock_C", "Horizon_k",
            "R2", "IS_R2", "OOS_R2", "Rolling_Sig_Pct", "Shuffle_pval", "Tests_Passed"]
    print(validated[cols].head(20).to_string(index=False))

    print(f"\n=== VALIDATED TRIPLETS BY HORIZON ===")
    print(validated.groupby("Horizon_k").size().rename("Count"))

    print(f"\n=== MOST FREQUENTLY VALIDATED Stock_C ===")
    print(validated["Stock_C"].value_counts().head(10))
else:
    print("\nNo triplets passed ≥2 tests. Showing best by OOS R²:")
    cols = ["Stock_A", "Stock_B", "Stock_C", "Horizon_k",
            "R2", "IS_R2", "OOS_R2", "Rolling_Sig_Pct", "Shuffle_pval"]
    print(out_df[cols].head(20).to_string(index=False))

print(f"\nSaved → validated_triplets.csv")
