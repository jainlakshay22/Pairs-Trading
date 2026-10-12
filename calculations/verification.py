import numpy as np
import statsmodels.api as sm
from methods.ac_method import compute_ac

def run_synthetic_verification():
    np.random.seed(42)
    T = 300
    b = 0.8
    a = 0.8
    f = 0.2
    sigma_xi = 1.0
    sigma_eta = f * sigma_xi

    xi = np.random.normal(0, sigma_xi, T + 50)
    eta = np.random.normal(0, sigma_eta, T)

    # Generate AR(1) driver X
    X_raw = np.zeros(T + 50)
    for t in range(1, T + 50):
        X_raw[t] = b * X_raw[t - 1] + xi[t]

    offset = 20
    X = X_raw[offset : offset + T]

    # Piecewise ground-truth lead-lag path g(t) for STS-1
    g_true = np.zeros(T, dtype=int)
    g_true[0:50] = 0
    g_true[50:100] = 5
    g_true[100:150] = 10
    g_true[150:200] = -10
    g_true[200:250] = -5
    g_true[250:300] = 0

    # Follower Y(t) = a * X(t - g(t)) + eta
    Y = np.zeros(T)
    for t in range(T):
        idx = offset + t - g_true[t]
        Y[t] = a * X_raw[idx] + eta[t]

    # Convert to log returns / first differences for AC
    rx = np.diff(X)
    ry = np.diff(Y)

    # Compute AC and alignment path
    ac_dist, best_k, path_p, path_q = compute_ac(rx, ry, k_values=[5, 12, 25])
    print(f"STS-1 Alignment complete: Best k = {best_k}, AC Distance = {ac_dist:.4f}")

    # Map recovered lead-lag path g_hat(t) along Y's index
    n_ret = len(ry)
    g_hat = np.zeros(n_ret)
    for t in range(n_ret):
        closest = np.argmin(np.abs(path_q - t))
        g_hat[t] = path_q[closest] - path_p[closest]

    # Self-Consistency Test (Section 4 of paper: 201 moving windows of size 100)
    sig_count = 0
    a_estimates = []
    total_windows = 300 - 100 + 1  # 201 windows

    for w in range(total_windows):
        y_seg = Y[w : w + 100]
        x_indices = np.clip([w + i - int(g_hat[min(w + i, len(g_hat) - 1)]) for i in range(100)], 0, T - 1)
        x_seg = X[x_indices]

        X_mat = sm.add_constant(x_seg)
        model = sm.OLS(y_seg, X_mat).fit()
        a_hat = model.params[1]
        p_val = model.pvalues[1]

        # 97.5% confidence level -> p_val < 0.025
        if p_val < 0.025:
            sig_count += 1
            a_estimates.append(a_hat)

    pct_sig = (sig_count / total_windows) * 100
    mean_a = np.mean(a_estimates) if a_estimates else 0.0
    print("\n--- Paper Verification Results (Table 1 STS-1) ---")
    print(f"Significant Windows: {sig_count}/{total_windows} ({pct_sig:.2f}%) [Paper Target: 100%]")
    print(f"Mean Estimated 'a': {mean_a:.4f} [Ground Truth: {a}]")

if __name__ == "__main__":
    run_synthetic_verification()