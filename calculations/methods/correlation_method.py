import numpy as np

def compute_pearson_corr(series_x: np.ndarray, series_y: np.ndarray) -> float:
    """Computes zero-lag Pearson correlation in [-1, +1]."""
    x = np.asarray(series_x, dtype=np.float64)
    y = np.asarray(series_y, dtype=np.float64)
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    norm_x = np.linalg.norm(xc)
    norm_y = np.linalg.norm(yc)
    if norm_x == 0.0 or norm_y == 0.0:
        return 0.0
    return float(np.clip(np.dot(xc, yc) / (norm_x * norm_y), -1.0, 1.0))

def compute_max_lagged_corr(series_x: np.ndarray, series_y: np.ndarray, max_lag: int = 15) -> tuple[float, int]:
    """Finds maximum cross-correlation and optimal lag in [-max_lag, max_lag]."""
    best_corr = 0.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            c = compute_pearson_corr(series_x[:-lag], series_y[lag:])
        elif lag < 0:
            c = compute_pearson_corr(series_x[-lag:], series_y[:lag])
        else:
            c = compute_pearson_corr(series_x, series_y)
        if abs(c) > abs(best_corr):
            best_corr = c
            best_lag = lag
    return float(best_corr), int(best_lag)