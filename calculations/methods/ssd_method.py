import numpy as np

def compute_ssd(norm_price_x: np.ndarray, norm_price_y: np.ndarray) -> float:
    """Computes SSD = (1/n) * sum((x_t - y_t)^2). Lower is better."""
    x = np.asarray(norm_price_x, dtype=np.float64)
    y = np.asarray(norm_price_y, dtype=np.float64)
    return float(np.mean((x - y) ** 2))