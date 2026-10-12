import numpy as np

def compute_dcct(returns_x: np.ndarray, returns_y: np.ndarray, p: int = 25, psi: int = 25) -> float:
    """
    Computes DCCT similarity measure.
    returns_x, returns_y: Daily returns series of length N
    p: Window size parameter (segment length p)
    psi: Post-suffix invariant endpoint relaxation parameter
    """
    rx = np.asarray(returns_x, dtype=np.float64)
    ry = np.asarray(returns_y, dtype=np.float64)
    n = len(rx)
    
    # Calculate local correlation cost matrix between segment of length p
    num_steps = n - p + 1
    if num_steps <= 0:
        return 0.0
        
    CR = np.zeros((num_steps, num_steps), dtype=np.float64)
    for i in range(num_steps):
        sx = rx[i : i + p]
        norm_sx = np.linalg.norm(sx)
        for j in range(num_steps):
            sy = ry[j : j + p]
            norm_sy = np.linalg.norm(sy)
            if norm_sx == 0.0 or norm_sy == 0.0:
                CR[i, j] = 0.0
            else:
                CR[i, j] = np.clip(np.dot(sx, sy) / (norm_sx * norm_sy), -1.0, 1.0)
                
    # Dynamic Programming to find maximum correlation accumulation path
    D = np.full((num_steps, num_steps), -np.inf, dtype=np.float64)
    
    # Boundary relaxation up to psi
    init_bound = min(psi, num_steps)
    for i in range(init_bound):
        for j in range(init_bound):
            D[i, j] = CR[i, j]
            
    for i in range(num_steps):
        for j in range(num_steps):
            if i == 0 and j == 0:
                continue
            prev_max = -np.inf
            if i > 0:
                prev_max = max(prev_max, D[i - 1, j])
            if j > 0:
                prev_max = max(prev_max, D[i, j - 1])
            if i > 0 and j > 0:
                prev_max = max(prev_max, D[i - 1, j - 1])
            if prev_max != -np.inf:
                D[i, j] = CR[i, j] + prev_max
                
    # Endpoint relaxation condition (n - psi)
    end_bound_i = max(0, num_steps - psi)
    end_bound_j = max(0, num_steps - psi)
    dcct_score = np.max(D[end_bound_i:, end_bound_j:]) / float(num_steps)
    return float(dcct_score)