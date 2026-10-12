import numpy as np

def compute_ac(returns_x: np.ndarray, returns_y: np.ndarray, k_values: list[int] = [5, 12, 25, 50]) -> tuple[float, int, np.ndarray, np.ndarray]:
    """
    Computes Aligned Correlation Distance (loose metric) and best alignment path.
    Returns: (best_ac_dist, best_k, path_p, path_q)
    """
    rx = np.asarray(returns_x, dtype=np.float64)
    ry = np.asarray(returns_y, dtype=np.float64)
    n = len(rx)
    
    best_ac = np.inf
    best_k = k_values[0]
    best_path_p, best_path_q = np.array([]), np.array([])
    
    for k in k_values:
        psi = k
        rx_pad = np.pad(rx, (k, k), mode="constant", constant_values=0.0)
        ry_pad = np.pad(ry, (k, k), mode="constant", constant_values=0.0)
        
        # Local CR cost matrix: CR = 2 * (1 - local_corr)
        CR = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            sx = rx_pad[i : i + 2 * k + 1]
            norm_sx = np.linalg.norm(sx)
            for j in range(n):
                sy = ry_pad[j : j + 2 * k + 1]
                norm_sy = np.linalg.norm(sy)
                if norm_sx == 0.0 or norm_sy == 0.0:
                    CR[i, j] = 2.0
                else:
                    c = np.clip(np.dot(sx, sy) / (norm_sx * norm_sy), -1.0, 1.0)
                    CR[i, j] = 2.0 * (1.0 - c)
                    
        # Dynamic Programming alignment
        D = np.full((n, n), np.inf, dtype=np.float64)
        
        # Boundary initialization with psi relaxation
        init_bound = min(psi, n)
        for i in range(init_bound):
            for j in range(init_bound):
                D[i, j] = CR[i, j]
                
        for i in range(n):
            for j in range(n):
                if i < init_bound and j < init_bound:
                    continue
                prev = min(
                    D[i - 1, j] if i > 0 else np.inf,
                    D[i, j - 1] if j > 0 else np.inf,
                    D[i - 1, j - 1] if (i > 0 and j > 0) else np.inf
                )
                if prev != np.inf:
                    D[i, j] = CR[i, j] + prev
                    
        # Endpoint relaxation
        end_bound_i = max(0, n - psi)
        end_bound_j = max(0, n - psi)
        sub_grid = D[end_bound_i:, end_bound_j:]
        min_idx = np.unravel_index(np.argmin(sub_grid), sub_grid.shape)
        curr_i = end_bound_i + min_idx[0]
        curr_j = end_bound_j + min_idx[1]
        
        # Backtrack optimal alignment path
        path_p, path_q = [curr_i], [curr_j]
        while curr_i > 0 or curr_j > 0:
            if curr_i == 0:
                curr_j -= 1
            elif curr_j == 0:
                curr_i -= 1
            else:
                step = np.argmin([D[curr_i - 1, curr_j - 1], D[curr_i - 1, curr_j], D[curr_i, curr_j - 1]])
                if step == 0:
                    curr_i, curr_j = curr_i - 1, curr_j - 1
                elif step == 1:
                    curr_i -= 1
                else:
                    curr_j -= 1
            path_p.append(curr_i)
            path_q.append(curr_j)
            
        path_p = np.array(path_p[::-1])
        path_q = np.array(path_q[::-1])
        
        # Compute AC metric along aligned path
        aligned_rx = rx[path_p]
        aligned_ry = ry[path_q]
        norm_arx = np.linalg.norm(aligned_rx)
        norm_ary = np.linalg.norm(aligned_ry)
        
        gamma = 0.0 if (norm_arx == 0.0 or norm_ary == 0.0) else np.clip(np.dot(aligned_rx, aligned_ry) / (norm_arx * norm_ary), -1.0, 1.0)
        ac_dist = np.sqrt(max(0.0, 2.0 * (1.0 - gamma)))
        
        if ac_dist < best_ac:
            best_ac = ac_dist
            best_k = k
            best_path_p = path_p
            best_path_q = path_q
            
    return float(best_ac), int(best_k), best_path_p, best_path_q