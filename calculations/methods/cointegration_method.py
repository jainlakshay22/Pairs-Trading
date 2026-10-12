import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def compute_cointegration(log_p1: np.ndarray, log_p2: np.ndarray) -> tuple[float, float, float]:
    """
    Fits: log(P1) = gamma * log(P2) + mu + spread_t
    Runs ADF test on spread_t.
    
    Returns:
        (adf_statistic, p_value, hedge_ratio_gamma)
        Lower p-value (< 0.05) indicates strong stationarity/cointegration.
    """
    y = np.asarray(log_p1, dtype=np.float64)
    x = np.asarray(log_p2, dtype=np.float64)
    X = sm.add_constant(x)
    
    model = sm.OLS(y, X).fit()
    gamma = float(model.params[1])
    spread = model.resid
    
    adf_result = adfuller(spread, autolag="AIC")
    adf_stat = float(adf_result[0])
    p_val = float(adf_result[1])
    
    return adf_stat, p_val, gamma