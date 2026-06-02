import numpy as np
from scipy.stats import rankdata

def spearman_matrix(window: np.ndarray) -> np.ndarray:
    ranked = np.apply_along_axis(rankdata, 1, window)
    return np.corrcoef(ranked)