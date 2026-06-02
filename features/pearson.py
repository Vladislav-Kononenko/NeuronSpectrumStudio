import numpy as np

def pearson_matrix(window: np.ndarray) -> np.ndarray:
    return np.corrcoef(window)