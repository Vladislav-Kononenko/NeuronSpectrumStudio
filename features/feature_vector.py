import numpy as np

def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]

def build_feature_vector(pearson: np.ndarray, spearman: np.ndarray) -> np.ndarray:
    return np.concatenate([
        upper_triangle_values(pearson),
        upper_triangle_values(spearman),
    ])