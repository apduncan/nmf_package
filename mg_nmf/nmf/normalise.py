"""Methods to normalise WH from matrix decomposition X ~ WH. Based on method in Yang and Seoighe, ‘Impact of the Choice
 of Normalization Method on Molecular Cancer Class Discovery Using Nonnegative Matrix Factorization’."""

import pandas as pd
import numpy as np
from typing import Tuple, Callable, Optional

def map_maximum(W: pd.DataFrame, xarg: float = None) -> pd.DataFrame:
    """Map W to a vector of the maximum of each column."""
    return W.max(axis=0)

def map_quantile(W: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """Map W to a vector for the <quantile> quantile of each column."""
    return W.quantile(quantile, axis=0)

def normalise(N: pd.DataFrame, W: Optional[pd.DataFrame], H: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Given a mapping of column Wj to some value Nj, normalise WH by N. Set W or H to none to skip normalising."""
    # Make D, a diagonal matrix whose j-th diagonal entry Dj = Nj
    D: np.ndarray = np.diag(N)
    H_norm: pd.DataFrame = None
    if H is not None:
        H_norm: pd.DataFrame = D.dot(H)
    W_norm: pd.DataFrame = None
    if W is not None:
        W_norm: pd.DataFrame = W.dot(np.linalg.inv(D))
    return (W_norm, H_norm)

def difference_filter(W_norm: pd.DataFrame, T: float = 0.5) -> pd.DataFrame:
    """Filter potentially insignificant metagenes based on the difference between min/max in samples. T is proportion
    of genes to filter out. W should have normalised using one of the normalisation methods."""
    u: pd.DataFrame = W_norm.max(axis=1) - W_norm.min(axis=1)
    u_quantile = u.quantile(T)
    # Filter u to contain only those entries < u_quantile
    u = u[u > u_quantile]
    return W_norm.loc[u.index]

def variance_filter(W_norm: pd.DataFrame, T: float = 0.5) -> pd.DataFrame:
    """Filter potentially insignificant metagenes based on the variance among samples. T is proportion to discard, from 0
    to 1. W should have been normalised using one of the normalisation methods."""
    u: pd.DataFrame = pd.DataFrame(W_norm.var(axis=1))
    u_quantile = u.quantile(T)
    u = u[u[0] > u_quantile[0]]
    return W_norm.loc[u.index]

if __name__ == "__main__":
    # Lazy tests
    test_w: pd.DataFrame = pd.DataFrame([[0, 2, 0], [1, 1, 3], [6, 7, 8]])
    test_h: pd.DataFrame = pd.DataFrame([[1, 2], [3, 4], [5, 6]])
    Wprime, Hprime = normalise(map_maximum(test_w), test_w, test_h)
    print(test_w.dot(test_h))
    print(Wprime.dot(Hprime))
    Wprime, Hprime = normalise(map_quantile(test_w, 0.95), test_w, test_h)
    print(test_w.dot(test_h))
    print(Wprime.dot(Hprime))
    # Try filtering
    print(variance_filter(Wprime))

