"""Generate synthetic data with a known number of components. Allows generation of overlapping block-diagonal clusters
with some ubiquitous features. Also make a dense structure by randomly populating W & H.
Adapted from methods used in Muzzarelli et al., ‘Rank Selection in Non-Negative Matrix Factorization’,
https://doi.org/10.1109/IJCNN.2019.8852146.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Callable, Optional, List, Union, Any

def sparse_overlap(size: Tuple[int, int], rank: int, n_overlap: float = 0.0, m_overlap: float = 0.0,
                   p_ubiq: float = 0.0, noise: Optional[Tuple[float, float]] = (0, 1),
                   w_fill: Callable[[int, int], np.ndarray] = np.random.rand,
                   h_fill: Callable[[int], np.ndarray] = lambda x: np.ones(x)) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                                        pd.DataFrame]:
    """
    Create data with a sparse underlying structure. Will have a block diagonal structure, with a portion of features
    and observations overlapping based on m_overlap and n_overlap parameters. Can add some ubiquitous features which
    have weight in all components.

    :param size: Size of desired matrix, m x n.
    :type size: Tuple[int, int]
    :param rank: Number of components to create in underlying structure.
    :type rank: int
    :param n_overlap: Proportion of observations to occur in more than one component. 0 - 1.
    :type n_overlap: float
    :param m_overlap: Proportion of non-ubiquitous features to have weight in more than one component. 0 - 1.
    :type m_overlap: float
    :param p_ubiq: Proportion of features to be ubiqutious, having weight in all components. 0 - 1.
    :type p_ubiq: float
    :param noise: Normally distributed noise to be apply to X matrix, in form (mean, sd)
    :type noise: Tuple[float, float]
    :param w_fill: Method to generate values to fill blocks in W. Default to uniform
        random between 0 - 1.
    :type w_fill: Callable[[int], np.ndarray]
    :param h_fill: Method to generate values to fill block in H. Default to filling with 1.
    :type h_fill: Callable[[int], np.ndarray]

    :return: Matrices X, W, H. X is the synthetic data matrix, W and H are the matrics forming it's decomposition.
    :rtype: Tuple[DataFrame, DataFrame, DataFrame]
    """

    m, n = size
    H: pd.DataFrame = _create_overlap((rank, n), n_overlap, h_fill)
    # Make non-ubiquitous part of W
    n_ubiq: int = int(m * p_ubiq)
    non_ubiq = m - n_ubiq
    W: pd.DataFrame = _create_overlap((rank, non_ubiq), m_overlap, w_fill).T
    # Make ubiquitous part of W
    W_ubiq: pd.DataFrame = pd.DataFrame(w_fill(n_ubiq, rank), columns=W.columns, index=['ubiq'] * n_ubiq)
    W = W.append(W_ubiq)#, axis=0)
    # Scale W by 10
    W = W * 10

    # Make X
    X: pd.DataFrame = pd.DataFrame(W.dot(H), columns=H.columns, index=W.index)

    #Apply noise
    if noise is not None:
        mean, sd = noise
        X = _apply_normal_noise(X, mean, sd)

    return X, W, H

def _create_overlap(size: Tuple[int, int], overlap: float, fill: Callable[[int], np.ndarray]):
    """Create a matrix of size m x n, with a number of diagonal blocks where a proportion of them overlap.

    size: Tuple[int, int] - Size of matrix, m x n. m will be used as the number of components.
    overlap: float - Proportion of the columns which participate in overlapping blocks. 0 -1.
    """
    # Create matrix
    m, n = size
    matrix: np.ndarray = np.zeros(size)
    # Determine number of columns in H which will participate in multiple components
    n_overlapping: int = int(n * overlap)
    # Number of samples to have weights for each component k
    n_per_k: int = int(np.floor(n / m))
    # Number of columns n in each overlap
    n_per_overlap: int = int(n_overlapping / (m - 1))

    # Embed blocks
    n_labels: List[str] = [''] * n
    for k in range(m):
        n_start = k * n_per_k
        n_end = min(n, n_start + n_per_k + n_per_overlap)
        n_labels[n_start:n_end] = [f'c{str(k)}'] * (n_end - n_start)
        if k > 0:
            n_labels[n_start:n_start + n_per_overlap] = [f'c{str(k - 1)}|c{str(k)}'] * n_per_overlap
        fill_arr: Union[np.ndarray, List] = fill(n_end - n_start)
        # Flatten an ndarray before use
        if isinstance(fill_arr, np.ndarray):
            fill_arr = fill_arr.flatten()
        matrix[k][n_start:n_end] = fill_arr

    c_labels = [f'c{x}' for x in range(m)]

    return pd.DataFrame(matrix, columns=n_labels, index=c_labels)

def _apply_normal_noise(matrix: np.ndarray, mu: float, sigma: float, scale_coeff: np.ndarray) -> np.ndarray:
    """Apply normally distributed noise to a matrix.

    :param matrix: Data to apply noise to
    :type matrix: ndarray
    :param mu: Mean of distribution to draw noise from
    :type mu: float
    :param sigma: Standard deviation of distribution to draw noise from
    :type sigma: float
    :return: matrix with noise added
    :rtype: ndarray
    """
    noise = np.random.normal(mu, sigma, matrix.shape)
    noise = np.multiply(noise, scale_coeff)
    # Scale noise to the scale coefficient of each row
    matrix = matrix + noise
    # Zero out any negative values
    matrix = np.where(matrix < 0, 0, matrix)
    return matrix

def __defualt_row_scale(size: int) -> float:
    return np.random.uniform(1, 10, size)

def sparse_overlap_even(size: Tuple[int, int], rank: int, n_overlap: float = 0.0, m_overlap: float = 0.0,
                   p_ubiq: float = 0.0, noise: Optional[Tuple[float, float]] = (0, 1),
                   fill: Callable[[int, int], np.ndarray] = np.random.rand,
                    feature_scale: Union[bool, Callable[[], np.array]] = True) -> pd.DataFrame:
    """
    Create data with a sparse underlying structure. Will have a block diagonal structure, with a portion of features
    and observations overlapping based on m_overlap and n_overlap parameters. Can add some ubiquitous features which
    have weight in all components. Overlapping blocks will not have higher values in the results X matrix in this
    implementation, where they will in sparse_overlap.

    :param size: Size of desired matrix, m x n.
    :type size: Tuple[int, int]
    :param rank: Number of components to create in underlying structure.
    :type rank: int
    :param n_overlap: Proportion of observations to occur in more than one component. 0 - 1.
    :type n_overlap: float
    :param m_overlap: Proportion of non-ubiquitous features to have weight in more than one component. 0 - 1.
    :type m_overlap: float
    :param p_ubiq: Proportion of features to be ubiqutious, having weight in all components. 0 - 1.
    :type p_ubiq: float
    :param noise: Normally distributed noise to be apply to X matrix, in form (mean, sd)
    :type noise: Tuple[float, float]
    :param w_fill: Method to generate values to fill blocks in W. Default to uniform random
        between 0 - 1.
    :type w_fill: Callable[[int], np.ndarray]
    :param h_fill Method to generate values to fill block in H. Default to filling with 1.
    :type h_fill: Callable[[int], np.ndarray] -
    :param feature_scale: Scale each feature (row) by a value n, so row * n. Default to uniform random values between
                            1 and 10. False if no scaling desired. Can provide a method for custom scaling.
    :return: X matrix with the specified properties.
    :rtype: DataFrame
    """

    m, n  = size
    # Make non-ubiquitous part of matrix
    n_ubiq: int = int(m * p_ubiq)
    non_ubiq = m - n_ubiq
    matrix: np.ndarray = np.zeros((non_ubiq, n))
    # Determine how many rows in a component
    m_per_k = int(np.floor(non_ubiq / rank))
    # Determine how many columns in a component
    n_per_k = int(np.floor(n / rank))
    # Determine number of row / col per overlap
    m_per_over: int = int((non_ubiq * m_overlap) / (rank - 1))
    n_per_over: int = int((n * n_overlap) / (rank - 1))

    # Embed blocks
    m_labels: List[str] = [''] * m
    n_labels: List[str] = [''] * n
    for k in range(rank):
        # Column positions & labels
        n_start = k * n_per_k
        n_end = min(n, n_start + n_per_k + n_per_over)
        # Handle labelling
        n_labels[n_start:n_end] = [f'c{str(k)}'] * (n_end - n_start)
        if k > 0:
            n_labels[n_start:n_start+n_per_over] = [f'c{str(k-1)}|c{str(k)}'] * n_per_over

        # Row positions & labels
        m_start = k * m_per_k
        m_end = min(non_ubiq, m_start + m_per_k + m_per_over)
        m_labels[m_start:m_end] = [f'c{str(k)}'] * (m_end - m_start)
        if k > 0:
            m_labels[m_start:m_start + m_per_over] = [f'c{str(k - 1)}|c{str(k)}'] * m_per_over

        # Embed block in matrix
        matrix[m_start:m_end, n_start:n_end] = fill(m_end - m_start, n_end - n_start)

    # Make ubiquitous part of W
    matrix = np.append(matrix, fill(n_ubiq, n), axis=0) * 10
    m_labels[non_ubiq:m] = ['ubiq'] * (m - non_ubiq)

    # Scale each feature
    scale_coeff: np.ndarray = None
    m_len, n_len = matrix.shape
    if feature_scale == False:
        scale_coeff = np.ones(m_len)
    else:
        if feature_scale == True:
            feature_scale = __defualt_row_scale
        scale_coeff = feature_scale(m_len)
    scale_coeff = scale_coeff.reshape(-1, 1)
    scale_coeff = np.insert(scale_coeff, [1] * (n_len - 1), scale_coeff[:, [0]], axis=1)
    matrix = np.multiply(matrix, scale_coeff)

    # Apply noise
    if noise is not None:
        mean, sd = noise
        matrix = _apply_normal_noise(matrix, mean, sd, scale_coeff)

    # Convert to labelled DataFrame
    return pd.DataFrame(matrix, index=m_labels, columns=n_labels)

def multipathway(size: Tuple[int, int], rank: int, noise: Optional[Tuple[float, float]], m_prob: List[float],
                 n_prob: List[float], feature_scale: Union[bool, Callable[[], np.array]] = True) -> pd.DataFrame:
    """Create a matrix where each gene is assigned to a cluster probabilistically, and the same for each sample.
    Will create a denser structure than sparse_overlap, with more genes and sample spanning 2+ components."""
    m, n = size
    matrix: np.ndarray = np.ndarray(size)

    # Assign each feature to modules based on m_prob
    m_rnd: pd.DataFrame = pd.DataFrame(np.random.uniform(size=(m, rank)))
    m_thrs: pd.Series = pd.Series(m_prob)
    m_assign = m_rnd.apply(lambda x: x < m_thrs, axis=1)

    # Assign each observation to have modules present based on n_prob
    n_rnd: pd.DataFrame = pd.DataFrame(np.random.uniform(size=(n, rank)))
    n_thrs: pd.Series = pd.Series(n_prob)
    n_assign = n_rnd.apply(lambda x: x < n_thrs, axis=1).T

    # Compase a boolean matrix, of whether feature is present in a given pbservation
    rows: List[pd.Series] = []
    for i in range(m):
        feature: pd.Series = m_assign.iloc[i]
        obs: pd.DataFrame = n_assign[feature]
        rows.append(obs.any())
    presence_tbl: pd.DataFrame = pd.DataFrame(rows)

    # Place values in True cells
    rnd: np.ndarray = np.random.rand(m, n) * 10
    vals = np.multiply(presence_tbl, rnd)

    # Scale each feature
    scale_coeff: np.ndarray = None
    m_len, n_len = matrix.shape
    if feature_scale == False:
        scale_coeff = np.ones(m_len)
    else:
        if feature_scale == True:
            feature_scale = __defualt_row_scale
        scale_coeff = feature_scale(m_len)
    scale_coeff = scale_coeff.reshape(-1, 1)
    scale_coeff = np.insert(scale_coeff, [1] * (n_len - 1), scale_coeff[:, [0]], axis=1)
    vals = np.multiply(vals, scale_coeff)

    # Apply noise
    if noise is not None:
        mean, sd = noise
        vals = _apply_normal_noise(vals, mean, sd, scale_coeff)

    # Compose indices
    m_idx = m_assign.apply(lambda x: '|'.join(('c' + str(y) for y in x[x].index)), axis=1)
    n_idx = n_assign.apply(lambda x: '|'.join(('c' + str(y) for y in x[x].index)), axis=0)
    tbl: pd.DataFrame = pd.DataFrame(vals)
    tbl.index = m_idx
    tbl.columns = n_idx
    return tbl

def dense(size: Tuple[int, int], rank: int, noise: Optional[Tuple[float, float]] = (0, 1),
          fill: Callable[[int, int], np.ndarray] = np.random.rand) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create matrix with a dense underlying structure. Create randomly filled H & W, X = W x H + noise.

    :param size: Matrix size, m x n.
    :type size: Tuple[int, int]
    :param rank: Number of components in underlying structure.
    :type rank: int
    :param noise: Normally distributed noise to apply, in format (mu, sigma)
    :type noise: Tuple[float, float]
    :param fill: Method to fill an array of size with values
    :type fill: Callable[[int, int], np.ndarray]
    :return: Matrices X, W, H. X is the synthetic data matrix, W and H are the matrices forming it's decomposition.
    :type: Tuple[DataFrame, DataFrame, DataFrame]
    """

    m, n = size
    H: pd.DataFrame = pd.DataFrame(fill(rank, n))
    W: pd.DataFrame = pd.DataFrame(fill(m, rank))
    X: pd.DataFrame = pd.DataFrame(W.dot(H))

    if noise is not None:
        mu, sigma = noise
        X = _apply_normal_noise(X, mu, sigma)
    return X, W, H

if __name__ == "__main__":
    # Some testing
    # x = sparse_overlap_even(size=(150, 50), rank=8, n_overlap=0.5, m_overlap=0.3, noise=(0, 1), p_ubiq=0.3)
    # x2, w, h = sparse_overlap(size=(150, 50), rank=8, n_overlap=0.5, m_overlap=0.3, noise=(0, 1), p_ubiq=0.3)
    # x3, w, h = dense(size=(150, 50), rank=8, noise=(0, 1))
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # plt.figure()
    # sns.heatmap(x)
    # plt.show()
    # plt.figure()
    # sns.heatmap(x2)
    # plt.show()
    # plt.figure()
    # sns.heatmap(x3)
    # plt.show()
    #
    # np.random.seed(0)
    # x = sparse_overlap_even(size=(150, 50), rank=8, n_overlap=0.5, m_overlap=0.3, noise=(0, 1), p_ubiq=0.3)
    x = multipathway(size=(500, 150), rank=5, n_prob=[1/3]*5, m_prob=[1/3]*5, noise=(0, 0))
    # np.random.seed(0)
    # x2 = sparse_overlap_even(size=(150, 50), rank=8, n_overlap=0.5, m_overlap=0.3, noise=(0, 1), p_ubiq=0.3)
    # print(x.equals(x2))

    # A small
    foo = 'bar'