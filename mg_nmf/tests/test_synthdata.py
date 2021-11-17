# Standard library imports
import os
import random
import tempfile
from typing import Dict, Tuple
import unittest

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from mg_nmf.nmf import synthdata

class TestSyntheticData(unittest.TestCase):
    """Test the three methods of creating synthetic data"""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise any needed object or parmeters here. """
        # Currently nothing for this set of test cases
        pass

    def test_sparse_overlap(self):
        """Test synthetic data create by constructing W and H, then X = WH."""

        test_size: Tuple[int, int] = (45, 60)
        test_rank: int = 3
        test_p_ubiq: float = 0.0
        test_noise: Tuple[float, float] = (0.0, 1.0)

        # Test that resetting the random seed produces the same result
        np.random.seed(0)
        X, W, H = synthdata.sparse_overlap(size=test_size, rank=test_rank, p_ubiq=test_p_ubiq, noise=test_noise)
        np.random.seed(0)
        X2, W2, H2 = synthdata.sparse_overlap(size=test_size, rank=test_rank, p_ubiq=test_p_ubiq, noise=test_noise)
        self.assertTrue(np.array_equal(X, X2))

        # Check that dimensions of W and H are as expected
        # W should be <size[0]> high, <rank> wide, H should be <rank> high, <size[1]> wide
        self.assertSequenceEqual((test_size[0], test_rank), W.shape)
        self.assertSequenceEqual((test_rank, test_size[1]), H.shape)

        # Test that custom fill methods function, and provide a different output
        np.random.seed(0)
        Xc, Wc, Hc = synthdata.sparse_overlap(size=test_size, rank=test_rank, p_ubiq=test_p_ubiq, noise=test_noise,
                                              w_fill=lambda x, y=1: np.ones((x,y)))
        self.assertFalse(np.array_equal(W, Wc))
        # Same for h_fill
        np.random.seed(0)
        Xd, Wd, Hd = synthdata.sparse_overlap(size=test_size, rank=test_rank, p_ubiq=test_p_ubiq, noise=test_noise,
                                              h_fill=np.random.rand)
        self.assertFalse(np.array_equal(H, Hd))
        # Can't check X = WH, as X = WH + Noise

    def test_dense(self):
        """Test synthetic data create by constructing W and H, then X = WH."""

        test_size: Tuple[int, int] = (45, 60)
        test_rank: int = 3
        test_noise: Tuple[float, float] = (0.0, 1.0)

        # Test that resetting the random seed produces the same result
        np.random.seed(0)
        X, W, H = synthdata.dense(size=test_size, rank=test_rank, noise=test_noise)
        np.random.seed(0)
        X2, W2, H2 = synthdata.dense(size=test_size, rank=test_rank, noise=test_noise)
        self.assertTrue(np.array_equal(X, X2))

        # Check that dimensions of W and H are as expected
        # W should be <size[0]> high, <rank> wide, H should be <rank> high, <size[1]> wide
        self.assertSequenceEqual((test_size[0], test_rank), W.shape)
        self.assertSequenceEqual((test_rank, test_size[1]), H.shape)

    def test_sparse_even(self):
        """Test synthetic data create by constructing W and H, then X = WH."""

        test_size: Tuple[int, int] = (45, 60)
        test_rank: int = 3
        test_p_ubiq: float = 0.0
        test_noise: Tuple[float, float] = (0.0, 1.0)

        # Test that resetting the random seed produces the same result
        np.random.seed(0)
        X = synthdata.sparse_overlap_even(size=test_size, rank=test_rank, p_ubiq=test_p_ubiq, noise=test_noise)
        np.random.seed(0)
        X2 = synthdata.sparse_overlap_even(size=test_size, rank=test_rank, p_ubiq=test_p_ubiq, noise=test_noise)
        self.assertTrue(np.array_equal(X, X2))

        # Check that dimensions of W and H are as expected
        # W should be <size[0]> high, <rank> wide, H should be <rank> high, <size[1]> wide
        self.assertSequenceEqual(test_size, X.shape)

        # Test that custom fill methods function, and provide a different output
        np.random.seed(0)
        Xc = synthdata.sparse_overlap_even(size=test_size, rank=test_rank, p_ubiq=test_p_ubiq, noise=test_noise,
                                              fill=lambda x, y=1: np.ones((x, y)))
        self.assertFalse(np.array_equal(X, Xc))