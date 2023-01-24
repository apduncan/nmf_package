# Standard library imports
import os
import tempfile
from typing import Dict
import unittest

# Third party imports
import numpy as np
import pandas as pd

# Local imports
from mg_nmf.nmf import selection

# Define a test dataset
TEST_DATA: pd.DataFrame = pd.read_csv('data/k2_testdata.csv', index_col=0)


class TestNMFConsensusSelection(unittest.TestCase):
    """Test consensus based selection methods. NMFModelSelection is abstract, so testing the base functions here."""

    sel: selection.NMFConsensusSelection
    data: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise a selection object for our test class to work with, and load some synthetic data."""

        cls.sel = selection.NMFConsensusSelection(
            data=TEST_DATA, k_min=2, k_max=3, solver='mu', beta_loss='frobenius', iterations=10, nmf_max_iter=1000
        )

    def test_run(self):
        """Perform an NMF run on the simple data, check we get the expected outputs."""

        # Run the selection object we set up in the classSetUp
        coph, disp = self.sel.run()
        # Assert that we get two NMFResults objects
        self.assertIsInstance(coph, selection.NMFResults)
        self.assertIsInstance(disp, selection.NMFResults)

    def test_properties(self):
        """Set properties, and check they get changed or throw exceptions as expected. These are base class properties,
        don't need to be tested in other subclasses."""

        # Create a new object to mess with it's properties
        sel: selection.NMFConsensusSelection = selection.NMFConsensusSelection(
            data=TEST_DATA, k_min=2, k_max=3, solver='mu', beta_loss='frobenius', iterations=10, nmf_max_iter=1000
        )

        # Metrics
        with self.assertRaises(Exception, msg='Setting metric to invalid value should raise exception'):
            sel.metric = 'coph'
        sel.metric = 'cophenetic'
        self.assertEqual(sel.metric, 'cophenetic')

        # Solvers
        with self.assertRaises(Exception, msg='Setting solver to invalid value should raise exception'):
            sel.solver = 'magic'
        sel.solver = 'cd'
        self.assertEqual(sel.solver, 'cd')

        # Beta-loss
        with self.assertRaises(Exception, msg='Setting beta-loss to invalid value should raise exception'):
            sel.beta_loss = 'magic'
        sel.beta_loss = 'kullback-leibler'
        self.assertEqual(sel.beta_loss, 'kullback-leibler')

        # k min/max
        # Should refuse values which are too low
        prev_kmin: int = sel.k_min
        sel.k_min = 1
        self.assertEqual(sel.k_min, prev_kmin)
        sel.k_min = -3
        self.assertEqual(sel.k_min, prev_kmin)
        # Max canot be lower than min
        sel.k_min = 4
        sel.k_max = 3
        # Max cannot be changed to lower than min
        self.assertEqual(sel.k_max, sel.k_min)
        sel.k_max = 10
        self.assertEqual(sel.k_max, 10)
        # Ensure iterations cannot be below 1
        sel.iterations = -100
        self.assertEqual(sel.iterations, 1)
        # Ensure nmf_max_iter cannot be below 0, should reset to default
        sel.nmf_max_iter = 0
        self.assertEqual(sel.nmf_max_iter, sel.DEF_NMF_MAX_ITER)


class TestNMFConcordanceSelection(unittest.TestCase):
    """Test Jiang stability based selection methods."""

    sel: selection.NMFConcordanceSelection
    data: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise a selection object for our test class to work with, and load some synthetic data."""

        cls.sel = selection.NMFConcordanceSelection(
            data=TEST_DATA, k_min=2, k_max=3, solver='mu', beta_loss='frobenius', iterations=10, nmf_max_iter=1000
        )

    def test_run(self):
        """Perform an NMF run on the simple data, check we get the expected outputs."""

        # Run the selection object we set up in the classSetUp
        jiang = self.sel.run()
        # Assert that we get two NMFResults objects
        self.assertIsInstance(jiang, selection.NMFResults)


class TestNMFSplitHalfSelection(unittest.TestCase):
    """Test split half based selection methods."""

    sel: selection.NMFSplitHalfSelection
    data: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise a selection object for our test class to work with, and load some synthetic data."""

        cls.sel = selection.NMFSplitHalfSelection(
            data=TEST_DATA, k_min=2, k_max=3, solver='mu', beta_loss='frobenius', iterations=10, nmf_max_iter=1000
        )

    def test_run(self):
        """Perform an NMF run on the simple data, check we get the expected outputs."""

        # Run the selection object we set up in the classSetUp
        ari = self.sel.run()
        # Assert that we get two NMFResults objects
        self.assertIsInstance(ari, selection.NMFResults)


class TestNMFPermutationSelection(unittest.TestCase):
    """Test permutation based selection methods."""

    sel: selection.NMFPermutationSelection
    data: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise a selection object for our test class to work with, and load some synthetic data."""

        cls.sel = selection.NMFPermutationSelection(
            data=TEST_DATA, k_min=2, k_max=3, solver='mu', beta_loss='frobenius', iterations=10, nmf_max_iter=1000
        )

    def test_run(self):
        """Perform an NMF run on the simple data, check we get the expected outputs."""

        # Run the selection object we set up in the classSetUp
        kprop = self.sel.run()
        # Assert that we get two NMFResults objects
        self.assertIsInstance(kprop, selection.NMFResults)


class TestNMFImputationSelection(unittest.TestCase):
    """Test Jiang stability based selection methods."""

    sel: selection.NMFImputationSelection
    data: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        """Initialise a selection object for our test class to work with, and load some synthetic data."""

        cls.sel = selection.NMFImputationSelection(
            data=TEST_DATA, k_min=2, k_max=3, solver='mu', beta_loss='frobenius', iterations=10, nmf_max_iter=1000,
            p_holdout=0.1, metric='both'
        )

    def test_run(self):
        """Perform an NMF run on the simple data, check we get the expected outputs."""

        # Run the selection object we set up in the classSetUp
        mse, mad = self.sel.run()
        # Assert that we get two NMFResults objects
        self.assertIsInstance(mse, selection.NMFResults)
        self.assertIsInstance(mad, selection.NMFResults)


class TestNMFMultiSelect(unittest.TestCase):
    """Tests for class which allows running of multiple methods of model selection and returns their results."""

    multi: selection.NMFMultiSelect

    @classmethod
    def setUp(cls) -> None:
        """Intialise a model selection object, to use multiple model selection methods."""

        # TEMP - Testing threading
        import os
        os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
        os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
        os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
        os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6
        cls.multi = selection.NMFMultiSelect(
            ranks=(2, 6), iterations=100, nmf_max_iter=1000, solver='mu', beta_loss='frobenius'
        )

    def test_run(self) -> None:
        """Perform model selection using different methods."""

        # Temp - used large test data
        import mg_nmf.nmf.synthdata as synthdata
        large_td = synthdata.sparse_overlap_even((50, 150), 6, 0.1, 0.1, 0.1, (0, 1))
        res: Dict[str, selection.NMFResults] = self.multi.run(large_td, 1)

        # Ensure all methods were run, and returned correct type
        for method in selection.NMFMultiSelect.PERMITTED_METHODS:
            self.assertIn(method, res)
            self.assertIsInstance(res[method], selection.NMFResults)

    def test_properties(self) -> None:
        """Test property changes."""

        # Iterations should throw exception if  < 1
        with self.assertRaises(Exception):
            self.multi.iterations = 0
        with self.assertRaises(Exception):
            self.multi.iterations = -1
        self.multi.iterations = 6
        self.assertEqual(self.multi.iterations, 6)

        # Methods should throw an error if asking for method not in list
        with self.assertRaises(Exception):
            self.multi.methods = ['coph', 'error']
        self.multi.methods = ['coph', 'disp']
        self.assertEqual(self.multi.methods, ['coph', 'disp'])

        # Solver should throw an error if asking for solver not in list
        with self.assertRaises(Exception):
            self.multi.solver = 'error'
        self.multi.solver = 'cd'
        self.assertEqual(self.multi.solver, 'cd')

        # Beta loss should throw an error if asking for function not in list
        with self.assertRaises(Exception):
            self.multi.beta_loss = 'error'
        self.multi.beta_loss = 'kullback-leibler'
        self.assertEqual(self.multi.beta_loss, 'kullback-leibler')


class TestNMFResults(unittest.TestCase):
    """Tests for class which holds results for model selection runs for multiple values of k."""

    sel: selection.NMFConcordanceSelection
    res: selection.NMFResults
    K_FROM: int = 2
    K_TO: int = 4
    ITER: int = 5

    @classmethod
    def setUp(cls) -> None:
        """Run a simple model selection method, to obtain an NMFResults object to work with."""

        cls.sel = selection.NMFConcordanceSelection(
            k_min=cls.K_FROM, k_max=cls.K_TO, iterations=cls.ITER, nmf_max_iter=1000, solver='mu',
            beta_loss='frobenius', data=TEST_DATA
        )
        cls.res = cls.sel.run()

    def test_properties(self) -> None:
        """Ensure properties behave as expected."""

        # Correct number of results returned
        self.assertEqual(len(self.res.results), self.K_TO - self.K_FROM + 1)
        # Check that the selected result does actually have optimal value for selection criteria
        top_metric: float = -np.inf
        top_res: selection.NMFModelSelectionResults
        for res in self.res.results:
            if res.metric >= top_metric:
                top_metric = res.metric
                top_k = res
        self.assertIs(self.res.selected, top_k)

    def test_cophenetic_table(self) -> None:
        """Make sure the table of cophenetic correlation comes out as expected."""

        c_tbl: pd.DataFrame = self.res.measure(None, None)
        # Test has expected length
        self.assertEqual(len(c_tbl.columns), len(self.res.results))
        # Test each value in the table, check matches it's table entry
        for i in range(0, len(self.res.results)):
            res_val: float = self.res.results[i].metric
            tbl_val: float = float(c_tbl.iloc[1,i])
            self.assertAlmostEqual(res_val, tbl_val, places=7)

        # Test both plot and file can be written
        with tempfile.TemporaryDirectory() as tmpdirname:
            plot_f, table_f = os.path.join(tmpdirname, 'tmp_plot.png'), os.path.join(tmpdirname, 'tmp_tbl.csv')
            self.res.measure(plot_f, table_f)
            self.assertTrue(os.path.exists(plot_f))
            self.assertTrue(os.path.exists(table_f))


# TODO: NMFModelSelectionResults, Connectivity Matrix


if __name__ == '__main__':
    unittest.main()
