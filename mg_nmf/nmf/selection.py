"""Perform Non-Negative Matrix Factorization model selection.
Initially designed for use on """

# Standard library imports
from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import reduce
import multiprocessing
import random
import time
from typing import List, Tuple, Dict, Callable, Union, Optional, Iterable, NamedTuple
import warnings

# Dependency imports
from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.cluster.hierarchy as hierarchy
import scipy.spatial.distance as distance
import scipy.optimize as optimize
from scipy import stats
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances, adjusted_rand_score
from wNMF import wNMF

import normalise
# Local imports
from mg_nmf.nmf import normalise as norm, synthdata as synthdata


class BetaLoss(Enum):
    """Supported beta-loss functions."""
    KULLBACK_LEIBLER: str = "kullback-leibler"
    FROBENIUS: str = "frobenius"
    ITAKURA_SAITO: str = "itakura-saito"


class Initialization(Enum):
    """Supported W and H initialisation methods."""
    RANDOM: str = "random"
    NNSVD: str = "nnsvd"
    NNSVDA: str = "nnsvda"
    NNSVDAR: str = "nnsvdar"


class Solver(Enum):
    MULTIPLICATIVE_UPDATE: str = "mu"
    COORDINATE_DESCENT: str = "cd"


class NMFModel(NamedTuple):
    """Filtering can lead to a model being fit on a subset of features. This class stores both the model, and
    the features it was learned from, to allow refitting of data."""
    feature_subset: List[str]
    model: Union[NMF, wNMF]

    def w(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using this model (get W matrix).

        :param data: Data to transform. Must contain all features in feature subset.
        :type data: pd.DataFrame

        """

        restrict_data: pd.DataFrame = data.loc[:, self.feature_subset]
        w: np.ndarray = self.model.transform(restrict_data)
        w_df: pd.DataFrame = pd.DataFrame(w)
        w_df.index = data.index
        w_df.columns = NMFDecomposer.component_names(self.model.n_components)
        return w_df

    @property
    def h(self) -> pd.DataFrame:
        """Get H matrix as DataFrame."""

        return pd.DataFrame(
            self.model.components_,
            index=NMFDecomposer.component_names(self.model.n_components),
            columns=self.feature_subset
        )


class NMFOptions:
    """Hold all options offered for NMF decomposition in the package."""

    DEF_NMF_MAX_ITER = 1000

    def __init__(self,
                 solver: Solver = Solver.MULTIPLICATIVE_UPDATE,
                 beta_loss: BetaLoss = BetaLoss.KULLBACK_LEIBLER,
                 nmf_max_iter: int = None,
                 normalise: Callable[[pd.DataFrame, float], pd.DataFrame] = None,
                 normalise_arg: float = None,
                 filter_fn: Callable[[pd.DataFrame, float], pd.DataFrame] = None,
                 filter_threshold: float = None) -> None:
        """Intialise NMF options object

        :param solver: nmf solver method
        :type solver: Solver
        :param beta_loss: beta loss function
        :type beta_loss: BetaLoss
        :param nmf_max_iter: maximum number of iterations for a single nmf run
        :type nmf_max_iter: int
        :param normalise: function to normalise data during training, from normalise module
        :type normalise: function
        :param normalise_arg: arguments to pass to normalise function, see function in normalise module for details
        :type normalise_arg: any
        :param filter_fn: function to reduce data during training, from normalise module
        :type filter_fn: function
        :param filter_threshold: argument to be passed to filter function, generally proportion of features to discard
        :type filter_threshold: float
        """
        self.solver = solver
        self.beta_loss = beta_loss
        self.nmf_max_iter = nmf_max_iter
        self.normalise = normalise
        self.filter = filter_fn
        self.filter_threshold = filter_threshold
        self.normalise_arg = normalise_arg

    @property
    def nmf_max_iter(self) -> int:
        """Return maximum iterations during individual nmf run."""
        return self.__nmf_max_iter

    @nmf_max_iter.setter
    def nmf_max_iter(self, nmf_max_iter: int) -> None:
        """Set maximum number of iterations during individual nmf run."""
        if nmf_max_iter is None:
            self.__nmf_max_iter: int = self.DEF_NMF_MAX_ITER
        elif nmf_max_iter < 1:
            self.__nmf_max_iter: int = self.DEF_NMF_MAX_ITER
        else:
            self.__nmf_max_iter: int = nmf_max_iter

    @property
    def solver(self) -> Solver:
        """Return solver type."""
        return self.__solver

    @solver.setter
    def solver(self, solver: Solver) -> None:
        """Set solver type."""
        # Support providing enum or string value matching an enum value
        if not isinstance(solver, Solver):
            if str(solver) in Solver:
                solver = Solver[str(solver)]
            else:
                raise Exception(f'Value {solver} for solver now allowed, must be one of {Solver}')
        self.__solver: Solver = solver

    @property
    def beta_loss(self) -> BetaLoss:
        """Return the beta loss function."""
        return self.__beta_loss

    @beta_loss.setter
    def beta_loss(self, beta_loss: BetaLoss) -> None:
        """Set the beta loss function."""
        # Support providing enum or string value matching an enum value
        if not isinstance(beta_loss, BetaLoss):
            if beta_loss in BetaLoss:
                beta_loss = BetaLoss[str(beta_loss)]
            else:
                raise Exception(f'Value {beta_loss} for solver now allowed, must be one of {BetaLoss}')
        self.__beta_loss: BetaLoss = beta_loss


    @property
    def normalise(self) -> Callable[[pd.DataFrame, float], pd.DataFrame]:
        """Function used to map each column of W to a value to normalise W & H by."""
        return self.__normalise

    @normalise.setter
    def normalise(self, normalise: Callable[[pd.DataFrame, float], pd.DataFrame]) -> None:
        """Function used to map each column of W to a value to normalise W & H by."""
        self.__normalise = normalise

    @property
    def normalise_arg(self) -> float:
        """Numerical argument to pass to normalisation function."""
        return self.__normalise_arg

    @normalise_arg.setter
    def normalise_arg(self, normalise_arg: float) -> None:
        """Numerical argument to pass to normalisation function."""
        self.__normalise_arg: float = normalise_arg

    @property
    def filter(self) -> Callable[[pd.DataFrame, float], pd.DataFrame]:
        return self.__filter

    @filter.setter
    def filter(self, filter_fn: Callable[[pd.DataFrame, float], pd.DataFrame]) -> None:
        self.__filter = filter_fn

    @property
    def filter_threshold(self) -> float:
        return self.__filter_threshold

    @filter_threshold.setter
    def filter_threshold(self, filter_threshold: float) -> None:
        self.__filter_threshold: float = filter_threshold

    @classmethod
    def from_options(cls, options: NMFOptions) -> NMFOptions:
        """Copy constructor

        :param options: Object to copy
        :type options: NMFOptions
        :return: Copied object
        :rtype: NMFOptions
        """
        return NMFOptions(
            solver=options.solver,
            beta_loss=options.beta_loss,
            nmf_max_iter=options.nmf_max_iter,
            normalise=options.normalise,
            normalise_arg=options.normalise_arg,
            filter_fn=options.filter,
            filter_threshold=options.filter_threshold
        )

    def __repr__(self):
        vals: str = ", ".join(
            filter(lambda x: x is not None,
                [f'solver={self.solver.value}',
                 f'beta_loss={self.beta_loss.value}'
                 f'nmf_max_iter={self.__nmf_max_iter}',
                 f'normalise={self.normalise}' if self.normalise is not None else None,
                 f'normalise_args={self.normalise_arg}' if self.normalise_arg is not None else None,
                 f'filter={self.filter}' if self.filter is not None else None,
                 f'filter_arg={self.filter_threshold}' if self.filter_threshold is not None else None]
            )
        )
        return f'NMFOptions({vals})'


class NMFDecomposer:
    """Perform decompositions for a given matrix, and keep the results. Results are held in memory, and if n
    decompositions are requested, and p had been previously run, max(n-p, 0) new decompositions will be run and
    returned along with the prior decompositions. This behaviour can be disabled with the cache parameters.

    Some of the rank selection methods implemented are derived from multiple decompositions, so it is more
    computationally efficient to perform these decompositions once and calculate measures on all of them. This class
    is intended to transparently handle this, rather than having to externally make a dict of decompositions and
    feed them into the rank selection classes.
    """

    MAX_RAND_INT: int = 214748264

    @dataclass
    class RankCache:
        """Want to ensure reproducible results, regardless of which order decompositions are requested in. Each rank
        will have separate random number generator used to get seeds for the n-th decomposition of a rank. The seed
        for the rank generator is the decomposer seed + rank. Uncached results do not use these random number genrators
        and instead use the default random methods."""
        seed_generator: random.Random
        decompositions: List[NMFModel]

    def __init__(self, data: pd.DataFrame, options: NMFOptions = NMFOptions(), cache: bool = True,
                 processes: int = 1, seed: int = None) -> None:
        """Initialise decomposer. All properties other than processes are read-only, cannot be altered after
        initialisation to ensure cache results consistent with parameters."""
        self.__data: pd.DataFrame = data
        self.__options: NMFOptions = options
        self.__cache: bool = False
        self.__decompositions: Dict[int, NMFDecomposer.RankCache] = dict()
        self.__seed: int = seed if seed is not None else random.randint(0, self.MAX_RAND_INT)
        self.__seed_generators: Dict[int, random.Random] = dict()
        self.__cache = cache
        self.processes = processes
        self._data_warn()

    @property
    def data(self) -> pd.DataFrame:
        """Return the X matrix being decomposed."""
        # Copy for full encapsulatioon
        return self.__data.copy()

    @property
    def options(self) -> NMFOptions:
        return self.__options

    @property
    def cache(self) -> bool:
        return self.__cache

    @property
    def seed(self) -> int:
        return self.__seed

    @property
    def processes(self) -> int:
        return self.__processes

    @processes.setter
    def processes(self, processes: int) -> None:
        if processes < 1:
            processes = 1
        self.__processes: int = processes

    def _fetch_from_cache(self, rank: int, n: int) -> List[NMFModel]:
        """Get as many of n decompositions for rank as possible from the cache.

        :param rank: Number of modules in decomposition
        :type rank: int
        :param n: Number of decompositions
        :type n: int
        :return: List of decompositions, potentially empty
        :rtype: List[NMFModel]
        """

        if rank not in self.__decompositions:
            self.__decompositions[rank] = NMFDecomposer.RankCache(random.Random(self.seed + rank), list())
        return list(itertools.islice(self.__decompositions[rank].decompositions, n))

    def _data_warn(self) -> None:
        """Provide warning if it appears that data may be in incorrect orientation.
        NMF literature describes X as a matrix with samples on columns, and features on rows. However, sk-learn
        has this transposed, which is important for fitting new samples. Want to detect if input data is provided in
        what is likely the wrong orientation. This is detected based on number of samples being larger than number of
        features; we would generally expect in meta-omic data that number of features >> number samples."""
        if self.data is not None:
            rows, cols = self.data.shape
            if rows > cols:
                warnings.warn('Data orientation may be incorrect. Input matrix X should have features on columns, ' 
                              'samples on rows. sk-learn orientation is transposed from majority of NMF literature.')

    @staticmethod
    def component_names(rank: int) -> List[str]:
        """Names for components in result dataframes

        :param rank: Number of components
        :type rank: int
        :return: List of component names
        :rtpye: List[str]
        """

        return [f'm{x}' for x in range(1, rank + 1)]

    def decompose(self, rank: int, n: int, cached: bool = True) -> List[NMFModel]:
        """Get n decompositions for rank k, by default returning the cached results.

        :param rank: Number of modules in decomposition
        :type rank: int
        :param n: Number of decompositions
        :type n: int
        :param cached: Use the result cache for this request
        :type cached: bool
        :return: A list of n NMF decompositions
        :rtype: List[NMF]
        """

        from_cache: List[NMFModel] = self._fetch_from_cache(rank, n) if self.cache and cached else []
        needed: int = n - len(from_cache)

        if needed == 0:
            return from_cache

        # Construct a list of arguments for the call to _run_for_k
        random_state: random.Random = (self.__decompositions[rank].seed_generator if not cached or not self.__cache
                                       else random)
        args: Iterable[Tuple[int, int]] = (
            (rank, self.__decompositions[rank].seed_generator.randint(0, self.MAX_RAND_INT)) for _ in range(needed)
        )

        # Run the required number of times
        new: List[NMFModel]
        if self.processes > 1:
            with multiprocessing.Pool(processes=self.processes) as pool:
                new = pool.starmap(self._run_for_k, args)
        else:
            new = [self._run_for_k(*x) for x in args]

        # Add new to cache
        if cached:
            self.__decompositions[rank].decompositions += new

        return from_cache + new

    def filter(self, model: NMFModel) -> pd.DataFrame:
        """Reduce the H matrix of a model to only the more informative features, using given filter and
        normalisation functions defined in NMFOptions. This method only filters, does not relearn the model based only
        on these features.

        :param model: NMF model to be filtered
        :type model: NMF
        :return: Reduced W matrix
        :rtype: pd.DataFrame
        """

        h: pd.DataFrame = model.h
        components: List[str] = list(h.index)
        if self.options.filter is not None:
            h_norm: pd.DataFrame = h
            if self.options.normalise is not None:
                h_norm: pd.DataFrame = norm.normalise(
                    self.options.normalise(h, self.options.normalise_arg), None, h)[1]
            # Filter
            u: pd.DataFrame = self.options.filter(h_norm, self.options.filter_threshold)
            # Reduce H to the restricted list
            h = h.loc[:,list(u.columns)]
        h.index.name = "modules"
        return h

    def normalise(self, model: NMFModel, w: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Normalise W and H matrices of decomposition using the given normalisation methods provided in NMFOptions.

        :param model: NMF model to be normalised
        :type model: NMF
        :param w: Reduced W matrix to use in normalising; if not provided, uses full matrix
        :type w: pd.DataFrame
        :return: Normalised W and H matrices
        :rtype: Tuple[pd.DataFrame, pd.DataFrame]
        """

        h = model.h
        components: List[str] = list(h.index)
        if w is None:
            w = model.w(self.data)
        if self.options.normalise is not None:
            w, h = norm.normalise(
                self.options.normalise(h, self.options.normalise_arg), w, h)
        w_res: pd.DataFrame = w
        w.columns = components
        w_res.index.name = 'samples'
        h_res: pd.DataFrame = pd.DataFrame(data=h, index=components,
                                           columns=self.data.columns)
        h_res.index.name = 'modules'

        return w_res, h_res

    def _run_for_k(self, k: int, seed: int) -> NMFModel:
        """Create models and assess stability for k components.

        :param k: number of modules to identify
        :type k: int
        :param seed: random seed
        :type seed: int
        :return: reduced dimension model
        :rtype: NMFModel
        """
        start_time = time.time()
        start_time_text = time.strftime("%X")
        # Create a model with our parameters for this value of k
        # try:
        feature_subset: List[str] = list(self.data.columns)
        model = NMF(n_components=k,
                    solver=self.options.solver.value,
                    beta_loss=self.options.beta_loss.value,
                    init=Initialization.RANDOM.value,
                    random_state=seed,
                    max_iter=self.options.nmf_max_iter)
        w = model.fit_transform(self.data)
        if self.options.filter is not None:
            # Normalise, filter source data, rerun on reduced data
            u: pd.DataFrame = self.filter(NMFModel(list(self.data.columns), model))
            feature_subset: List[str] = list(u.columns)
            # Restrict source data to these indices
            reduced: pd.DataFrame = self.data.loc[:, feature_subset]
            # Rerun NMF and return model
            reduced_model = NMF(n_components=k,
                                solver=self.options.solver.value,
                                beta_loss=self.options.beta_loss.value,
                                init='random',
                                random_state=seed,
                                max_iter=self.options.nmf_max_iter)
            # Fit model to reduced data - don't need the returned W matrix
            _ = reduced_model.fit_transform(reduced)
            model = reduced_model
        duration = time.time() - start_time
        print(f'[{time.strftime("%X")} - {start_time_text}] [k={k}] Ran, took {duration} seconds')
        return NMFModel(feature_subset, model)
        # except Exception as err:
        #     print(err)

    def for_data(self, data: pd.DataFrame) -> NMFDecomposer:
        """Create an object with the same settings but for different data, with empty cache.

        :param data: Data to be decomposed
        :type data: pd.DataFrame
        :return: Decomposer for the new data, with empty cache
        :rtype: NMFDecomposer
        """
        return NMFDecomposer(data=data, options=self.options, cache=self.cache, processes=self.processes,
                             seed=self.seed)

    def __repr__(self):
        return (f'NMFDecomposer(id={id(self)} | data={id(self.data)}, cache={self.cache}, processes={self.processes}, '
                f'seed={self.seed})')


class NMFModelSelection(ABC):
    """Abstract class to handle model selection. Subclasses should implement abstract methods for their criteria.

    Public methods:
    run                 -- run the model selection with specified parameters
    list_solvers        -- provide a list of the supported solver methods
    list_beta_loss      -- provide a list of the supported beta-loss functions

    Instance variables:
    decomposter:        --  decomposer object to perform decompositions
        NMFDecomposer
    k_min: int          -- value to start searching for k
    k_max: int          -- value to end search for k
    k_interval: int     -- interval between values of k to search (i.e. interval 2 = 2, 4, 6 ...)
    k_values: List[int] -- specific values of k to search. k_min, k_max, k_interval ignored if this is provided
    iterations: int     -- number of iterations to perform for each value of k
    metric: str         -- model selection metric(s) to generate, varies depending on concrete implementation
    metrics: list[str]  -- list of the metrics available in this concrete implementation
    """

    ABS_K_MIN: int = 2
    DEF_K_MIN: int = 2
    DEF_K_MAX: int = 15
    DEF_K_INTERVAL: int = 1
    DEF_ITERATIONS: int = 50
    DEF_NMF_MAX_ITER = 1000
    SOLVERS: List[str] = ['cd', 'mu']
    BETA_LOSS: List[str] = ['frobenius', 'kullback-leibler', 'itakura-saito']
    INIT: List[str] = ['random', 'nndsvd', 'nndsvda', 'nndsvdar']

    def __init__(self,
                 decomposer: NMFDecomposer,
                 k_min: int = None,
                 k_max: int = None,
                 k_interval: int = None,
                 k_values: List[int] = None,
                 iterations: int = None,
                 metric: str = None) -> None:
        """Intialise a model selection object ready to be run.

        :param decomposer: Object to perform decomposition (contains the data)
        :type decomposer: NMFDecomposer
        :param k_min: value to start searching k from
        :type k_min: int
        :param k_max:value to search k to
        :type k_max: int
        :param k_interval: interval between values of k, equivalent to range(1, 10, k)
        :type k_interval: int
        :param k_values: values of k to search. If None, will create a list from k_min, k_max, k_interval
        :type k_values: List[str]
        :param metric: model selection metric(s) to produce
        :type metric: str, list[str]
        """
        self.k_min = k_min
        self.k_max = k_max
        self.k_interval = k_interval
        self.k_values = k_values
        self.metric = self.metrics[-1] if metric is None else metric
        self.weighting = None
        self.__decomposer: NMFDecomposer = decomposer
        self.iterations = iterations

    @property
    def k_min(self) -> int:
        """Return number of components for model."""
        return self.__k_min

    @k_min.setter
    def k_min(self, k_min: int) -> None:
        """Set minimum number of components for model."""
        if k_min is None:
            k_min = self.DEF_K_MIN
        self.__k_min: int = max(k_min, self.ABS_K_MIN)

    @property
    def k_max(self) -> int:
        """Return maximum number of components."""
        return self.__k_max

    @k_max.setter
    def k_max(self, k_max: int) -> None:
        """Set maximum numbers of components."""
        if k_max is None:
            k_max = self.DEF_K_MAX
        # Set to passed value, or the minimum degree if the minimum is larger
        # than the requested max value
        self.__k_max: int = max(self.k_min, k_max)

    @property
    def k_interval(self) -> int:
        """Return interval at which to search k space."""
        return self.__k_interval

    @k_interval.setter
    def k_interval(self, k_interval: int) -> None:
        """Set interval at which to search k space."""
        if k_interval is None:
            k_interval = self.DEF_K_INTERVAL
        self.__k_interval: int = k_interval

    @property
    def k_values(self) -> List[int]:
        """Return list of values of k to be searched."""
        return self.__k_values

    @k_values.setter
    def k_values(self, k_values: List[int]) -> None:
        """Set the values of k to be searched."""
        if k_values is None:
            k_values = self.__compose_k_values()
        self.__k_values: List[int] = k_values

    def __compose_k_values(self) -> List[int]:
        """Make a list of values of k to be searched from min, max and interval."""
        return list(range(self.k_min, self.k_max+1, self.k_interval))

    @property
    def metric(self):
        return self.metrics[-1] if self.__metric is None else self.__metric

    @metric.setter
    def metric(self, metric: str):
        if metric not in self.metrics:
            raise Exception(f'Value {metric} for metric not allowed, must be one of {self.metrics}')
        self.__metric: str = metric

    @property
    def iterations(self) -> int:
        return self.__iterations

    @iterations.setter
    def iterations(self, iterations: int) -> None:
        if iterations is None:
            iterations = 1
        self.__iterations: int = max(iterations, 1)

    @property
    def decomposer(self) -> NMFDecomposer:
        return self.__decomposer

    @property
    @abstractmethod
    def metrics(self) -> List[str]:
        """Return a list of the metrics allowed for this method."""
        pass

    @abstractmethod
    def run(self, processes: int = 1) -> NMFResults:
        """Run process for model selection of NMF."""
        pass


# noinspection SpellCheckingInspection
class NMFConsensusSelection(NMFModelSelection):
    """Perform model selection based on consensus metrics, either Brunet (2004) cophenetic correlation
    or Kim & Park (2007) dispersion criteria."""
    METRICS: List[str] = ['cophenetic', 'dispersion', 'both']

    @property
    def metrics(self) -> List[str]:
        return self.METRICS

    def run(self, processes: int = 1) -> Union[Tuple[NMFResults, NMFResults], NMFResults]:
        """Run process for model selection of NMF.

        Returns results for all k. For each k returns the average connectivity matrix and the cophenetic
        correlation coefficient for that k, the w & h components, and the model with lowest beta divergence.

        :return: results of tuples of model selection metrics, first is cophenetic correlation, second dispersion
        :rtype: (NMFResults, NMFResults), or NMFResults
        """
        results: List[Tuple[NMFModelSelectionResults, NMFModelSelectionResults]] = list()
        for i in self.k_values:
            print(f'[{time.strftime("%X")}] AGGREGATE {i}')
            kmodels: List[NMFModel] = self.decomposer.decompose(i, self.iterations)
            kresults: Tuple[NMFModelSelectionResults, NMFModelSelectionResults] = self._results_for_k(i, kmodels)
            results.append(kresults)
        print(f'[{time.strftime("%X")}] AGGREGATION COMPLETE')
        # Process all the different runs for a value of k
        coph_res: List[NMFModelSelectionResults] = [x[0] for x in results]
        disp_res: List[NMFModelSelectionResults] = [x[1] for x in results]
        if self.metric == 'cophenetic':
            return NMFResults(coph_res, self.decomposer.data)
        if self.metric == 'dispersion':
            return NMFResults(disp_res, self.decomposer.data)
        return (
            NMFResults(coph_res, self.decomposer.data),
            NMFResults(disp_res, self.decomposer.data)
        )

    @staticmethod
    def _create_connectivity_matrix(model_selection: Tuple[NMFModel, NMFConsensusSelection]) -> ConnectivityMatrix:
        """Return a connectivity matrix for a model. Intended to be used in multiprocessing.

        :param model_selection: pair of a learnt model, and the object which learnt it
        :type model_selection: (NMF, NMFConsensusSelection)
        :return: connecticity matrix for model
        :rtype: ConnectivityMatrix
        """

        model, selector = model_selection
        w, h = selector.decomposer.normalise(model)
        return ConnectivityMatrix(w.values)

    def _results_for_k(self,
                       k: int,
                       models: List[NMFModel]
                       ) -> Tuple[NMFModelSelectionResults, NMFModelSelectionResults]:
        """Create models and assess cophenetic correlation and dispersion for k components.

        :param k: number of components in the learnt models
        :type k: int
        :param models: list of models learnt for this value of k
        :type models: list[NMF]
        :return: model selection results using cophenetic correlation, and dispersion
        :rtype: (NMFSelectionResults, NMFSelectionResults)
        """
        i: int
        c_matrices: List[ConnectivityMatrix]
        # Find best model, normalise if required
        best_model: NMFModel = min(models, key=lambda x: x.model.reconstruction_err_)
        model_args: List[Tuple[NMFModel, NMFConsensusSelection]] = [(x, self) for x in models]
        with multiprocessing.Pool(self.decomposer.processes) as pool:
            c_matrices = list(map(self._create_connectivity_matrix, model_args))
            pool.close()
        c_bar: pd.DataFrame = ConnectivityMatrix.c_bar(c_matrices)
        # Find cophenetic correlation value
        c, co_distt = ConnectivityMatrix.cophenetic_corr(c_bar)
        # Find dispersion
        dispersion: float = ConnectivityMatrix.dispersion(c_bar)
        # Make dataframes out of w and h with lowest error
        w: pd.DataFrame = best_model.w(self.decomposer.data)
        w, h = self.decomposer.normalise(best_model, w)
        return (
            NMFModelSelectionResults(k, c, c_bar, w, h, best_model, self.decomposer.data),
            NMFModelSelectionResults(k, dispersion, c_bar, w, h, best_model, self.decomposer.data)
        )

    def __repr__(self):
        return (f'NMFConsensusSelection(k_values={self.k_values}, iterations={self.iterations}, metric={self.metric}, '
                f'decomposer={id(self.decomposer)})')

# noinspection SpellCheckingInspection
class NMFConcordanceSelection(NMFModelSelection):
    """Similarity selection from  Jiang, X., Weitz, J. S. & Dushoff, J. A non-negative matrix factorization framework
    for identifying modular patterns in metagenomic profile data. J. Math. Biol. 64, 697–711 (2012).
    """

    @property
    def metrics(self) -> List[str]:
        """The metrics available from using this class."""
        return ['condordance']

    def run(self, processes: int = 1) -> NMFResults:
        """Run process for model selection of NMF.

        Returns results for all k. For each k returns the average connectivity matrix and the cophenetic
        correlation coefficient for that k, the w & h components, and the model with lowest beta divergence.

        :return: model selection results
        :rtype: NMFResults
        """
        results: List[NMFModelSelectionResults] = list()
        for i in self.k_values:
            print(f'[{time.strftime("%X")}] AGGREGATE {i}')
            kmodels: List[NMFModel] = self.decomposer.decompose(rank=i, n=self.iterations)
            kresults: NMFModelSelectionResults = self._results_for_k(i, kmodels)
            results.append(kresults)
        print(f'[{time.strftime("%X")}] AGGREGATION COMPLETE')
        # Process all the different runs for a value of k
        return NMFResults(results, self.decomposer.data)

    @staticmethod
    def _extract_tril(a: np.ndarray) -> np.ndarray:
        """Return the lower triangle entries of a matrix in 1D array."""

        a_list: List[float] = np.tril(a).tolist()
        extract: List[float] = []
        for i in range(len(a_list)):
            extract += a_list[i][:i]
        return extract

    def _results_for_k(self, k: int, models: List[NMFModel]) -> NMFModelSelectionResults:
        """Create models and assess stability for k components.

        :param k: number of components in models
        :type k:  int
        :param models: list of models learnt
        :type models: [NMF]
        :return: model selection results for this value of k
        :rtype: NMFModelSelectionResults
        """

        # The paper states that similarity matrix S is calculates from 'sample projection' matrix H
        # Due to transposition of data in sklearn implementation, our sample matrix is actually W, and sould be
        # calculated from this instead.

        # Select model with the lowest reconstruction error
        best_model: NMFModel = min(models, key=lambda p: p.model.reconstruction_err_)
        best_W = best_model.w(self.decomposer.data)
        best_W, best_H = self.decomposer.normalise(best_model, best_W)
        # best_H_enorm: np.ndarray = np.linalg.norm(best_H, axis=0)
        # best_H_bar: pd.DataFrame = (best_H.T / best_H_enorm[:, None]).T
        # best_S: pd.DataFrame = best_H_bar.T.dot(best_H_bar)
        # Reshape to modules on rows, samples on columns
        bW_reshape: np.ndarray = best_W.T.values
        best_W_enorm: np.ndarray = np.linalg.norm(bW_reshape, axis=0)
        best_W_bar: pd.DataFrame = (bW_reshape / best_W_enorm)
        best_S: pd.DataFrame = best_W_bar.T.dot(best_W_bar)
        diffs: List[float] = []
        for model in models:
            # Skip if this model is the one which had the lowest reconstruction error
            if model is best_model:
                continue
            w_model = model.w(self.decomposer.data)
            w_model, h_model = self.decomposer.normalise(model, w_model)
            # Derive similarity matrix for this iteration
            w_reshape = w_model.T
            w_enorm = np.linalg.norm(w_reshape, axis=0)
            w_bar = (w_reshape / w_enorm)
            # Convert to similarity
            s: pd.DataFrame = w_bar.T.dot(w_bar)
            # Get D - difference between selected model and this
            d: np.ndarray = best_S - s
            d_ltri = self._extract_tril(np.square(d))
            diffs += d_ltri
        # Find mean of squared difference between best_S and S_j
        # ci = 1 - np.median(diffs)
        ci = 1 - np.mean(diffs)
        return NMFModelSelectionResults(k, ci, None, best_W, best_H, best_model, self.decomposer.data)


# noinspection SpellCheckingInspection
class NMFSplitHalfSelection(NMFModelSelection):
    """Split half validation with Adjusted Rand Index for assessing similarity.
    Based on description in Muzzarelli L, Weis S, Eickhoff SB, Patil KR. Rank Selection in Non-negative Matrix
    Factorization: systematic comparison and a new MAD metric. 2019 International Joint Conference on Neural Networks
    (IJCNN). 2019. pp 1–8.
    """

    @property
    def metrics(self) -> List[str]:
        return ['ari']

    def run(self, processes: int = 1) -> NMFResults:
        """Run process for model selection of NMF.

        Returns results for all k. For each k returns the Adjusted Rand Index for that k, the w & h components, and the
        model with lowest beta divergence.

        :return: results of model selection with adjusted rand index as criteria
        :rtype: NMFResults
        """

        i: int
        multiproc_args: List[int] = []
        for i in self.k_values:
            multiproc_args += [i] * self.iterations
        with multiprocessing.Pool(processes) as pool:
            models: List[Tuple[Tuple[NMF, pd.DataFrame], Tuple[NMF, pd.DataFrame]]] = \
                pool.map(self._run_split_for_k_single, multiproc_args)
            pool.close()
        results: List[NMFModelSelectionResults] = []
        print(f'[{time.strftime("%X")}] ITERATIONS COMPLETE, AGGREGATING RESULTS')
        for i in self.k_values:
            print(f'[{time.strftime("%X")}] AGGREGATE {i}')
            kmodels: List[Tuple[Tuple[NMF, pd.DataFrame], Tuple[NMF, pd.DataFrame]]] \
                = [x for x in models if x[0][0].n_components == i]
            kresults: NMFModelSelectionResults = self._results_for_k(i, kmodels)
            results.append(kresults)
        print(f'[{time.strftime("%X")}] AGGREGATION COMPLETE')
        # Process all the different runs for a value of k
        return NMFResults(results, self.data)

    def _run_split_for_k_single(self, k: int) \
            -> Tuple[Tuple[NMF, Optional[pd.DataFrame]], Tuple[NMF, Optional[pd.DataFrame]]]:
        """Perform a single run of the selection method."""

        # First step is to randomly split the data in half
        pair: Tuple[pd.DataFrame, pd.DataFrame] = train_test_split(self.data, train_size=0.5, test_size=0.5)
        # Run NMF training as normal on both halves
        pair_models = tuple(self._run_for_k_single(k, x) for x in pair)
        del pair
        return (pair_models[0], None), (pair_models[1], None)

    def _results_for_k(self, k: int, dm_pairs: List[Tuple[Tuple[NMF, pd.DataFrame], Tuple[NMF, pd.DataFrame]]]) \
            -> NMFModelSelectionResults:
        """Generate aRI for data which has been split and models fitted for both halves."""
        aris: List[float] = []
        for dm_pair in dm_pairs:
            # Have to match the components generated. Using Hungarian algorithm.
            # Call our two H matrices H_1  and H_2. First factor in H_1 is H_1_f1.
            # Need a measure of the cost of 'assigning'/matching H_1_f1 to H_2_f1. Not clear from descriptions how
            # other papers have done this, using Euclidean distance between observations here
            M_1, M_2 = dm_pair[0][0], dm_pair[1][0]
            H_1, H_2 = M_1.components_, M_2.components_
            costs: np.ndarray = pairwise_distances(H_1, H_2)
            row_ind, col_ind = optimize.linear_sum_assignment(costs)
            # Reorder H_2
            H_2 = H_2[col_ind]
            # To determine aRI index, need to assign each feature to a component
            H_1_class: np.ndarray = np.argmax(H_1, axis=0)
            H_2_class: np.ndarray = np.argmax(H_2, axis=0)
            ari: float = adjusted_rand_score(H_1_class, H_2_class)
            aris.append(ari)
        ari_median: float = np.median(aris)
        ari_mean: float = np.mean(aris)

        # Can't return a model, as not trained on full data. Return all other parts of results.
        return NMFModelSelectionResults(k, ari_median, None, None, None, None, self.data)


# noinspection SpellCheckingInspection
class NMFPermutationSelection(NMFModelSelection):
    """
    Perform model selection by computing NMF on input data & data with each row's values shuffled. k is selected
    as the first value of k for which the slope of reconstruction error in the original matrix is lower than that of
    the shuffled matrix.

    We run this process n times, and select the modal value.
    """

    @property
    def metrics(self) -> List[str]:
        return ['kprop']

    def run(self, processes: int = 1) -> NMFResults:
        """Run model selection, using method comparing reconstruction error in shuffled & original data

        :return: results of model selection
        :rtype: NMFResults
        """
        i: int
        with multiprocessing.Pool(processes) as pool:
            results: List[Tuple[NMF, int]] = pool.map(self._run_once, range(self.iterations))
            pool.close()
        resultobjs: List[NMFModelSelectionResults] = []
        print(f'[{time.strftime("%X")}] ITERATIONS COMPLETE, AGGREGATING RESULTS')
        nonnull_results: List[Tuple[NMF, int]] = [x for x in results if x[0] is not None]
        for i in self.k_values:
            print(f'[{time.strftime("%X")}] AGGREGATE {i}')
            # Select the models for which i was selected as optimal k
            kmodels: List[Tuple[NMF, int]] = [x for x in nonnull_results if x[0].n_components == i]
            # Make this into an NMF model selection results  object
            proportion: float = len(kmodels) / len(results) if len(results) > 0 else 0
            # Select the model with best reconstruction error
            best_model: NMF = min(kmodels, key=lambda x: x[0].reconstruction_err_)[0] if len(kmodels) > 0 else None
            resobj: NMFModelSelectionResults = (
                NMFModelSelectionResults(i, proportion, None, None, None, best_model, self.data)
            )
            resultobjs.append(resobj)
        print(f'[{time.strftime("%X")}] AGGREGATION COMPLETE')
        # Output proportion of runs where no stopping criteria was met
        null_models: List[Tuple[NMF, int]] = [x for x in results if x[0] is None]
        null_prop: float = len(null_models) / len(results) if len(results) > 0 else 0
        print(f'[{time.strftime("%X")}] PERMUTATION STOPPING REPORT')
        print(f'[{time.strftime("%X")}] Condition not met: {null_prop:.2%}')
        print(f'[{time.strftime("%X")}] If stopping condition not met in high proportion of runs, ' +
              f'consider raising value of k search')
        # Process all the different runs for a value of k
        return NMFResults(resultobjs, self.data)

    def _run_once(self, i: int) -> Tuple[NMF, int]:
        """Run permutation once. Of n runs, this is the i-th. Not used, but parameter included as pool map method
        passes a parameter

        :param i: i-th run
        :type i: int
        :return: the model, and selected value for k
        :rtype: (NMF, int)
        """
        # First step is to permute each row of the data
        shuffled: pd.DataFrame = self.data.copy().T
        for col in shuffled.columns:
            shuffled[col] = np.random.permutation(shuffled[col])
        shuffled = shuffled.T
        recon_errs: List[Tuple[float, float]] = []
        for k in self.k_values:
            # Run NMF as usual on both shuffled and permuted data
            reg_model: NMF = self._run_for_k_single(k, self.data)
            shuf_model: NMF = self._run_for_k_single(k, shuffled)
            # Calculate slope from k-1 to k for each
            # Below is code to manually calculate euclidean norm, no longer using, taking whatever distance specified
            # curr_r: float = np.linalg.norm(self.data -
            # reg_model.transform(self.data).dot(reg_model.components_),  ord='fro')
            # curr_s : float = np.linalg.norm(shuffled -
            # shuf_model.transform(shuffled).dot(shuf_model.components_),  ord='fro')
            curr_r, curr_s = reg_model.reconstruction_err_, shuf_model.reconstruction_err_
            recon_errs.append((curr_r, curr_s))
            if len(recon_errs) > 1:
                prev_r, prev_s = recon_errs[-2]
                delta_r, delta_s = curr_r - prev_r, curr_s - prev_s
                if delta_r > delta_s:
                    # Return the model for value of k preceding this one, as increasing k here suggested no additional
                    # information to extract
                    return prev_rmodel, k - 1
                if k == self.k_values[-1]:
                    # No stopping condition reached within the specified values of k searched, so wish to return a
                    # null result
                    return None, None
            # Save the current model - will want to return them if next k shows levelling of slope
            prev_rmodel: NMF = reg_model


class NMFImputationSelection(NMFModelSelection):
    """Select model by holding out a random selection of values, assessing how well WH imputes these values. This uses
    the wNMF package, which is considerably slower than the sklearn NMF implementation."""
    METRICS: List[str] = ['mse', 'mad', 'both']

    def __init__(self, data: pd.DataFrame, k_min: int = None, k_max: int = None, k_interval: int = None,
                 k_values: List[int] = None, solver: str = None, beta_loss: str = None, iterations: int = None,
                 nmf_max_iter=None, normalise: Callable[[pd.DataFrame, float], pd.DataFrame] = None,
                 normalise_arg: float = None, filter_fn=None, filter_threshold: float = None, metric: str = 'mse',
                 p_holdout: float = 0.1) -> None:
        """Set up properties for model selection.

        :param data: data to learn model from
        :type data: DataFrame
        :param k_min: value of k to begin search at
        :type k_min: int
        :param k_max:  value of k to end search at
        :type k_max: int
        :param solver: nmf solver method
        :type solver: str
        :param beta_loss: beta loss function
        :type beta_loss: function
        :param iterations: number of iterations for each value of k
        :type iterations: int
        :param nmf_max_iter: number of iterations during each nmf learning
        :type nmf_max_iter: int
        :param normalise: method to normalise during learning, available in normalise module
        :type normalise: function
        :param normalise_arg: arguments to pass to normalise function
        :type normalise_arg: any
        :param filter_fn: function to reduce number of features in model
        :type filter_fn: function
        :param filter_threshold: argument for filter method, generally proportion of features to discard
        :type filter_threshold: float
        :param metric: the metric or metrics to use
        :type metric: str
        :param p_holdout: proportion of data to holdout, 0 < p_holdout < 1
        :type p_holdout: float
        """
        super().__init__(data=data, k_min=k_min, k_max=k_max, solver=solver, beta_loss=beta_loss, iterations=iterations,
                         nmf_max_iter=nmf_max_iter, normalise=normalise, normalise_arg=normalise_arg,
                         filter_fn=filter_fn, k_values=k_values,
                         filter_threshold=filter_threshold, metric=metric)
        self.p_holdout = p_holdout
        if filter_fn is not None or normalise is not None:
            raise Exception('Imputation selection does not support filtering and normalising during selection.')

    @property
    def p_holdout(self) -> float:
        """Return the proportion of the input data to holdout."""
        return self.__p_holdout

    @p_holdout.setter
    def p_holdout(self, p_holdout: float) -> None:
        """Set the proportion of the input data to holdout. 0 < p_holdout < 1."""
        self.__p_holdout: float = p_holdout

    @property
    def metrics(self) -> List[str]:
        return self.METRICS

    def run(self, processes: int = 1) -> Union[NMFResults, Tuple[NMFResults, NMFResults]]:
        """Run imputation based model selection. Return either the request metric, or a tuple of mse and mad."""
        i: int
        multiproc_args: List[int] = []
        for i in self.k_values:
            multiproc_args += [i] * self.iterations
        with multiprocessing.Pool(processes) as pool:
            models: List[Tuple[wNMF, float]] = pool.map(self._run_for_k_single, multiproc_args)
            pool.close()
        results: List[Tuple[NMFModelSelectionResults, NMFModelSelectionResults]] = []
        print(f'[{time.strftime("%X")}] ITERATIONS COMPLETE, AGGREGATING RESULTS')
        for i in self.k_values:
            print(f'[{time.strftime("%X")}] AGGREGATE {i}')
            k_models: List[Tuple[wNMF, float]] = [x for x in models if x[0].n_components == i]
            k_results: Tuple[NMFModelSelectionResults, NMFModelSelectionResults] = self._results_for_k(i, k_models)
            results.append(k_results)
        print(f'[{time.strftime("%X")}] AGGREGATION COMPLETE')
        # Process all the different runs for a value of k
        mse_results: List[NMFModelSelectionResults] = [x[0] for x in results]
        mad_results: List[NMFModelSelectionResults] = [x[1] for x in results]
        if self.metric == 'mse':
            return NMFResults(mse_results, self.data, lambda x, y: x if x.metric < y.metric else y)
        if self.metric == 'mad':
            return NMFResults(mad_results, self.data, lambda x, y: x if x.metric < y.metric else y)
        if self.metric == 'both':
            return (
                NMFResults(mse_results, self.data, lambda x, y: x if x.metric < y.metric else y),
                NMFResults(mad_results, self.data, lambda x, y: x if x.metric < y.metric else y)
            )

    def _run_for_k_single(self, k: int, data: pd.DataFrame = None) -> Tuple[wNMF, float]:
        """Run a single learning using the imputation method."""
        # Timing initialisation
        start_time = time.time()
        start_time_text = time.strftime("%X")
        print(f'[{start_time_text}] [k={k}] Starting wNMF run...')
        # Select random indices to hold out
        holdout_num: int = int(self.data.size * self.p_holdout)
        # Matrix for weighting of each entry. Set held out entry to weight 0.
        weights: np.ndarray = np.ones(self.data.shape)
        holdout_idxs: List[Tuple[int, int]] = []
        filled: int = 0
        # Select random indices to set to 0
        while filled < holdout_num:
            i, j = random.randrange(0, self.data.shape[0] - 1), random.randrange(0, self.data.shape[1] - 1)
            if weights[i, j] > 0:
                filled += 1
                weights[i, j] = 0
                holdout_idxs.append((i, j))
        model: wNMF = wNMF(k, init='random', beta_loss=self.beta_loss, max_iter=self.nmf_max_iter)
        model.fit(self.data.values, weights)
        # Get Mean Squared Error for WH
        WH: np.ndarray = model.U.dot(model.V)
        errs: List[float] = [(self.data.iloc[x] - WH[x[0], x[1]]) ** 2 for x in holdout_idxs]
        mse: float = np.mean(errs)
        duration = time.time() - start_time
        print(f'[{time.strftime("%X")} - {start_time_text}] [k={k}] wNMF ran, took {duration} seconds')
        return model, mse

    def _results_for_k(self, k: int, results: List[Tuple[wNMF, float]]) \
            -> Tuple[NMFModelSelectionResults, NMFModelSelectionResults]:
        models: List[wNMF] = [x[0] for x in results]
        metrics: List[float] = [x[1] for x in results]
        best_model: wNMF = min(models, key=lambda p: p.reconstruction_err_)
        components: List[str] = [f'c{str(i + 1)}' for i in range(k)]
        W: pd.DataFrame = pd.DataFrame(best_model.U, columns=components,
                                       index=self.data.index)
        H: pd.DataFrame = pd.DataFrame(best_model.V, columns=self.data.columns, index=components)
        # Use median MSE as metric
        median_mse: float = np.median(metrics)
        # Median absolute deviation function name changed in scipy - want to handle both
        if hasattr(stats, 'median_abs_deviation'):
            mad: float = stats.median_abs_deviation(metrics)
        else:
            mad: float = stats.median_absolute_deviation(metrics)
        mse_result, mad_result = (
            NMFModelSelectionResults(k, median_mse, None, W, H, best_model, self.data),
            NMFModelSelectionResults(k, mad, None, W, H, best_model, self.data)
        )
        return mse_result, mad_result


class NMFMultiSelect:
    """Run multiple model selection criteria on a single set of data.

    Public methods:
    run(DataFrame)          -- run the selected model selection methods on the data provided

    Instance variables:
    ranks: Tuple[int,int]   -- rank to search from and to
    methods: List[str]      -- model selection methods to use
    iterations: int         -- number of iterations for each value of k, for each method
    nmf_max_iter: int       -- max number of iterations for each nmf run
    solver: str             -- solver function to use
    beta_loss: str          -- beta loss function to use for reconstruction error
    """

    # noinspection SpellCheckingInspection
    PERMITTED_METHODS: List[str] = [
        'coph', 'disp', 'conc', 'perm', 'split', 'mse', 'mad'
    ]

    def __init__(self, ranks: List[int], methods: List[str] = None, iterations: int = None,
                 nmf_max_iter: int = None, solver: str=None, beta_loss: str=None) -> None:
        """Initialise model selection using one or more of the available methods.

        :param ranks: values of k to search
        :type ranks: List[int]
        :param methods: list of model selection methods to apply
        :type methods: str
        :param iterations: number of iterations to run for each value of k, for each method
        :type iterations: int
        :param nmf_max_iter: number of iterations per nmf run, ends earlier if converges
        :type nmf_max_iter: int
        :param solver: nmf solver method to use
        :type solver: int
        :param beta_loss: beta loss function to use
        :type beta_loss: str
        """
        self.ranks = ranks
        self.methods = methods
        self.iterations = iterations
        self.nmf_max_iter = nmf_max_iter
        self.solver = solver
        self.beta_loss = beta_loss

    @property
    def ranks(self) -> List[int]:
        return self.__ranks

    @ranks.setter
    def ranks(self, ranks: List[int]) -> None:
        self.__ranks: List[int] = ranks

    @property
    def iterations(self) -> int:
        return self.__iterations

    @iterations.setter
    def iterations(self, iterations: int) -> None:
        if iterations < 1:
            raise Exception('Iterations must be a positive integer.')
        self.__iterations: int = iterations

    @property
    def methods(self) -> List[str]:
        return self.__methods

    @methods.setter
    def methods(self, methods: List[str]) -> None:
        self.__methods: List[str] = []
        if methods is None:
            self.__methods = self.PERMITTED_METHODS
        else:
            invalid: List[str] = [x for x in methods if x not in self.PERMITTED_METHODS]
            if len(invalid) > 0:
                raise Exception(f'Invalid methods specified: {invalid}')
            self.__methods = methods

    @property
    def nmf_max_iter(self) -> int:
        return self.__nmf_max_iter

    @nmf_max_iter.setter
    def nmf_max_iter(self, nmf_max_iters: int) -> None:
        self.__nmf_max_iter: int = nmf_max_iters

    @property
    def solver(self) -> str:
        return self.__solver

    @solver.setter
    def solver(self, solver: str) -> None:
        if solver not in NMFModelSelection.SOLVERS and solver is not None:
            raise Exception(f'Invalid solver "{solver}", solvers allowed: {NMFModelSelection.SOLVERS}')
        self.__solver: str = solver

    @property
    def beta_loss(self) -> str:
        return self.__beta_loss

    @beta_loss.setter
    def beta_loss(self, beta_loss: str) -> None:
        if beta_loss not in NMFModelSelection.BETA_LOSS:
            raise Exception(f'Invalid beta loss "{beta_loss}", values allowed: {NMFModelSelection.BETA_LOSS}')
        self.__beta_loss: str = beta_loss

    # noinspection SpellCheckingInspection
    def run(self, data: pd.DataFrame, processes: int = 1) -> Dict[str, NMFResults]:
        """Run each of the selected methods for the provided data."""
        res: Dict[str, NMFResults] = {}

        if 'coph' in self.methods or 'disp' in self.methods:
            sel_coph: NMFModelSelection = NMFConsensusSelection(data, k_values=self.ranks, solver=self.solver,
                                                                beta_loss=self.beta_loss, iterations=self.iterations,
                                                                nmf_max_iter=self.nmf_max_iter)
            coph, disp = sel_coph.run(processes)
            res['coph'] = coph
            res['disp'] = disp

        if 'conc' in self.methods:
            sel_conc: NMFModelSelection = NMFConcordanceSelection(data, k_values=self.ranks, solver=self.solver,
                                                                   beta_loss=self.beta_loss, iterations=self.iterations,
                                                                   nmf_max_iter=self.nmf_max_iter)
            conc = sel_conc.run(processes)
            res['conc'] = conc

        if 'perm' in self.methods:
            sel_perm: NMFModelSelection = NMFPermutationSelection(data, k_values=self.ranks,
                                                                  solver=self.solver,
                                                                  beta_loss=self.beta_loss, iterations=self.iterations,
                                                                  nmf_max_iter=self.nmf_max_iter)
            perm = sel_perm.run(processes)
            res['perm'] = perm

        if 'split' in self.methods:
            sel_split: NMFModelSelection = NMFSplitHalfSelection(data, k_values=self.ranks,
                                                                 solver=self.solver,
                                                                 beta_loss=self.beta_loss, iterations=self.iterations,
                                                                 nmf_max_iter=self.nmf_max_iter)
            split = sel_split.run(processes)
            res['split'] = split

        if 'mse' in self.methods or 'mad' in self.methods:
            sel_impute: NMFModelSelection = NMFImputationSelection(data, k_values=self.ranks,
                                                                   solver=self.solver,
                                                                   beta_loss=self.beta_loss, iterations=self.iterations,
                                                                   nmf_max_iter=self.nmf_max_iter,
                                                                   metric='both')
            mse, mad = sel_impute.run(processes)
            res['mse'] = mse
            res['mad'] = mad

        return res


class NMFResults:
    """Provide results and output methods from the NMFModel selection process.
    
    Public methods:
    measure(plot_file, table_file)   -- Return cophenetic correlation for each value of k.
                                                       Optionally output a plot and csv to file.

    Instance variables:
    results: List[NMFModelSelectionResults]         -- List of results, one for each value of k searched.
    selected: NMFModelSelectionResults              -- Results for value of k for which the criteria was optimal.
    data: DataFrame                                 -- Source data the model was learned from.
    """

    def __init__(self, results: List[NMFModelSelectionResults], data: pd.DataFrame = None,
                 reduce_fn: Callable[[NMFModelSelectionResults, NMFModelSelectionResults], NMFModelSelectionResults]
                 = lambda x, y: x if x.metric > y.metric else y) -> None:
        """Initialise NMFResults.

        :param results: List of results objects, one for each value of k tested.
        :type results: List[NMFModelSelectionResults]
        :param data: Data which the model was learned from
        :type data: DataFrame
        :param reduce_fn: Function to decide which k is optimal, reduce style function. By default reduces to highest.
        :type reduce_fn: Callable[[NMFModelSelectionResults, NMFModelSelectionResults], NMFModelSelectionResults]
        """
        self.__results: List[NMFModelSelectionResults] = results
        self.__data: pd.DataFrame = data
        opt_k: NMFModelSelectionResults = reduce(
            reduce_fn,
            results
        )
        # select and store optimal k by cophenetic correlation
        self.__selected: NMFModelSelectionResults = opt_k

    @property
    def results(self) -> List[NMFModelSelectionResults]:
        """Return results for all values of k tested."""
        return self.__results

    @property
    def selected(self) -> NMFModelSelectionResults:
        """Return results for optimal k tested."""
        return self.__selected

    @property
    def data(self) -> pd.DataFrame:
        """Return the input data."""
        return self.__data

    def measure(self, plot: str = None, table: str = None
                ) -> pd.DataFrame:
        """Return a table of the rank selection measure against k and output.

        :param plot: location to output a line chart of data to
        :type plot: str
        :param table: location to write a csv table of the data to
        :type table: str
        """
        # See if handling old format results which only store mean value
        if isinstance(self.selected.metric, tuple):
            c: List[float] = [float(x.metric[0]) for x in self.results]
        else:
            c: List[float] = [float(x.metric) for x in self.results]
        ks: List[int] = [x.k for x in self.results]
        arr: np.array = np.array([['k'] + ks, ['measure'] + c])
        data: pd.DataFrame = pd.DataFrame(data=arr[:, 1:], index=arr[:, 0:1])
        # Ensure using float as datatype for the criteria values
        # TEMP TIMING
        for col in data.columns:
            data[col] = data[col].astype('float')
        if plot is not None:
            # Output plot desired
            _ = plt.figure()
            plt.title('Measure across ranks')
            plt.xlabel('k')
            plt.ylabel('measure')
            plt.plot(ks, c)
            plt.savefig(plot)
        if table is not None:
            data.to_csv(table)
        return data

    def result_for_k(self, k: int) -> NMFModelSelectionResults:
        """Return the model selection results for the specified rank k. Useful for exporing ranks near the elbow
        points to identify nost suitable rank.

        :param k: Rank to get results for
        :type k: int
        """

        try:
            return next(x for x in self.results if x.k == k)
        except StopIteration:
            raise Exception(f'No results found for rank {k}')

    def write_results(self, file: str) -> None:
        """Dump this object to a file.

        :param file: Location to write object to.
        :type file: str
        """
        dump(self, file, compress=True)

    @classmethod
    def load_results(cls, file: str) -> NMFResults:
        """Load a serialised NMFResults object.

        :param file: Location to load object from
        :type file: str
        :return: NMFResults object
        :rtype: NMFResults
        """
        return load(file)


class NMFModelSelectionResults:
    """Contain results from NMFModel selection process.

    Public methods:
    write_w             -- Write W matrix to file
    write_h             -- Write H matrix to file
    write_model         -- Pickle model to file
    load_model          -- Factory method to restore an instance of this class from file

    Instance variables:
    k: int              -- number of components
    metric: float       -- value of the model selectiom metric for this number of components
    connectivity:       -- connectivity matrix for consensus based metrics
        ConnectivityMatrix
    h: DataFrame        -- h matrix of model for this k
    w: DataFrame        -- w matrix of model for this k
    model: NMF/wNMF     -- model with lowest reconstruction error for this k
    data: DataFrame     -- data this model was built from
    """

    def __init__(self,
                 k: int,
                 cophenetic_corr: float,
                 connectivity: Optional[pd.DataFrame],
                 w: Optional[pd.DataFrame],
                 h: Optional[pd.DataFrame],
                 model: Optional[NMFModel],
                 data: pd.DataFrame) -> None:
        """Create for NMFModelResults.

        :param k: Value of k tested
        :type k: int
        :param cophenetic_corr: Value of the criteria for this test. While this is named cophenetic corr, will be
                                whichever criteria is appropriate for the method.
        :type cophenetic_corr: float
        :param connectivity: Connectivity matrix across iterations. Will return none for methods which do not use
                             connectivity matrices.
        :type connectivity: DataFrame
        :param w: The W matrix of the model
        :type w: DataFrame
        :param h: The H matrix of the model
        :type h: DataFrame
        :param model: The NMF model trained
        :type model: NMF
        :param data: Data which the model was learned from
        :type data: DataFrame
        """
        self.__k: int = k
        self.__metric: float = cophenetic_corr
        self.__connectivity: pd.DataFrame = connectivity
        self.__w: pd.DataFrame = w
        self.__h: pd.DataFrame = h
        self.__model: Optional[NMFModel] = model
        self.__data: pd.DataFrame = data
        self.__name_dfs()

    @property
    def k(self) -> int:
        """Return k value for these results."""
        return self.__k

    @property
    def metric(self) -> float:
        """Return selection metric for this k."""
        return self.__metric

    @property
    def connectivity(self) -> pd.DataFrame:
        """Return average connectivity matrix for this k."""
        return self.__connectivity

    @property
    def h(self) -> pd.DataFrame:
        """Return w matrix of NMF for this k."""
        return self.__h

    @property
    def w(self) -> pd.DataFrame:
        """Return h matrix of NMF for this k."""
        return self.__w

    @property
    def model(self) -> NMFModel:
        """Return the fitted NMF model."""
        return self.__model

    @property
    def data(self) -> pd.DataFrame:
        """Return the source data."""
        return self.__data

    def __str__(self) -> str:
        """Return string representation of this object."""
        return f'{self.k},{self.metric}'

    def __repr__(self) -> str:
        """Return representation of the this object."""
        return f'<{str(self)}>'

    def __name_dfs(self) -> None:
        """Give the indices W, H the correct names."""
        # W
        if self.__w is not None:
            self.__w.columns = NMFDecomposer.component_names(self.k)
            self.__w.columns.name = 'modules'
            self.__w.index.name = self.__data.index.name
        # H
        if self.__h is not None:
            self.__h.index = NMFDecomposer.component_names(self.k)
            self.__h.index.name = 'modules'
            self.__h.columns.name = self.__data.columns.name

    @staticmethod
    def _write_df(df: pd.DataFrame, file: str = None) -> None:
        """Write a dataframe to a file or stdout.

        :param df: DataFrame to write
        :type df: DataFrame
        :param file: Location to write to
        :type file:  str
        """
        df.to_csv(path_or_buf=file, index=True)

    @staticmethod
    def _heatmap_df(df: pd.DataFrame, file: str = None) -> None:
        """Write a cluster heatmap for a dataframe.

        :param df: DataFrame to write
        :type df: DataFrame
        :param file: Location to write to
        :type file:  str
        """
        _ = plt.figure()
        sns.clustermap(df)
        plt.savefig(file)

    def write_w(self, file: str = None, plot: str = None) -> None:
        """Write w matrix.

        :param file: location to write table to, omit to skip writing table
        :type file: Optional[str]
        :param plot: location to write plot to, omit to skip writing plot
        :type plot: Optional[str]
        """
        if file is not None:
            self._write_df(self.w, file)
        if plot is not None:
            plt.figure()
            sns.clustermap(self.w, xticklabels=True, yticklabels=True)
            plt.subplots_adjust(top=0.96,
                                bottom=0.06,
                                left=0.02,
                                right=0.725,
                                hspace=0.2,
                                wspace=0.2)
            plt.savefig(plot)

    def write_h(self, file: str = None, plot: str = None) -> None:
        """Write h matrix.

        :param file: location to write table to, omit to skip writing table
        :type file: Optional[str]
        :param plot: location to write plot to, omit to skip writing plot
        :type plot: Optional[str]
        """
        if file is not None:
            self._write_df(self.h, file)
        if plot is not None:
            self._heatmap_df(self.h, plot)

    def write_model(self, file: str) -> None:
        """Dump the NMF model file so it can be used later."""
        dump(self, file, compress=True)

    @classmethod
    def load_model(cls, file: str) -> NMFModelSelectionResults:
        """Restore the results from a model file and data."""
        return load(file)


class ConnectivityMatrix:
    """Generate connectivity and consensus matrices from NMF iterations.

    Public methods:
    heatmap                 -- display a heatmap of this consensus matrix
    c_bar                   -- class method, produce a consensus matrix from multiple connectivity matrices
    cophenetic_corr         -- class method, return cophenetic correlation coefficient of a consensus matrix
    dispersion              -- class method, return dispersion coefficient of a consensus matrix

    Instance variables:
    classes: Dict[int, int] -- give 'classification' for each observation, classifying as the component with highest
                               weight
    matrix: DataFrame       -- the connectivity matrix in DataFrame format
    labels: List[int]       -- list of item labels
    """

    def __init__(self, w: np.array) -> None:
        """Return a symmetric connectivity matrix for this h.

        :param w: H matrix of an NMF model
        :type w: np.ndarray
        """
        self.__labels: List[int] = list(range(w.shape[1]))
        self.__classes: Dict[int, int] = self._classify_w(self.labels, w)
        self.__matrix: pd.DataFrame = self._build_matrix()

    @property
    def classes(self) -> Dict[int, int]:
        """Return each item and the component it was classified as."""
        # Should never be set after matrix creation, so no setter included
        return self.__classes

    @property
    def matrix(self) -> pd.DataFrame:
        """Return connectivity matrix and dataframe."""
        return self.__matrix

    @property
    def labels(self) -> List[int]:
        """Return list of item labels."""
        return self.__labels

    @staticmethod
    def _classify_w(labels: List[int], w: np.array) -> Dict[int, int]:
        """Determine which component each site is classified into.

        :param labels: List of labels for each component
        :type labels: List[int]
        :param w: W matrix of an NMF model
        :type w: np.ndarray
        :return: Dictionary associating each item with a class label
        :type: Dict[int, int]
        """

        w_df: pd.DataFrame = pd.DataFrame(w)
        classification: Dict[int, int] = dict(w_df.idxmax(axis=1))
        return classification

    def _build_matrix(self) -> pd.DataFrame:
        """Create a symmetric connectivity matrix as a dataframe.

        :return: Connectivity matrix
        :type: pd.DataFrame
        """
        class_df: pd.DataFrame = pd.DataFrame(self.classes.values(), index=self.classes.keys())
        # Perform a boolean matrix multiplication for each component, and sum them
        bmat: np.ndarray = None
        for label in self.labels:
            bclass: pd.DataFrame = class_df == label
            res: np.ndarray = bclass.T.values * bclass.values
            bmat = res if bmat is None else bmat + res
        return pd.DataFrame(bmat.astype('uint16'), index=class_df.index, columns=class_df.index)

    def heatmap(self, output=None) -> None:
        """Display heatmap of this connectivity matrix."""
        _ = sns.heatmap(self.matrix)
        plt.show()

    @classmethod
    def c_bar(cls, c_matrices: List[ConnectivityMatrix]) -> pd.DataFrame:
        """Generate an element-wise mean matrix of many connectivity matrices.

        :param c_matrices: list of connectivity matrices for with the same dimensions
        :type c_matrices: List[ConnectivityMatrix]
        :return: consensus matrix
        :rtype: DataFrame
        """
        sum_df: pd.DataFrame = reduce(lambda x, y: x.add(y, fill_value=0),
                                      [c.matrix for c in c_matrices])
        return sum_df / len(c_matrices)

    @classmethod
    def cophenetic_corr(cls, c_bar: pd.DataFrame, linkage: str = 'complete') -> float:
        """Return cophenetic correlation coefficient of consensus matrix.

        :param c_bar: consensus matrix (from c_bar class method)
        :type c_bar: DataFrame
        :param linkage: linkage method, as available in scipy.hierarchy.linkage
        :type linkage: str
        :return: cophenetic correlation coefficient
        :rtype: float
        """
        # Transform c_bar into distance rather than similarity matrix
        # Create a lower matrix filled with one values
        width: int = c_bar.values.shape[0]
        distances: np.array = np.ones((width, width))
        distances = distances - c_bar.values
        distances = np.tril(distances)
        # Converted to condensed form used by scipy
        distances = symmetrize(distances)
        distances = distance.squareform(distances)
        # Perform a hierarchical clustering of the matrix
        h_cluster = hierarchy.linkage(distances, linkage)
        # Calculate cophenetic correlation between clustering and distance
        c = hierarchy.cophenet(h_cluster, distances)
        return c

    @classmethod
    def dispersion(cls, c_bar: pd.DataFrame) -> float:
        """Determine dispersion of a consensus matrix, as defined in Kim & Park

        :param c_bar: consensus matrix (from c_bar class method)
        :type c_bar: DataFrame
        :return: dispersion coefficient
        :rtype: float
        """
        # Defined as (1 / n^2 ) * sum((4*(C_ij - 1/2)^2))
        cb: pd.DataFrame = 4 * ((c_bar - 0.5) ** 2)
        dispersion: float = (1 / (cb.shape[0] ** 2)) * cb.values.sum()
        return dispersion


def symmetrize(a: np.ndarray) -> np.ndarray:
    """Return a symmetric matrix from a lower diagonal matrix.
    :param a: Lower diagonal matrix
    :type a: np.ndarray
    :return: A symmetric matrix
    :type: np.ndarray
    """
    return a + a.T - np.diag(a.diagonal())


def load_table(path: str, sep: str = '\t') -> pd.DataFrame:
    """Load and return a dataframe from a delimited file."""
    return pd.read_csv(path, sep=sep)


def shuffle_frame(frame):
    """Shuffle columns and rows of a DataFrame."""
    cols = list(range(0, len(frame.columns)))
    random.shuffle(cols)
    return frame.sample(frac=1).iloc[:, cols]


def tests() -> None:
    """Lazy function to do some testing along the way."""
    # Simulate a large dataset
    N_METAGENES: int = 6
    N_GENES: int = 8000
    N_SAMPLES: int = 25
    P_OVERLAP: float = 0.0
    NOISE: Tuple[float, float] = (0, 1)
    df: pd.DataFrame = synthdata.sparse_overlap_even(rank=N_METAGENES, m_overlap=P_OVERLAP, n_overlap=P_OVERLAP,
                                                     size=(N_GENES, N_SAMPLES), noise=NOISE, p_ubiq=0.0)
    df = shuffle_frame(df)
    # select: NMFModelSelection = (NMFSplitHalfSelection(
    #     df.T, k_min=5, k_max=7, solver='mu', beta_loss='frobenius', iterations=10, nmf_max_iter=10000, metric='ari'
    # ))
    # Load the data we're having errors with
    data = pd.read_csv('/media/hal/safe_hal/work/nmf_otherenv/sediment/input_dump.csv', index_col=0)
    select = NMFMultiSelect(ranks=list(range(2, 10, 1)), beta_loss='kullback-leibler', iterations=150, nmf_max_iter=10000,
                        solver='mu', methods=['coph', 'disp'])
    results = select.run(data)
    print(results['jiang'].selected.k)
    print(results['jiang'].measure().T)

    from mg_nmf.nmf import visualise
    visualise.multiselect_plot(results).show()
    # results.write_results('data/sample.results')

def orientation_tests():
    # Make some synthetic data to play with
    data: pd.DataFrame = synthdata.sparse_overlap_even(rank=12, m_overlap=0.3, n_overlap=0.3,
                                                     size=(2000, 100), noise=(0,4), p_ubiq=0.0, feature_scale=True)
    # Transpose to sk-learn orientation
    data = data.T
    # plt.show()
    # sel: NMFModelSelection = NMFConsensusSelection(data, k_values=[2,6], solver='mu', beta_loss='kullback-leibler',
    #                                            iterations=10)
    opts = NMFOptions(
        nmf_max_iter=3000,
        filter_fn=normalise.variance_filter,
        filter_threshold=0.5,
        normalise=normalise.map_maximum,
        normalise_arg=None
    )
    decomposer = NMFDecomposer(
        data=data,
        options=opts,
        seed=7297108
    )
    sel: NMFModelSelection = NMFConcordanceSelection(
        k_values=[9,10,11,12,13,14,15,16], iterations=5, decomposer=decomposer
    )
    sel2: NMFModelSelection = NMFConsensusSelection(
        k_values=[9,10,11,12,13,14,15,16], iterations=5, decomposer=decomposer
    )
    res1 = sel.run()
    res2 = sel2.run()
    # res2 = sel2.run()
    res1 == res2
    print(res1.results)
    print(res2[0].results)
    print(res2[1].results)

def leukemia():
    pass
    # import pickle
    # Transpose, as in file has features on rows
    # leuk: pd.DataFrame = pd.read_csv('~/nmf/data/surface_data.csv', index_col=0).astype('float64')
    # tara.drop(columns=['description'],inplace=True)
    # select = NMFMultiSelect(ranks=(8, 8), beta_loss='kullback-leibler', iterations=10, nmf_max_iter=10000,
                            # solver='mu', methods=['mad'])
    # results = select.run(leuk)
    # with open('/home/hal/nmf/rerun_surf.res', 'wb') as f:
    #     pickle.dump(results, f)

if __name__ == "__main__":
    import cProfile
    prof = cProfile.Profile()
    prof.enable()
    # tests()
    orientation_tests()
    prof.disable()
    prof.dump_stats("refactor.stats")
    # leukemia()
