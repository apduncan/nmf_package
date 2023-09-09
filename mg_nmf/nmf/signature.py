"""Identifying genes associated with identified components.

Methods to determine which genes are associated to component identified by
Non-Negative Matrix Factorization decomposition.
"""

from __future__ import annotations
from tkinter import W
from typing import List, Tuple, NamedTuple, Optional
import pandas as pd, math
import numpy as np
import scipy.stats as stats
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde, pearsonr

# from mg_nmf.nmf.selection import NMFModelSelectionResults, NMFModelSelection, NMFModel, NMFOptions, \
#     NMFDecomposer
from mg_nmf.nmf import selection as sel


class SelectionResults(NamedTuple):
    """Feature selection returns some measure of importance, and in some cases an associated p-value."""
    measure: pd.DataFrame
    p: Optional[pd.DataFrame]


class FeatureSelection():
    """Abstract class handle feature importance methods for NMF results.

    Public methods:
    select                              -- generate the feature importance measures

    Instance variables:
    model:NMFModelSelectionResults      -- the model to evaluate
    data: DataFrame                     -- the input data this model was generated from
    feature_name: str                   -- the name of the feature in the input data
    """

    def __init__(self, model: sel.NMFModel, data: pd.DataFrame) -> None:
        """Intialise feature importance method.

        :param model: Model to evaluate
        :type model: DataFrame
        :param data: Input data model was learnt from
        :type data: DataFrame
        """
        self.__model: sel.NMFModel = model
        self.__data: pd.DataFrame = data
        self.__feature_name = data.index.name

    @property
    def model(self) -> sel.NMFModel:
        """Return model."""
        return self.__model

    @property
    def data(self) -> pd.DataFrame:
        """Return data used to train model."""
        return self.__data

    @property
    def feature_name(self) -> str:
        """Return feature name."""
        return self.__feature_name

    def select(self) -> SelectionResults:
        """Perform feature selection, returns results as data frames. First is the measure, second is p-value if
        possible.

        :return: DataFrame giving feature importance, and if available p-values.
        :rtype: Tuple[DataFrame, Optional[DataFrame]]
        """
        pass


class Specificity(FeatureSelection):
    """Find genes which are specific to one component.

    For each gene in the input data, measure the extent to which it is specific
    to one component identified by Non-Negative Matrix Factorisation, using 
    the method explained in Jiang, Xingpeng, Joshua S. Weitz, and Jonathan 
    Dushoff. “A Non-Negative Matrix Factorization Framework for Identifying 
    Modular Patterns in Metagenomic Profile Data.” Journal of Mathematical 
    Biology 64, no. 4 (March 1, 2012): 697–711. 
    https://doi.org/10.1007/s00285-011-0428-2.
    """

    def __init__(self, model: sel.NMFModel, data: pd.DataFrame) -> None:
        """Initialise this object."""
        super().__init__(model, data)

    def select(self) -> SelectionResults:
        """Generate specifity measure.

        :return: Returns the specificty of features for each module, p-values not available.
        :rtype: Tuple[DataFrame, None]
        """
        rs: List[Tuple[str, float]] = []
        for index, row in self.model.h.T.iterrows():
            sqrt_k = math.sqrt(self.model.model.n_components)
            sum_wij = sum([abs(x) for x in row])
            sum_wij_sq = sum(math.pow(x, 2) for x in row)
            specificity = (sqrt_k - (sum_wij / math.sqrt(sum_wij_sq))) / (sqrt_k - 1)
            rs.append((str(index), specificity))
        rs_frame: pd.DataFrame = pd.DataFrame(rs, columns=[self.feature_name, 'specificity'])
        rs_frame.set_index(self.feature_name, inplace=True)
        return SelectionResults(rs_frame, None)


class Correlation(FeatureSelection):
    """Find correlation of gene profile to component profiles across sites.
    
    For each gene in the input data, provide the Pearson correlation between 
    the abundance of that gene across sites in the original data and the 
    abundance of each component in the H matrix of the decomposition.
    Method is as briefly described in 
    Jiang, Xingpeng, Morgan G. I. Langille, Russell Y. Neches, Marie Elliot, 
    Simon A. Levin, Jonathan A. Eisen, Joshua S. Weitz, and Jonathan Dushoff. 
    “Functional Biogeography of Ocean Microbes Revealed through Non-Negative 
    Matrix Factorization.” PLOS ONE 7, no. 9 (September 18, 2012): e43866. 
    https://doi.org/10.1371/journal.pone.0043866.

    """

    def __init__(self, model: sel.NMFModel, data: pd.DataFrame) -> None:
        """Initialise for correlation signature generation."""
        super().__init__(model, data)

    def select(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate correlation feature importance measure.

        :return: Return the correlation of each feature and module, and the p-value for these correlations
        :rtype: Tuple[DataFrame, DataFrame]
        """
        # Handle one component at a time
        component: str = ''
        r_frame: Optional[pd.DataFrame] = None
        p_frame: Optional[pd.DataFrame] = None
        for component in list(self.model.h.index):
            r, p = self._correlate_one(component)
            if r_frame is None:
                r_frame, p_frame = r, p
            else:
                r_frame[component] = r[component]
                p_frame[component] = p[component]
        return SelectionResults(r_frame, p_frame)

    def _correlate_one(self, component: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Correlate component with given index to each gene in data."""
        # Get component values by slicing model results
        comp: pd.DataFrame = self.model.w(self.data).loc[:, component]
        # Loop over each gene in source data and correlate
        res: List[Tuple[str, float, float]] = []
        for index, row in self.data.T.iterrows():
            r, p = stats.pearsonr(row, comp)
            res.append((index, r, p))
        r_frame: pd.DataFrame = pd.DataFrame([(x[0], x[1]) for x in res],
                                             columns=[self.feature_name, component]).set_index(self.feature_name)
        p_frame: pd.DataFrame = pd.DataFrame([(x[0], x[2]) for x in res],
                                             columns=[self.feature_name, component]).set_index(self.feature_name)
        return SelectionResults(r_frame, p_frame)


class LeaveOneOut(FeatureSelection):
    """Use the decrease in correlation between source data and WH with one component left out to assess which
    features are significant to which component."""

    def __init__(self, model: sel.NMFModel, data: pd.DataFrame,
                 fillna: float = 0) -> None:
        """
        :param fillna: Where Pearson correlation is undefined, replace with this value. Set to None to leave undefined.
                       This will mean where a feature only has weight in one component, the LOO change in correlation 
                       will appear as NaN. Such features will not be assigned using ThresholdAssignment. Default fills
                       with 0.
        :type fillna: float
        """
        super().__init__(model, data)
        self.fillna = fillna
    
    @property
    def fillna(self) -> float:
        return self.__fillna
    
    @fillna.setter
    def fillna(self, fillna: float) -> None:
        self.__fillna: float = fillna

    def select(self) -> SelectionResults:
        """Compute correlations between X and WH with one component at a time left out of WH. A decrease in correlation
        compared to full WH indicates that the gene is significant to that component.

        :return: Return the LOOCD of each feature and module, p-value not available
        :rtype: Tuple[DataFrame, None]
        """
        w: pd.DataFrame = self.model.w(self.data)
        wh: pd.DataFrame = w.dot(self.model.h)
        corr_complete: pd.Series = wh.corrwith(self.data, axis=0)
        delta_corr: List[pd.series] = []
        for c_lo in w.columns:
            # c_lo is the component we're interested in and will leave out
            c_include: List[str] = list(filter(lambda x: x != c_lo, w.columns))
            # Make WH with c_lo removed
            rh: pd.DataFrame = self.model.h.loc[c_include]
            rw: pd.DataFrame = w[c_include]
            rcorr: pd.DataFrame = rw.dot(rh).corrwith(self.data, axis=0)
            if self.fillna is not None:
                rcorr = rcorr.fillna(self.fillna)
            r_diff: pd.Series = rcorr - corr_complete
            r_diff.name = c_lo
            delta_corr.append(r_diff)
        return SelectionResults(pd.DataFrame(delta_corr).T, None)


class Permutation(FeatureSelection):
    """Compare the weight of a gene in the component to the distribution of weights learnt from permuted data. Use the
    probability of getting that weight as an indication of how related to a component the gene is.

    Instance variables:
    permute_leanrer: NMFModelSelection      -- Object to performed learning on permuated data
    permutations: int                       -- Number of times to perform permuted learning
    """

    def __init__(self, model: sel.NMFModel, data: pd.DataFrame, permute_options: sel.NMFOptions,
                 permutations: int = 50) -> None:
        """Must provide a model section method with the parameters for training on the permuted data set.

        :param permute_options: Settings to use when decomposing permuted data
        :type permute_learner: NMFOptions
        :param permutations: Number of times to perform permuted learning
        :type permutations: int
        """
        super().__init__(model, data)
        self.permute_options = permute_options
        self.permutations = permutations

    @property
    def permute_options(self) -> sel.NMFOptions:
        return self.__permute_options

    @permute_options.setter
    def permute_options(self, permute_options: sel.NMFModelSelection) -> None:
        self.__permute_options: sel.NMFModelSelection = permute_options

    @property
    def permutations(self) -> int:
        return self.__permutations

    @permutations.setter
    def permutations(self, permutations: int) -> None:
        self.__permutations: int = permutations

    def __run_permutation(self) -> pd.DataFrame:
        """Permute data and run training once."""
        permuted: pd.DataFrame = self.data.copy()
        for col in permuted.columns:
            permuted[col] = permuted[col].sample(frac=1).values
        res: sel.NMFModel = sel.NMFDecomposer(permuted, self.permute_options, cache=False).decompose(
            rank=self.model.model.n_components,
            n=1,
            cached=False
        )[0]
        # Only need the weights of features in components
        return res.h

    def __gene_probability(self, row: pd.Series) -> pd.Series:
        """Compute probabilities for one feature"""
        # Fit normal distribution to these weights
        mu, std = stats.norm.fit(row)
        p: np.array = 1 - stats.norm.cdf(self.model.h[row.name], mu, std)
        p_s: pd.Series = pd.Series(p, index=self.model.h.index, name=row.name)
        return p_s

    def select(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Learn models on permuted data, assess importance of feature to component by looking at probability of model
        weight coming from distribution fitted to weights from permuted data.

        :return: Probability of observed weight being drawn from weights learnt by permutation, no other p-values
        :rtype: Tuple[DataFrame, None]
        """
        # Run permuted learning
        perm_tbl: pd.DataFrame = pd.concat([self.__run_permutation() for i in range(self.permutations)], axis=1)
        # Find probabilities that a given weight would be drawn from distribution of permuted weights
        p: pd.DataFrame = perm_tbl.apply(self.__gene_probability, axis=1)
        return SelectionResults(p, None)


class FeatureAssignment:
    """Defines an interface for methods to perform binary assignments of features to components."""
    def __init__(self, measure: pd.DataFrame) -> None:
        """Intialise class.

        :param measure: Table of the feature importance measure
        :type: DataFrame
        """
        self.__measure: pd.DataFrame = measure

    @property
    def measure(self) -> pd.DataFrame:
        """Return model."""
        return self.__measure

    def assign(self) -> pd.DataFrame:
        """Perform feature assignment, returning a boolean table indicating which features are assigned to a module.

        :return: Boolean table of feature assignments to modules
        :rtype: DataFrame
        """
        pass


class KDEAssignment(FeatureAssignment):
    """Assign genes to one or more components, by fitting a Kernel Density Estimate to the data, looking for a local
    minima, and use that value as a cutoff."""
    def __init__(self, measure: pd.DataFrame, cut_low: bool = True, cut_default: float = 0.05) -> None:
        """
        Initialise
        :param cut_low: Assign low or high values. True for low values, False for high values
        :type cut_low: float
        :param cut_default: Where there is no local minimum, use this value as a default cut point
        :type cut_default: float
        """
        super().__init__(measure)
        self.cut_low = cut_low
        self.cut_default = cut_default
        self.__cut_points = [None] * len(measure.columns)

    @property
    def cut_low(self) -> bool:
        return self.__cut_low

    @cut_low.setter
    def cut_low(self, cut_low: bool) -> None:
        self.__cut_low = cut_low

    @property
    def cut_default(self) -> float:
        return self.__cut_default

    @cut_default.setter
    def cut_default(self, cut_default: float) -> None:
        self.__cut_default: float = cut_default

    @property
    def cut_points(self) -> List[float]:
        return self.__cut_points

    def __assign_component(self, row: pd.Series,) -> pd.Series:
        """Assign genes for one component using KDE method"""
        d = gaussian_kde(row)
        x: np.array = np.linspace(row.min(), row.max(), 500)
        y: np.array = d(x)
        minima: np.array = argrelextrema(y, np.less)[0]
        cut: float
        # If no minima use the cut default
        if len(minima) == 0:
            cut = self.cut_default
        else:
            cut = x[minima[0 if self.cut_low else 0]]
        self.__cut_points[list(self.measure.columns).index(row.name)] = cut
        res: pd.Series = row < cut if self.cut_low else row > cut
        return res

    def assign(self) -> pd.DataFrame:
        """Assign genes to one or more components.
        
        :return: Boolean table of feature assignments to modules
        :rtype: DataFrame
        """
        c_res: List[pd.Series] = self.measure.apply(self.__assign_component, axis=0)
        return c_res


class ThresholdAssignment(FeatureAssignment):
    """Simple assignment based on whether the measure is above or below threshold value."""
    def __init__(self, measure: pd.DataFrame, threshold: float, cut_low: bool = True) -> None:
        super().__init__(measure)
        self.threshold = threshold
        self.cut_low = cut_low

    @property
    def threshold(self) -> float:
        return self.__threshold

    @threshold.setter
    def threshold(self, threshold: float) -> None:
        self.__threshold: float = threshold

    @property
    def cut_low(self) -> bool:
        return self.__cut_low

    @cut_low.setter
    def cut_low(self, cut_low: bool) -> None:
        self.__cut_low: bool = cut_low

    def assign(self) -> pd.DataFrame:
        """Assign features to modules based on specified threshold

        :return: Boolean table of feature assignments to modules
        :rtype: DataFrame
        """
        return self.measure < self.threshold if self.cut_low else self.measure > self.threshold


class GreedyCorrelationAssignment(FeatureAssignment):
    """Assign genes to components by adding to the set of components each gene represents, until the correlation
    between the model with those components and the source data inreases less than the delta threshold amount."""
    def __init__(self, correlation: pd.DataFrame, model: sel.NMFSingleResult, delta: float) -> None:
        super().__init__(correlation)
        self.model = model
        self.delta = delta

    @property
    def model(self) -> sel.NMFSingleResult:
        return self.__model

    @model.setter
    def model(self, model: sel.NMFSingleResult) -> None:
        self.__model: sel.NMFSingleResult = model

    @property
    def delta(self) -> float:
        return self.__delta

    @delta.setter
    def delta(self, delta: float) -> None:
        self.__delta: float = delta

    def __subset_corr(self, subset: List[str], feature: pd.Series) -> float:
        rw: pd.DataFrame = self.model.w[subset]
        rh: pd.DataFrame = self.model.h.loc[subset][feature.name]
        rwh: pd.DataFrame = rw.dot(rh)
        r: Tuple[float, float] = pearsonr(rwh, feature)
        return r[0]

    def __assign_feature(self, feature: pd.Series) -> List[bool]:
        desc: pd.Series = self.measure.loc[feature.name].sort_values(ascending=False)
        # Initialise the set of components this feature belongs to as the one with the highest weight
        assigned: List[str] = [desc.index[0]]
        corr: float = self.__subset_corr(assigned, feature)
        for n_c in desc.index[1:]:
            n_corr: float = self.__subset_corr(assigned + [n_c], feature)
            delta: float = corr - n_corr
            if delta > self.delta:
                assigned.append(n_c)
                corr = n_corr
            else:
                break
        # Convert to a boolean series
        return pd.Series([x in assigned for x in self.model.h.index], index=self.model.h.index, name=feature.name)

    def assign(self) -> pd.DataFrame:
        """Assign features to one or more modules

        :return: Boolean table of feature assignments to modules
        :rtype: DataFrame
        """
        return self.model.data.apply(self.__assign_feature, axis=0).T