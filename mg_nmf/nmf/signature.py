"""Identifying genes associated with identified components.

Methods to determine which genes are associated to component identified by
Non-Negative Matrix Factorization decomposition.
"""

from __future__ import annotations
from typing import List, Tuple
import pandas as pd, math
from mg_nmf.nmf.selection import NMFModelSelectionResults, NMFResults
import scipy.stats as stats

class FeatureSelection():
    """Defines an interface for feature selection methods for NMF results.""" 

    def __init__(self, model: NMFModelSelectionResults, data: pd.DataFrame) -> None:
        """Intialise class."""
        self.__model: NMFModelSelectionResults = model
        self.__data: pd.DataFrame = data
        self.__feature_name = data.index.name

    @property
    def model(self) -> NMFModelSelectionResults:
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
    
    def select(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform feature selection, returns results as data frames. First is the measure, second is p-value if
        possible."""
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

    def __init__(self, model: NMFModelSelectionResults, data: pd.DataFrame) -> None:
        """Initialise this object."""
        super().__init__(model, data)
    
    def select(self) -> pd.DataFrame:
        """Perform selection for each gene in the data."""
        rs: List[Tuple[str, float]] = []
        for index, row in self.model.h.T.iterrows():
            sqrt_k = math.sqrt(self.model.k)
            sum_wij = sum([abs(x) for x in row])
            sum_wij_sq = sum(math.pow(x,2) for x in row)
            specificity = (sqrt_k - (sum_wij/math.sqrt(sum_wij_sq)))/(sqrt_k-1)
            rs.append((index, specificity))
        rs_frame: pd.DataFrame = pd.DataFrame(rs, columns=[self.feature_name, 'specificity'])
        rs_frame.set_index(self.feature_name, inplace=True)
        return (rs_frame, None)
    
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

    def __init__(self, model: NMFModelSelectionResults, data: pd.DataFrame) -> None:
        """Initialise for correlation signature generation."""
        super().__init__(model, data)
    
    def select(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Correlate each component in model with each gene in source data."""
        #TODO: Consider also returning a matrix of p values for correlation
        #Handle one component at a time
        component: str = ''
        r_frame: pd.DataFrame = None
        p_frame: pd.DataFrame = None
        for component in list(self.model.h.index):
            r, p = self._correlate_one(component)
            if r_frame is None:
                r_frame, p_frame = r, p
            else:
                r_frame[component] = r[component]
                p_frame[component] = p[component]
        return (r_frame, p_frame)
    
    def _correlate_one(self, component: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Correlate component with given index to each gene in data."""
        #Get component values by slicing model results
        comp: pd.DataFrame = self.model.w.loc[:, component]
        #Loop over each gene in source data and correlate
        res: List[Tuple[str, float]] = []
        for index, row in self.data.T.iterrows():
            r, p = stats.pearsonr(row, comp)
            res.append((index, r, p))
        r_frame: pd.DataFrame = pd.DataFrame([(x[0], x[1]) for x in res],
                                             columns=[self.feature_name, component]).set_index(self.feature_name)
        p_frame: pd.DataFrame = pd.DataFrame([(x[0], x[2]) for x in res],
                                             columns=[self.feature_name, component]).set_index(self.feature_name)
        return (r_frame, p_frame)

if __name__ == '__main__':
    from mg_nmf.nmf.selection import NMFConsensusSelection
    from mg_nmf.nmf.synthdata import sparse_overlap_even
    # Make a model to work with
    d = sparse_overlap_even((400, 100), 6, 0.0, 0.0, 0.0, (0,1))
    model = NMFConsensusSelection(d, 5, 7, 'mu', 'kullback-leibler', 20, 10000).run()
    model_use = model[0].selected
    r = Correlation(model_use, d).select()
    s = Specificity(model_use, d).select()
    print(r)
