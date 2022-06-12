"""
Perform Gene Set Enrichment Analysis (GSEA) of a constructed NMF model. Initially scoped to do GO Term analysis, have
tried to make generic enough to allow analysis by any gene set (Reactome pathways etc...)
"""

# Standard library imports
from __future__ import annotations

import math
import tempfile
from typing import List, Any, Set, Dict, Optional, TextIO, Union, Tuple, Callable, Collection, Iterable

# Third party module imports
from Bio.KEGG.REST import *
import gseapy
from gseapy import gseaplot
from gseapy.gsea import Prerank
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# Local imports
from tqdm import tqdm

from mg_nmf.nmf import selection
from mg_nmf.nmf import signature


class NMFGeneSetEnrichment:
    """Perform GSEA analysis of a constructed NMF model. Uses prerank method from the GSEA module to identify
    enriched gene sets. W matrix of model should be indexed with the identifer of a gene, which is in the gene sets.
    Input to prerank is the Pearson correlation coefficient between the values of the gene in the input data across
    samples, and the weights of each component across samples.
    """

    def __init__(self, model: selection.NMFModelSelectionResults, data: pd.DataFrame, gene_sets: Dict[str, Set[str]],
                 gene_set_metadata: Dict[str, Dict[str, Any]] = None, gene_names: Dict[str, str] = None,
                 label: str = 'analysis', permutation_num: int = 100, outdir: str = None, processes: int = 1,
                 max_size: Optional[int] = 500, min_size: Optional[int] = 15) -> None:
        """ Initialise Gene Set Enrichment object.

        :param model: Results of a model selection process to look for term enrichment in
        :type model: NMFModelSelectionResults
        :param data: Training data the model was built from
        :type data: DataFrame
        :param gene_sets: Set of genes to test for enrichment. Provide as dictionary where key is name of gene set,
                            and value is a set of the all the gene identifiers in the gene set
        :type gene_sets: Dict[str, Set[str]]
        :param gene_set_metadata: Properties associated with each gene set. Provide as dictionary where key is name
                                    of gene set, value is dictionary key value pairs giving properties of gene set.
        :type gene_set_metadata: Dict[str, Dict[str, Any]]
        :param gene_names: Association of gene identifiers to more descriptive gene names.
        :type gene_names: Dict[str, str]
        :param label: A descriptive label for this analysis
        :type label: str
        :param permutation_num: Number of permutations to perform during false positive testing
        :type permutation_num: int
        :param outdir: Directory to output GSEA plots for each term to
        :type outdir: str
        :param processes: Number of processes to use
        :type processes: int
        :param max_size: Largest geneset to consider, where size is the number of genes in that set
        :type max_size: int
        :param min_size: Smallest geneset to consider, where size if the number of genes in that set
        :type min_size: int
        """

        self.model = model
        self.gene_sets = gene_sets
        self.gene_set_metadata = gene_set_metadata
        self.gene_names = gene_names
        self.label = label
        self.data = data
        self.permutation_num = permutation_num
        self.outdir = outdir
        self.processes = processes
        self.max_size, self.min_size = max_size, min_size
        self.__correlation: Optional[Dict[str, pd.DataFrame]] = None
        self.__results: Optional[Dict[str, Prerank]] = None

    @property
    def model(self) -> selection.NMFModelSelectionResults:
        """Model to analyse."""
        return self.__model

    @model.setter
    def model(self, model: selection.NMFModelSelectionResults) -> None:
        """Model to analyse."""
        self.__model: selection.NMFModelSelectionResults = model

    @property
    def data(self) -> pd.DataFrame:
        """Training data model was learnt from."""
        return self.__data

    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        """Training data model was learnt from."""
        self.__data: pd.DataFrame = data

    @property
    def gene_sets(self) -> Dict[str, Set[str]]:
        """Identifiers for sets of genes, and the list of genes in that set."""
        return self.__gene_sets

    @gene_sets.setter
    def gene_sets(self, gene_sets: Dict[str, Set[str]]) -> None:
        """Identifiers for sets of genes, and the list of genes in that set."""
        self.__gene_sets = gene_sets

    @property
    def gene_set_metadata(self) -> Dict[str, Set[str]]:
        """Metadata associated with each gene set. Will be added onto returned results."""
        return self.__gene_set_metadata

    @gene_set_metadata.setter
    def gene_set_metadata(self, gene_set_metadata: Dict[str, Set[str]]) -> None:
        """Metadata associated with each gene set. Will be added onto returned results."""
        self.__gene_set_metadata = gene_set_metadata

    @property
    def gene_names(self) -> Dict[str, str]:
        """Provide a more human readable name for gene (function) identifiers. """
        return self.__gene_names

    @gene_names.setter
    def gene_names(self, gene_names: Dict[str, str]) -> None:
        """Set a more human readable name for gene (function) identifiers. """
        self.__gene_names: Dict[str, str] = gene_names

    @property
    def permutation_num(self) -> int:
        """Number of permutations to perform to evaluate significance."""
        return self.__permutation_num

    @permutation_num.setter
    def permutation_num(self, permutation_num: int) -> None:
        """Number of permutations to perform to evaluator significance."""
        self.__permutation_num: int = permutation_num

    @property
    def outdir(self) -> str:
        """Where to write enrichment analysis plots. If not set, write to a temporary folder."""
        return self.__outdir

    @outdir.setter
    def outdir(self, outdir: str) -> None:
        """Where to write enrichment analysis plots. If not set, write to temporary folder."""
        self.__outdir: str = outdir

    @property
    def processes(self) -> int:
        """Number of processes to use when running analysis."""
        return self.__processes

    @processes.setter
    def processes(self, processes: int) -> None:
        """Number of processes to use when running analysis."""
        self.__processes: int = processes

    @property
    def label(self) -> str:
        """Descriptive label for this analysis."""
        return self.__label

    @label.setter
    def label(self, label: str) -> None:
        """Descriptive label for this analysis."""
        self.__label: str = label

    @property
    def correlation(self) -> Dict[str, pd.DataFrame]:
        """Return correlations used as input to prerank. """
        # Ensure analysis has run
        _ = self.results(0.05)
        return self.__correlation

    @property
    def max_size(self) -> int:
        """Return maximum geneset size to be considered. """
        return self.__max_size

    @max_size.setter
    def max_size(self, max_size: Optional[int]) -> None:
        """Set the maximum genset size to be considered. """
        if max_size is None:
            max_size = GeneSets.geneset_size_bounds(self.gene_sets)[1]
        self.__max_size: int = max_size

    @property
    def min_size(self) -> int:
        """Return minimum geneset size to be considered. """
        return self.__min_size

    @min_size.setter
    def min_size(self, min_size: Optional[int]) -> None:
        """Set the minimum geneset size to be considered. """
        if min_size is None:
            min_size = GeneSets.geneset_size_bounds(self.gene_sets)[0]
        self.__min_size: int = min_size

    def analyse(self, processes: int = None) -> Dict[str, Prerank]:
        """Run GSEA analysis

        :param processes: number of processes to run while performing analysis
        :type processes: int
        """

        # 1. Get correlations for prerank input, drop any nan rows
        correlation, _ = signature.Correlation(self.model, self.data).select()
        # Drop any nan, must only include real value correlations
        self.__correlation = correlation.dropna(axis=0, how='all')
        # 2. Perform enrichment analysis for each component
        prerank_results: Dict = {}
        # Determine whether to output a folder or need to make temp folder
        outdir: str = self.outdir
        tmpdir: tempfile.TemporaryDirectory = None
        # Number of processes
        if processes is None:
            processes = self.processes
        if outdir is None:
            tmpdir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory()
            outdir = tmpdir.name
        for component in self.__correlation:
            corr = pd.DataFrame(self.__correlation[component].sort_values(ascending=False).reset_index().values)
            # Convert datatypes
            corr = corr.astype({0: 'str', 1: 'float64'})
            result = gseapy.prerank(rnk=corr, gene_sets=self.gene_sets, permutation_num=self.permutation_num,
                                    outdir=outdir, format='png', processes=processes, max_size=self.max_size,
                                    min_size=self.min_size)
            prerank_results[component] = result
        # Close temporary directory if used
        if tmpdir is not None:
            tmpdir.cleanup()
        self.__results: Dict[str, Prerank] = prerank_results
        return prerank_results

    def results(self, significance: float = 0.05) -> pd.DataFrame:
        """Return a table of results meeting the specificed significance level.

        :param significance: significance level to report results at or under, 0 - 1
        :type significance: float
        """

        # If analysis hasn't been run, run it now
        if self.__results is None:
            self.__results = self.analyse(processes=self.processes)
        # Loop through each component and make a filtered DataFrame
        comp_df: List[pd.DataFrame] = []
        for component, result in self.__results.items():
            df: pd.DataFrame = result.res2d[result.res2d['fdr'] <= significance]
            if len(df) > 0:
                df.loc[:, 'component'] = component
                comp_df.append(df)
        # Stack the data from each component
        merged: pd.DataFrame = pd.concat(comp_df, axis=0)
        # Add on the gene set metadata
        merged = self._add_gs_metadata(merged)
        return merged

    def gene_name(self, gene_id: str, max_len: int = 40) -> str:
        """Return a gene name if known for a gene id, if not known return empty string.

        :param gene_id: Identifier for gene
        :type gene_id: str
        :param max_len: Maximum length of name string. Will trim to this length and append ... if exceeded.
        :type max_len: int
        :return: Name of the gene, trimmed to specified length
        :rtype: str
        """
        if self.gene_names is None:
            return ""
        if gene_id not in self.gene_names:
            return "Unknown"
        name: str = self.__gene_names[gene_id]
        if len(name) > max_len:
            name = name[:max_len - 3] + '...'
        return name

    def _add_gs_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add anything from the provided gene set metadata to a given dataframe."""

        # If no metaaata, just return input data
        if self.gene_set_metadata is None:
            return data

        # Performed via a join. First make a DataFrame of the metadata
        index, items = list(self.gene_set_metadata.keys()), list(self.gene_set_metadata.values())
        md_df: pd.DataFrame = pd.DataFrame(items, index=index)
        return data.join(md_df, how='left')

    def plot_enrichment(self, data: pd.DataFrame, group: str = None, label: str = None, width: int = 800,
                        height: int = 1000,
                        l_margin: int = 500) -> go.Figure:
        """Generate a plot of enriched / depleted terms, from the output an NMFGeneSetEnrichment.results() call.

        :param data: results from a call of the results() method
        :type data: DataFrame
        :param group: the column to group enrichments by (often a metadata column, like namespace)
        :type group: str
        :param label: a descriptive column to add to the term id in the plot
        :type label: str
        :param width: width of figure, in pixels
        :type width: int
        :param height: height of figure, in pixels
        :type height: int
        :param l_margin: width of left margin, adjust to fit all text
        :type l_margin: int
        :return: heatmap figure divided up into groups
        :rtype: plotly figure
        """

        # Validate that group exists
        if group not in data.columns and group is not None:
            raise Exception(f'Column {group} not found in data while plotting enrichment results.')
        # If no group provided, add a column to use for grouping into one group (to standardise operations)
        df: pd.DataFrame = data.copy()
        if group is None:
            df.loc[:, 'group_col'] = 'enrichment analysis'
            group = 'group_col'
        # Determine how many individual groups are in the requested column
        groups: List[str] = list(set(df[group]))
        # Determine how many entries are in each group
        group_spc: pd.DataFrame = df.groupby(group).count()
        group_spc_list: List[int] = list(group_spc.loc[groups]['component'])
        group_spc_prop: List[float] = [x / sum(group_spc_list) for x in group_spc_list]
        # Set up some subplots to slot the different groups into
        fig: go.Figure = make_subplots(rows=len(groups), cols=1, subplot_titles=groups, row_heights=group_spc_prop,
                                       vertical_spacing=0.05)

        # Produce a heatmap for each group
        row: int = 1
        idx_cols = ['index'] if label is None else ['index', label]
        for group_name in groups:
            # Reduce to results only from this group
            df_reduce: pd.DataFrame = df[df[group] == group_name].reset_index()
            df_piv: pd.DataFrame = df_reduce.pivot(index=idx_cols, columns='component', values='nes')
            # Order by component with max enrichment
            df_piv.loc[:, 'max'] = df_piv.idxmax(axis=1)
            df_piv = df_piv.sort_values(by='max')
            df_piv.drop(columns=['max'], inplace=True)
            # Join term name + id
            df_deind: pd.DataFrame = df_piv.reset_index()
            y_labels: List[str] = df_deind['index'] if label is None else df_deind[label] + ' (' + df_deind[
                'index'] + ')'
            # Add to plot
            fig.add_trace(go.Heatmap(
                z=df_piv.values,
                x=df_piv.columns,
                y=y_labels,
                coloraxis='coloraxis'
            ), row=row, col=1)
            row += 1

        # Format colorbar to be shared, perform some layout on plot
        # Show all labels on axis
        fig.update_yaxes(dtick=1)
        fig.update_layout(coloraxis=dict(colorscale='RdBu_r', cmid=0), width=width, height=height,
                          margin=dict(l=l_margin, t=30, b=0))
        return fig

    def plot_geneset_correlation(self, component: str, gene_set_id: str, cols: int = 4, width: int = 800,
                                 height: int = 175, vspace: float = 0.02) -> go.Figure:
        """Produce a series of scatter plots, showing the correlations underlying the GSEA analysis, for a given
        component in the model and gene set.

        :param component: Component name
        :type component: str
        :param gene_set_id: Gene set identifier
        :type gene_set_id: str
        :param cols: Number of columns in grid of plots
        :type cols: int
        :param width: Width of overall figure
        :type width: int
        :param height: Height of one row of plots
        :type height: int
        :param vspace: Vertical spacing
        :type vspace: float
        :return: Grid layout with scatter plots showing correlations between input data and component weight.
        :rtype: Figure
        """

        # Get gene ids which make up the gene set
        if gene_set_id not in self.gene_sets:
            raise Exception(f'Gene set {gene_set_id} not found in gene sets.')
        gene_ids: Set[str] = set(self.gene_sets[gene_set_id])
        # Find which gene ids are in this gene set and in the correlations
        gene_isect: Set[str] = gene_ids.intersection(set(self.__correlation.index))
        # Sort correlation for component by r value
        r_comp: pd.DataFrame = self.correlation[component].loc[gene_isect].sort_values(ascending=False)
        # Make titles for each individual plot
        labels = [f'{x[0]} (r={x[1]:.3f})<br>{self.gene_name(x[0], 40)}' for x in zip(r_comp.index, r_comp)]

        # Set up subplots
        num_rows: int = math.ceil(len(gene_isect) / cols)
        vspace_calc: float = 0.05 * (8 / num_rows)
        fig: go.Figure = make_subplots(cols=cols, rows=num_rows, vertical_spacing=vspace_calc,
                                       subplot_titles=labels)
        # Produce datafor all the gene ids in the gene set, and join
        data: List = []
        for gene_id in gene_isect:
            dval: pd.Series = self.data[gene_id]
            mval: pd.Series = self.model.w[component]
            samples: List[str] = list(self.model.w.index)
            joined: List[Tuple[str, float, float, str]] = list(zip([gene_id] * len(dval), dval, mval, samples))
            data += joined
        df: pd.DataFrame = pd.DataFrame(data, columns=['gene_id', 'data', 'model', 'sample'])

        # Make subplots
        row, col = 1, 1
        # Iterate through in order of descending r
        for gene_id in r_comp.index:
            gene_id_data: pd.DataFrame = df[df['gene_id'] == gene_id]
            rval: float = r_comp.at[gene_id]
            fig.add_trace(go.Scatter(x=gene_id_data['data'], y=gene_id_data['model'], mode='markers', name='data',
                                     text=gene_id_data['sample'],
                                     marker=dict(color=[rval] * len(gene_id_data), coloraxis='coloraxis')
                                     ),
                          row=row, col=col)
            col += 1
            if col > cols:
                col = 1
                row += 1
        # Sort out shared coloraxis
        fig.update_layout(coloraxis=dict(colorscale='RdBu_r', cmid=0), title=f'Correlations underlying {gene_set_id}',
                          height=height * row, width=width, showlegend=False, template='plotly_dark')
        for i in fig['layout']['annotations']:
            i['font'] = dict(size=10)
        return fig

    def plot_gsea(self, component: str, gene_set_id: str, ofname: str) -> None:
        """Return the GSEA plot for a term in a component.

        :param component: Component name
        :type component: str
        :param gene_set_id: Gene set identifier
        :type gene_set_id: str
        :param ofname: Output file name
        :type ofname: str
        """

        # Ensure results have been calculated
        _ = self.results(0.05)
        gs_res: Prerank = self.__results[component]
        gseaplot(gs_res.ranking, gene_set_id, **gs_res.results[gene_set_id], ofname=ofname)


class GeneSets(object):
    """Provide loaders for some standard gene sets, and mappings to genesets. Implemented as a Singleton. """

    # Hold the single instance of the class
    __instance: GeneSets = None

    # Constants for file locations - urls for public, local for downloaded or constructed
    LOCAL_PATH: str = 'gs_data'
    IPR2GO_URL: str = 'http://www.geneontology.org/external2go/interpro2go'
    IPR2GO_LOCAL: str = 'interpro2go'
    IPR_ID2GO_LOCAL: str = 'interpro_id2go'
    PFAM2GO_URL: str = 'http://current.geneontology.org/ontology/external2go/pfam2go'
    PFAM2GO_LOCAL: str = 'pfam2go'
    PFAM_ID2GO_LOCAL: str = 'pfam_id2go'
    GO_OBO_LOCAL: str = 'go-basic.obo'
    KO2PATHWAY_LOCAL: str = 'ko2pathway'

    # This constructor implements the Singleton pattern
    def __new__(cls) -> GeneSets:
        if GeneSets.__instance is None:
            GeneSets.__instance = object.__new__(cls)
        return GeneSets.__instance

    def __init__(self) -> None:
        self.genesets: Dict[str, Dict[str, Set[str]]] = {}
        self.geneset_metadata: Dict[str, Dict[str, Any]] = {}
        self.geneid_names: Dict[str, Dict[str, str]]

    def _local_file(self, filename: str) -> str:
        """Return the full path to look for a local file in.

        :param filename: Filename to look in local store for
        :type filename:  str
        :return: Full path of file located
        :rtype: str
        """
        # If local folder doesn't exist, try to make
        if not os.path.isdir(self.LOCAL_PATH):
            os.mkdir(self.LOCAL_PATH)
        return os.path.join(self.LOCAL_PATH, filename)

    def _get_interpro2go(self) -> TextIO:
        """Open and return file which maps interpro to go terms, download from the interpro website if needed.

        :return: Text stream object of Interpro2GO mappings
        :type: TextIO
        """

        ipr2go_local: str = self._local_file(self.IPR2GO_LOCAL)
        id2go_local: str = self._local_file(self.IPR_ID2GO_LOCAL)
        if not os.path.isfile(ipr2go_local):
            # Get interpro2go file
            r = requests.get(self.IPR2GO_URL)
            with open(ipr2go_local, 'wb') as f:
                f.write(r.content)

            # Convert to a more usable format
            mapping = {}
            with open(ipr2go_local, 'r') as f:
                for file_line in f:
                    # Ignore lines beginning !, thesea are comments
                    trim_line = file_line.strip()
                    if file_line[0] == '!':
                        continue
                    # Extract IPR
                    split = [x.strip() for x in trim_line.split(' > ')]
                    curr_ipr = split[0].split(' ')[0].replace('InterPro:', '')
                    curr_go = split[1].split(';')[1].strip()
                    if curr_ipr not in mapping:
                        mapping[curr_ipr] = []
                    mapping[curr_ipr].append(curr_go)

            # Load the mapping of IPR ids to go terms
            with open(id2go_local, 'w+') as f:
                for write_ipr, write_gos in mapping.items():
                    file_line = f'{write_ipr}\t{";".join(write_gos)}\n'
                    f.write(file_line)
        return open(id2go_local, 'r')

    def geneset_ipr2go(self, namespace: str = None) -> Dict[str, Set[str]]:
        """Load a geneset which maps GO terms -> Set of IPR accessions.

        :param namespace: GO namespace (e.g. biological_process) to limit mapping to.
        :type namespace: str
        :return: Dictionary with key being GO term and value the IPR accession which map to that GO term.
        :rtype: Dict[str, Set[str]]
        """

        gs_key: str = 'ipr2go'
        if gs_key in self.genesets:
            return self.genesets[gs_key]

        mapping: Dict[str, Set[str]] = {}
        with self._get_interpro2go() as f:
            for file_line in f:
                file_ipr, file_gos = [x.strip() for x in file_line.split('\t')]
                for file_go in file_gos.split(';'):
                    if file_go not in mapping:
                        mapping[file_go] = set()
                    mapping[file_go].add(file_ipr)

        self.genesets[gs_key] = mapping
        return mapping

    def geneset_metadata_go(self, namespace: Union[str, List[str]] = None, terms: List[str] = None) \
            -> Dict[str, Dict[str, Any]]:
        """Return metadata (name, namespace, etc) for each GO Terms. If namespace or terms specified, only return
        for those which are in both.

        :param namespace: Namespace(s) to restrict terms to (e.g. molecular_function)
        :type namespace: str, List[str]
        :param terms: Terms to restrict returned metadata to
        :type terms: List[str]
        :return: Dictionary where key is GO term identifier, value is a dictionary of term properties (name etc)
        :rtype: Dict[str, Dict[str, Any]]
        """

        from goatools.base import download_go_basic_obo
        from goatools.obo_parser import GODag, GOTerm

        obo_fname = download_go_basic_obo(obo=self._local_file(self.GO_OBO_LOCAL))
        obodag = GODag(self._local_file(self.GO_OBO_LOCAL))

        # If specific terms requested, reduce to those
        selected_terms: List[GOTerm]
        if terms is not None:
            selected_terms = [obodag[x] for x in terms]
        else:
            selected_terms = [obodag[x] for x in obodag]

        # Reduce to only those which are in the requested namespaces
        if namespace is not None:
            # Make namespace into a list of one item if a string provided
            if isinstance(namespace, str):
                namespace = [namespace]
            selected_terms = [x for x in selected_terms if x.namespace in namespace]

        # Map to dictionary with key parameters
        return dict(zip([x.id for x in selected_terms],
                        [dict(name=x.name, depth=x.depth, namespace=x.namespace) for x in selected_terms]))

    def geneid_names_interpro(self) -> Dict[str, str]:
        """Get a mapping of interpro accession -> name as a dictionary.

        :return: Dicionary where key is Interpro accession, value is name
        :rtype: Dict[str, str]
        """

        # Ensure the files have been downloaded
        with self._get_interpro2go() as f:
            # Don't want to use this file
            pass

        with open(self._local_file(self.IPR2GO_LOCAL), 'r') as f:
            data = []
            for line in f:
                if line[0] == '!':
                    continue
                split = line.strip().split(' ')
                ipr = split[0].split(':')[1]
                name = ' '.join(split[1:])
                name = name.split('>')[0]
                data.append((ipr, name))
        df = pd.DataFrame(data, columns=['ipr', 'name'])
        df = df.set_index('ipr')
        df = df.drop_duplicates()

        return dict(zip(df.index, df['name']))

    def _get_pfam2go(self) -> TextIO:
        """Open and return file which maps interpro to go terms, download from the interpro website if needed.

        :return: Text stream of Pfam2GO mapping
        :type: TextIO
        """

        pfam2go_local: str = self._local_file(self.PFAM2GO_LOCAL)
        id2go_local: str = self._local_file(self.PFAM_ID2GO_LOCAL)
        if not os.path.isfile(pfam2go_local):
            # Get interpro2go file
            r = requests.get(self.PFAM2GO_URL)
            with open(pfam2go_local, 'wb') as f:
                f.write(r.content)

            # Convert to a more usable format
            mapping = {}
            with open(pfam2go_local, 'r') as f:
                for file_line in f:
                    # Ignore lines beginning !, thesea are comments
                    trim_line = file_line.strip()
                    if file_line[0] == '!':
                        continue
                    # Extract IPR
                    split = [x.strip() for x in trim_line.split(' > ')]
                    curr_pfam = split[0].split(' ')[0].replace('Pfam:PF', 'pfam')
                    curr_go = split[1].split(';')[1].strip()
                    if curr_pfam not in mapping:
                        mapping[curr_pfam] = []
                    mapping[curr_pfam].append(curr_go)

            # Load the mapping of IPR ids to go terms
            with open(id2go_local, 'w+') as f:
                for write_pfam, write_gos in mapping.items():
                    file_line = f'{write_pfam}\t{";".join(write_gos)}\n'
                    f.write(file_line)
        return open(id2go_local, 'r')

    def geneset_pfam2go(self, namespace: str = None) -> Dict[str, Set[str]]:
        """Load a geneset which maps GO terms -> Set of Pfam accessions. 
        
        :param namespace: GO namespace (e.g. biological_process) to limit mapping to.
        :type namespace: str
        :return: Dictionary with key being GO term and value the IPR accession which map to that GO term.
        :rtype: Dict[str, Set[str]]
        """

        gs_key: str = 'pfam2go'
        if gs_key in self.genesets:
            return self.genesets[gs_key]

        mapping: Dict[str, Set[str]] = {}
        with self._get_pfam2go() as f:
            for file_line in f:
                file_pfam, file_gos = [x.strip() for x in file_line.split('\t')]
                for file_go in file_gos.split(';'):
                    if file_go not in mapping:
                        mapping[file_go] = set()
                    mapping[file_go].add(file_pfam)

        self.genesets[gs_key] = mapping
        return mapping

    def geneid_names_pfam(self) -> Dict[str, str]:
        """Get a mapping of Pfam accession -> name as a dictionary.

        :return: Dicionary where key is Pfam accession, value is name
        :rtype: Dict[str, str]
        """

        # Ensure the files have been downloaded
        with self._get_pfam2go() as f:
            # Don't want to use this file
            pass

        with open(self._local_file(self.PFAM2GO_LOCAL), 'r') as f:
            data = []
            for line in f:
                if line[0] == '!':
                    continue
                split = line.strip().split(' ')
                pfam = split[0].split(':')[1]
                pfam = pfam.replace('PF', 'pfam')
                name = ' '.join(split[1:])
                name = name.split('>')[0]
                data.append((pfam, name))
        df = pd.DataFrame(data, columns=['pfam', 'name'])
        df = df.set_index('pfam')
        df = df.drop_duplicates()

        return dict(zip(df.index, df['name']))

    def _kegg_cache_fetch(self, file: str) -> Optional[TextIO]:
        """Fetch a KEGG API responses from local cache if available, otherwise return None."""
        local: str = self._local_file(file)
        if not os.path.exists(local):
            return None
        if os.path.getsize(local) <= 0:
            return None
        return open(local, 'rt')

    def _kegg_cache_put(self, file: str, content: TextIO) -> None:
        """Write to a local cache file"""
        local: str = self._local_file(file)
        with open(local, 'wt') as f:
            f.writelines(content.readlines())

    def _kegg_list_resp(self, database: str, org: Optional[str]) -> TextIO:
        """Fetch a list of what is contained in a given KEGG database."""
        cache_local: str = f'kegg_list_{database}' + '' if org is None else f'_{org}'
        # Attempt to load from cache first
        resp: TextIO = self._kegg_cache_fetch(cache_local)
        if resp is None:
            # Fetch from API
            resp: TextIO = kegg_list(database) if org is None else kegg_list(database, org)
            # Write, then load (clumsy but ensures file and pointer in right place)
            self._kegg_cache_put(cache_local, resp)
            resp = self._kegg_cache_fetch(cache_local)
        return resp

    def _kegg_list(self, database: str, org: Optional[str]) -> Collection[str]:
        with self._kegg_list_resp(database, org) as f:
            pathway_list: List[str] = [x.split('\t')[0].split(':')[1] for x in f.readlines()]
        return pathway_list

    def _kegg_link_resp(self, target: str, source: str) -> TextIO:
        """Fetch a list of items which are linked to a given entry."""
        cache_local: str = f'kegg_link_{target}_{source}'
        # Attempt to load from cache first
        resp: TextIO = self._kegg_cache_fetch(cache_local)
        if resp is None:
            # Fetch from API
            resp: TextIO = kegg_link(target, source)
            # Write, then load (clumsy but ensures file and pointer in right place)
            self._kegg_cache_put(cache_local, resp)
            resp = self._kegg_cache_fetch(cache_local)
        return resp

    def _kegg_link(self, target: str, source: str) -> Collection[str]:
        """Fetch a list of items which are linked to a given entry."""
        def line_to_id(line: str) -> Optional[str]:
            cols: List[str] = line.strip().split('\t')
            if len(cols) < 2:
                return None
            item: List[str] = cols[1].split(':')
            if len(cols) < 2:
                return None
            return item[1]
        with self._kegg_link_resp(target, source) as f:
            item_list: List[str] = [line_to_id(x) for x in f.readlines()]
        item_list = [x for x in item_list if x is not None]
        return item_list

    def geneset_ko2pathway(self, limit_pathways: Union[Callable[[str], bool], Collection] = lambda name: True,
                           org: Optional[str] = None) -> Dict[str, Set[str]]:
        """Mapping of KEGG pathways to the KEGG Ortholog terms within that pathway. Can select a subset of all
        pathways.

        :param limit_pathways:  Select which pathways to consider. Either a list of identifiers, or a method which 
                                can be passed a pathway identifier and return a boolean indicating whether to include.
        :type limit_pathways:   Union[Callable[[str], bool], Collection]
        :param org: Organism to list pathways within. Maybe useful if not looking at metagenomics
        :type org: Optional[str]
        :return: Dictionary with key being KEGG Pathway and value the KO term which map to that pathway.
        :rtype: Dict[str, Set[str]]
        """
        # We need to interact with the KEGG API to get details, as we'll assume users (like me) don't have access to
        # the KEGG database / FTP
        # Retrieve a list of all pathways in the databse & organism
        pathway_list: Collection[str] = self._kegg_list('pathway', org)
        # Restrict based on list or predicate
        if isinstance(limit_pathways, Iterable):
            pathway_list = set(pathway_list).intersection(set(limit_pathways))
        if isinstance(limit_pathways, Callable):
            pathway_list = list(filter(limit_pathways, pathway_list))
        # For each pathway, retrieve KEGG Ortholog terms which are in the pathways
        pathway_dict: Dict[str, Set[str]] = {}
        for pathway in tqdm(pathway_list, desc='Fetch KEGG pathways'):
            pathway_dict[pathway] = set(self._kegg_link('ko', pathway))
        return pathway_dict

    def geneset_metadata_kegg_pathways(self, org: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Metadata for KEGG pathways (just provides name currently)

        :param org: Organism to list pathways within. Maybe useful if not looking at metagenomics
        :type org: Optional[str]
        :return: Dictionary with key being KEGG Pathway and value the KO term which map to that pathway.
        :rtype: Dict[str, Set[str]]
        """
        kegg_resp: TextIO = self._kegg_list_resp('pathway', org)
        kegg_md: Dict[str, Dict[str, Any]] = {}
        for line in (x.strip() for x in kegg_resp.readlines()):
            cols: List[str] = line.split('\t')
            pathway: str = cols[0].split(':')[1]
            name: str = cols[1]
            kegg_md[pathway] = dict(name=name)
        return kegg_md

    def geneid_names_ko(self) -> Dict[str, str]:
        """Names for each KO term (full name, includes EC number where given)

        :return: Dicionary where key is KO term, value is name
        :rtype: Dict[str, str] 
        """
        resp: TextIO = self._kegg_list_resp('ko', None)
        resp_dict: Dict[str, str] = {}
        for line in (x.strip() for x in resp.readlines()):
            cols: List[str] = line.split('\t')
            ko: str = cols[0].split(':')[1]
            name: str = cols[1]
            resp_dict[ko] = name
        return resp_dict

    @classmethod
    def geneset_size_bounds(self, geneset: Dict[str, Set[str]]) -> Tuple[int, int]:
        """Get the size bounds (min, max) of the genesets, where size is number of genes in the geneset.

        :param geneset: The geneset to find bounds of.
        :type geneset: Dict[str, Set[str]]
        """

        size_list: List[int] = [len(y) for x, y in geneset.items()]
        return min(size_list), max(size_list)


if __name__ == '__main__':
    from pickle import dump, load
    from mg_nmf.nmf.selection import NMFResults, NMFModelSelectionResults

    # # Load some premade model results
    # MODEL_RES = '/home/hal/nmf/rerun_surf.res'
    MODEL_RES = '/media/hal/safe_hal/work/nmf_otherenv/sediment/error_model.pickle'
    # res = selection.NMFModelSelectionResults.load_model(MODEL_RES)
    with open(MODEL_RES, 'rb') as f:
        res = load(f)
        # res = res['coph'].selected
    # Load a custom genset
    with open('/media/hal/safe_hal/work/nmf_otherenv/sediment/error_geneset.pickle', 'rb') as f:
        resgenes = load(f)
    # res = res['coph'].selected
    # Make an analysis object
    analysis: NMFGeneSetEnrichment = NMFGeneSetEnrichment(model=res, data=res.data,
                                                          gene_sets=resgenes,
                                                          processes=4, min_size=5,
                                                          max_size=None)
    # analysis: NMFGeneSetEnrichment = NMFGeneSetEnrichment(model=res, data=res.data,
    #                                                       gene_sets=GeneSets().geneset_ko2pathway(),
    #                                                       gene_set_metadata=GeneSets().geneset_metadata_kegg_pathways(),
    #                                                       gene_names=GeneSets().geneid_names_ko(),
    #                                                       label='go', processes=4, min_size=5,
    #                                                       max_size=None)
    anres = analysis.results(0.05)
    # with open('../tests/data/large_enrichment', 'rb') as f:
    #     analysis: NMFGeneSetEnrichment = load(f)
    pd.set_option('display.max_columns', None)
    res = analysis.results(significance=0.05)
    print(analysis.results(significance=0.05))
    analysis.plot_enrichment(res, group='namespace', label='name').show()
    analysis.plot_geneset_correlation('c1', 'GO:0009055').show()
    analysis.plot_gsea('c1', 'GO:0009055', ofname='t.png')
    # dump(analysis, open('/home/hal/Dropbox/PHD/FunctionalAbundance/nmf/data/metatranscriptome_k6.enrich', 'wb+'))
