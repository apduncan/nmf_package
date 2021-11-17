#!/usr/bin/env python3
"""Perform NMF and model selection for a given set of gene abundance.

Takes in a matrix of gene abundance at different stations, and returns a NMF 
matrix decomposition with optimal k as defined by cophenetic correlation method 
(Brunet, 2004). The decomposition with lowest reconstruction error for optimal 
k is returned.

Provides output in several forms:
    * W & H matrices 
    * W & H clustered heatmaps
    * plot of cophenetic correlation vs k & average connectivity matrix for
      optimal k
    * (optional) plots of components versus a table of metadata
"""

# Standard imports
import argparse
import json
import os
import pickle
from typing import Optional, List, Tuple, Dict

# Third party imports
import pandas as pd
import plotly.graph_objects as go

# Local imports
from mg_nmf.nmf.selection import NMFResults, NMFMultiSelect
import mg_nmf.nmf.visualise as visualise

def main():
    """Run the decomposition and plotting for pathway gene abundance data."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=argparse.FileType('r'), required=True,
                        help='Data to perform decomposition for. Not required if loading results.')
    parser.add_argument('--k_min', type=int, default=2, nargs='?', help='Minimum number of components to try.')
    parser.add_argument('--k_max', type=int, default=10, help='Maximum number of components to try.')
    parser.add_argument('--iter', type=int, default=100, help='Number of iterations per value of k for each method.')
    parser.add_argument('--nmf_iter', type=int, default=10000,
                        help='During NMF decomposition, max iterations to allow before convergence')
    parser.add_argument('-s', '--solver', type=str, default='cd', choices=['cd', 'mu'],
                        help='Solver to use during NMF decomposition.')
    parser.add_argument('-b', '--beta_loss', type=str, default='frobenius', choices=['frobenius', 'kullback-leibler'],
                        help='Beta loss function to use during NMF decomposition.')
    parser.add_argument('-m', '--methods', type=str, nargs='*', choices=NMFMultiSelect.PERMITTED_METHODS,
                        default=NMFMultiSelect.PERMITTED_METHODS,
                        help='Model selection methods to perform.')
    parser.add_argument('-o', '--output', type=str, help='Directory to write plots and tables to.', default='./')
    parser.add_argument('-p', '--prefix', type=str, default='nmf', help='Prefix to put before all otuput files.')

    args: argparse.Namespace = parser.parse_args()

    # Validate output directory
    if not os.path.isdir(args.output):
        print('Output directory does not exist.')
        quit()

    # Perform model selection
    df: pd.DataFrame = pd.read_csv(args.data, index_col=0)
    ms: NMFMultiSelect = NMFMultiSelect((args.k_min, args.k_max), methods=args.methods, iterations=args.iter,
                                        nmf_max_iter=args.nmf_iter, solver=args.solver, beta_loss=args.beta_loss)
    results: Dict[str, NMFResults] = ms.run(df)

    # Output these results
    obase: str = os.path.join(args.output, args.prefix)

    # Output the parameters used for selection
    with open(obase + '.paramters.json', 'w+') as op:
        # Convert file resource to path string, so can be dumped
        args.data = args.data.name
        json.dump(args.__dict__, op, indent=2)

    # Dump results
    with open(obase + '.results.res', 'wb+') as op:
        pickle.dump(results, op)

    # Write out plots for criteria
    fig: go.Figure = visualise.multiselect_plot(results)
    fig.write_html(obase + '.criteria_plot.html')

    # Write out table of criteria
    index: List[str] = []
    cols: List[str] = None
    vals: List[List[float]] = []
    for method, res in results.items():
        index.append(method)
        if cols is None:
            cols = [x.k for x in res.results]
        vals.append([x.metric for x in res.results])
    table: pd.DataFrame = pd.DataFrame(vals, index=index, columns=cols)
    table.to_csv(obase + '.criteria.csv')


if __name__ == '__main__':
    main()