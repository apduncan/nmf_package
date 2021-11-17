"""Methods for visualising NMF Results. This includes reordering / clustering methods to visually group components."""

from fastcluster import linkage
from scipy.cluster.hierarchy import optimal_leaf_ordering, leaves_list
from scipy import stats
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_kernels
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Tuple, Set, Callable, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.basemap import Basemap

from mg_nmf.nmf import selection
import math

def affinity_order(df: pd.DataFrame, axis=0, link_method='average', optlorder: bool = True) -> Tuple[pd.DataFrame, List[int]]:
    """Return the dataframe ordered based on heirarchical clustering of the affinity matrix, with optimal leaf
    ordering. For w matrix, pass axis = 1"""
    data: pd.DataFrame = df
    # Calculate euclidean norm
    if axis == 1:
        data = data.T
    df_enorm: pd.DataFrame = np.linalg.norm(data, axis=0)
    df_bar = (data.T / df_enorm[:,None]).T
    # Similarity matrix
    S: pd.DataFrame = df_bar.T.dot(df_bar)
    # Convert to distances
    sdist: pd.DataFrame = np.ones(S.shape) - S
    # Make affinity matrix based on radial basis kernel
    sdist = sdist.fillna(1)
    A: pd.DataFrame = pairwise_kernels(sdist, metric='rbf')
    # Heirarchically cluster the affinity matrix (symmetric, doesn't matter axis)
    hclust: np.ndarray = linkage(A, link_method)
    opt_lorder: List[int]
    if optlorder:
        opt_lorder = leaves_list(optimal_leaf_ordering(hclust, A))
    else:
        opt_lorder = leaves_list(hclust)
    data = data.iloc[:, opt_lorder]
    if axis == 1:
        data = data.T
    return (data, opt_lorder)

def hierarchical_order(df: pd.DataFrame, axis=0, link_method='average', optlorder: bool = True) -> Tuple[pd.DataFrame, List[int]]:
    """Return the dataframe ordered based on heirarchical clustering, with optimal leaf ordering. For w matrix, pass
    axis = 1"""
    data: pd.DataFrame = df
    # Calculate euclidean norm
    if axis == 0:
        data = data.T
    # Heirarchically cluster the data
    hclust: np.ndarray = linkage(data, link_method)
    opt_lorder: List[int]
    if optlorder:
        opt_lorder = leaves_list(optimal_leaf_ordering(hclust, data.to_numpy()))
    else:
        # print('Leaf list, non-optimal')
        opt_lorder = leaves_list(hclust)
        # print('End')
    data = data.iloc[opt_lorder]
    if axis == 0:
        data = data.T
    return (data, opt_lorder)

def heatmap_plot(result: selection.NMFModelSelectionResults, w_dot_h: bool = False, file: str = None,
                 figsize: Tuple[float, float] = (12, 16), dpi: int = 90, log: bool = False, cbar: bool = False,
                 ordering: str = 'affinity', linkage_method='average', axes = [0, 1],
                 optorder_axes: Tuple[bool, bool] = (True, True), return_fig: bool = False, flip = False):
    """Plot W & H along the axes of a large heatmap representing source data or W * H. Ordering is afffinity
    or hierarchical."""
    # Adapted from https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html
    # Get our data
    sns.set_style('dark')
    h = result.h.copy()
    w = result.w.copy()
    data = result.data.copy()
    if flip:
        h, w, data = w.T, h.T, data.T
    h_idx = list(range(len(h.columns)))
    w_idx = list(range(len(w)))
    if ordering == 'affinity':
        if 0 in axes:
            h, h_idx = affinity_order(h, link_method=linkage_method, optlorder=optorder_axes[0])
        if 1 in axes:
            w, w_idx = affinity_order(w, axis=1, link_method=linkage_method, optlorder=optorder_axes[1])
    else:
        if 0 in axes:
            h, h_idx = hierarchical_order(h, link_method=linkage_method, optlorder=optorder_axes[0])
        if 1 in axes:
            w, w_idx = hierarchical_order(w, axis=1, link_method=linkage_method, optlorder=optorder_axes[1])
    # Reorder the input matrix using values of component 1
    if w_dot_h:
        ordered = w.dot(h)
    else:
        ordered = data.iloc[w_idx, h_idx]
    # Set up the axes with gridspec
    fig = plt.figure(figsize=figsize, dpi=dpi)
    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[:-1, 1:])
    y_hist = fig.add_subplot(grid[:-1, 0])
    x_hist = fig.add_subplot(grid[-1, 1:])

    if log:
        # Replace 0 with NaN to allow
        w = w.replace(0, np.nan)
        h = h.replace(0, np.nan)
        ordered = ordered.replace(0, np.nan)
        sns.heatmap(ordered, ax=main_ax, cbar=cbar, yticklabels=False, xticklabels=False,
                    norm=LogNorm(vmin=ordered.values.min(), vmax=ordered.values.max()),
                   mask=ordered.isnull())
        sns.heatmap(h, ax=x_hist, cbar=cbar,
                    norm=LogNorm(vmin=h.values.min(), vmax=h.values.max()),
                   mask=h.isnull())
        sns.heatmap(w, ax=y_hist, cbar=cbar, yticklabels=1 if len(w) < 100 else 'auto',
                   norm=LogNorm(vmin=w.values.min(), vmax=w.values.max()),
                   mask=w.isnull())
    else:
        sns.heatmap(ordered, ax=main_ax, cbar=cbar, yticklabels=False, xticklabels=False)
        sns.heatmap(h, ax=x_hist, cbar=cbar)
        sns.heatmap(w, ax=y_hist, cbar=cbar, yticklabels=1 if len(w) < 100 else 'auto')
    if file is not None:
        plt.savefig(file)
    else:
        if return_fig:
            return fig
        else:
            plt.show()

def cophenetic_plot(result: selection.NMFResults, file: str = None) -> None:
    """Plot the cophenetic correlation as a line plot."""
    data: pd.DataFrame = result.cophenetic_correlation()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.iloc[0], y=data.iloc[1]))
    if file is None:
        fig.show()
    else:
        fig.write_image(file)

def multiselect_plot(result: selection.NMFMultiSelect, file: str = None, figsize: Tuple[float, float] = (1000, 400),
                     cols: int = 4) -> go.Figure:
    # Create a multiplot where each plot is one of the selected methods, and each column is a method
    # Determine cols / rows
    rows = math.ceil(float(len(result)) / float(cols))

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(result.keys()))

    colors = px.colors.qualitative.Plotly

    row = 1
    col = 1
    color = 0

    for method, results in result.items():
        data = results.cophenetic_correlation()

        fig.append_trace(go.Scatter(
            y=data.iloc[1, :].astype('float64'), x=data.iloc[0, :], mode='lines', line=dict(color=colors[color]), showlegend=False,
            name=method
        ), row=row, col=col)
        col += 1
        color += 1
        if col > cols:
            col = 1
            row += 1

    fig.update_layout(height=figsize[1], width=figsize[0], hovermode='x')
    return fig

def model_to_map(h: pd.DataFrame, coords: pd.DataFrame, lat_lon: Tuple[str, str], scale: bool = True,
                 split: Callable[[str], str] = lambda x: 'all') -> List[go.Figure]:
    """

    :param h: The H matrix of an NMF model. Must have at least 3 components.
    :param coords: Coordinates for each station. Should have index which includes all columns of H.
    :param lat_lon: Column names in coords for latitutde and longitude respectively.
    :param scale: Scale range of each channel in RGB to amount of variance explained by each axis.
    :param split: Make multiple maps, based on the station label. Function maps from station name, to group identifier
                    which should be a string.
    :return figs: Plotly ScatterGeo plots with each point mapped to a colour, based on similarity.
    """

    # Step 1: Convert H model to RGB colours
    color, _ = pca_rgb(h, scale)
    # Step 2: Combine coordinates with colours
    only_coords: pd.DataFrame = coords[list(lat_lon)]
    # Ensure coordinates exist for all stations in the model
    missing: Set[str] = set(color.index) - set(only_coords.index)
    if len(missing) > 0:
        raise Exception(f'Coordinates missing for stations: {missing}')
    merge: pd.DataFrame = color.join(only_coords)
    # Attach a group identifier for each station
    merge['grp'] = merge.index.map(split)
    # Step 3: Make maps, one per group
    lat, lon = lat_lon
    figs = []
    print(merge)
    for grp in set(merge['grp']):
        # Reduce data
        grp_data = merge[merge['grp'] == grp].drop(columns=['grp'])
        fig: go.Figure = go.Figure(
            data=go.Scattergeo(
                lon=grp_data[lon],
                lat=grp_data[lat],
                text=grp_data.index,
                mode='markers',
                marker=dict(
                    size=[15] * len(grp_data),
                    color=rgb_to_string(grp_data)
                )
            )
        )
        # Establish some map settings
        fig.update_layout(
            geo=dict(
                showland=True, showocean=True, oceancolor='LightBlue', projection_type='sinusoidal'
            ),
            title_text = f'group = {grp}'
        )
        figs.append(fig)
    return figs

def model_to_piemap(h: pd.DataFrame, coords: pd.DataFrame, lat_lon: Tuple[str, str],
                    latlon_margin: Tuple[float, float] = (5., 5.), pie_size: float = 350,
                    fig_width: float = 10.0, alpha: float = 0.9, colors: Optional[List[str]] = None,
                    lat_bounds: Optional[Tuple[float, float]] = None,
                    lon_bounds: Optional[Tuple[float, float]] = None) -> Tuple[plt.Figure, plt.Axes]:
    """

    :param h: The H matrix of an NMF model. Must have at least 3 components.
    :param coords: Coordinates for each station. Should have index which includes all columns of H.
    :param lat_lon: Column names in coords for latitutde and longitude respectively.
    :param latlon_margin: Margins to add to bounds of plot, in degrees. Tuple in order latitude, longitude.
    :param pie_size: Size of pie chart.
    :param fig_width: Width of figure, in inches. Height adjusted automatically.
    :param alpha: Transparency of pie chart (0-1).
    :param colors: Colors to use for pie segements. Default to plotly colours.
    :param lat_bounds: Upper and lower bound of latitude to plot. If not provided, calculated from input data.
    :param lon_bounds: Upper and lower bound of longitude to plot. If not provided, calculated from input data.
    :return fig, ax: Plotly ScatterGeo plots with each point mapped to a colour, based on similarity.
    """

    # Step 1: Combine coordinates with component values
    only_coords: pd.DataFrame = coords[list(lat_lon)]
    # Ensure coordinates exist for all stations in the model
    missing: Set[str] = set(h.index) - set(only_coords.index)
    if len(missing) > 0:
        raise Exception(f'Coordinates missing for stations: {missing}')
    merge: pd.DataFrame = h.join(only_coords)
    lat, lon = lat_lon
    # Set colors if not yet set
    if colors is None:
        colors = px.colors.qualitative.Plotly

    # Step 2: Set up base map figure
    # Sort out latitude and longitude bounds for this figure
    lat_margin, lon_margin = latlon_margin
    # Determine the min and max for both longitude and latitude, to form the edges of our map
    min_lat, max_lat = merge[lat].min(), merge[lat].max()
    min_lon, max_lon = merge[lon].min(), merge[lon].max()
    # If lat/lon bounds were provided, overwrite
    if lat_bounds is not None:
        min_lat, max_lat = lat_bounds
    if lon_bounds is not None:
        min_lon, max_lon = lon_bounds
    # Calculate corners with margins
    lllat = max(min_lat - lat_margin, -90)
    lllon = max(min_lon - lon_margin, -180)
    urlat = min(max_lat + lat_margin, 90)
    urlon = min(max_lon + lon_margin, 180)
    # Determine suitable plot height (not perfect due to variation in projections)
    height = ((urlat - lllat) / (urlon - lllon)) * fig_width
    # Create figure and axes
    fig = plt.figure(figsize=(fig_width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    # Create map
    m = Basemap(resolution='l', projection='merc', llcrnrlat=lllat, urcrnrlat=urlat, llcrnrlon=lllon,
                urcrnrlon=urlon, lat_ts=0)
    m.drawlsmask(land_color='#a69a85', ocean_color='#dcf2f2', lakes=True)

    # Step 3: Convert each station into a pie chart, and plot
    # Plot a pie for each station
    first = True
    for r in merge.iterrows():
        name, vals = r
        # m converts lat/lon to coordinate values on projection
        rlat, rlon = m(vals[lon], vals[lat])
        # Convert componennt weights to ratios
        weights = np.array([vals[x] for x in vals.index if x[0] == 'c'])
        ratios = weights / weights.sum()

        # Adapting https://www.geophysique.be/2010/11/15/matplotlib-basemap-tutorial-05-adding-some-pie-charts/
        # n = len(ratios)
        xy = []
        start = 0.0
        for ratio in ratios:
            x = [0] + np.cos(np.linspace(2*math.pi*start,2*math.pi*(start+ratio), 30)).tolist()
            y = [0] + np.sin(np.linspace(2*math.pi*start,2*math.pi*(start+ratio), 30)).tolist()
            xy1 = zip(x, y)
            xy.append(xy1)
            start += ratio

        for i, xyi in enumerate(xy):
            xy_arr = np.array(list(xyi))
            s = np.abs(xy_arr).max()
            ax.scatter([rlat], [rlon], marker=xy_arr, s=pie_size * s, facecolor=colors[i],
                       linewidth=0.0, alpha=alpha)
            if first:
                ax.scatter([], [], label=f'c{i+1}', facecolor=colors[i])
        first = False
    plt.legend()
    return fig, ax

def pca_rgb(df: pd.DataFrame, scale: bool = True):
    """Map rows of a DataFrame to an RGB color space.
    As explained in Richter, D. et al. Genomic evidence for global ocean 
    plankton biogeography shaped by large-scale current systems. (2020).
    df  : DataFrame where each row is an observation, each column a feature.
          Must have 3+ columns.
    scale : Scale the amount of each color channel used based on variance explained by each component."""

    # Validation
    if df is None:
        raise Exception('Must provide a DataFrame')
    if len(df.columns) < 3:
        raise Exception('Must provide a DataFrame with at least 3 columns')

    # Work with a copy of the data
    data: pd.DataFrame = df.copy()

    # Step 1: Power-transformed observations using the Box-Cox transformation
    # to have Gaussian-like distributions to mitigate the effect of outliers and scaled to have zero mean and
    # unit variance
    if (data == 0).any().any():
        data = data + 1
    h_bc = data.apply(lambda x: stats.boxcox(x)[0])

    # Step 2: Carry out PCA up to 3 components
    pca = PCA(n_components=3)
    t = pca.fit_transform(h_bc)
    # Sum the explained variance
    variance = pca.explained_variance_ratio_
    
    # Step 3: Rescale each component to have mean 0 and unit variance
    scaler = StandardScaler()
    t = scaler.fit_transform(t)
    
    # Step 4: Decorrelate using Mahalanobis transform
    t = mahalanobis_transform(t, t)
    
    # Step 5: Map each component to 0-255 to occupy a colour channel
    if not scale:
        # No scaling of space used in each RGB component desired, use full 0-255 for each
        t = np.apply_along_axis(color_map, 0, t)
    else:
        # Scaling desired, use full range for first axis, scale space used for second
        # If first axis explains 50% variation, and second explains 25%, use half the available space
        t[:, 0] = color_map(t[:, 0], 1)
        t[:, 1] = color_map(t[:, 1], variance[1] / variance[0])
        t[:, 2] = color_map(t[:, 2], variance[2] / variance[0])
    
    # Step 6: Convert to integers
    t = t.astype(int)
    
    # Step 7: Convert to DataFrame with index attached
    color_df = pd.DataFrame(t, index=df.index, columns=['r', 'g', 'b'])
    
    return color_df, pca

def mahalanobis_transform(x: pd.DataFrame, data: pd.DataFrame = None, cov: pd.DataFrame = None):
    """Compute the Mahalanobis transform between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to 
            be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    # To get Mahalanobis distance, return diagonal of below term
    # mahal = np.dot(left_term, x_minus_mu.T)
    return left_term

def color_map(arr, use_range=1):
    """Map each entry in the array to a value 0-255.
    arr  : 1d array of floats
    use_range : float 0-1, specifying how much of the color range to use (to scale amount used depending on variance
        explained by components.
    """
    ceil = max(arr)
    floor = min(arr)
    diff = ceil - floor
    # Richter paper usese 128 * (1 + (x / ceil)), however this leave some area at the top or tail
    # of the space unused, so changed implementation
    # Determine how much of the range to use
    crange = use_range * 255
    cstart = math.floor((255 - crange)/2)
    # Try alternative to use full space
    return [cstart + (crange * ((x - floor)/diff)) for x in arr]

def rgb_to_string(df):
    """Convert results of pca_rgb to an array of strings which can be used as colors in plotly"""
    colors = df.apply(lambda x: [int(y) for y in x], axis=1)
    color_str = [f"rgb({x[0]},{x[1]},{x[2]})" for x in colors]
    return color_str

if __name__ == '__main__':
    # Some tests
    rnd_data: pd.DataFrame = selection.shuffle_frame(selection.synthdata.sparse_overlap_even(
        rank=3, noise=(0, 2), size=(500, 20), p_ubiq=0.0, m_overlap=0.0, n_overlap=0.0)
    )
    # Apply standard NMF
    selector = selection.NMFConsensusSelection(rnd_data, k_min=2, k_max=4, solver='mu', beta_loss='kullback-leibler',
                                            iterations=10, nmf_max_iter=10000)
    results: selection.NMFResults = selector.run()

    print('TEST CORR')
    cophenetic_plot(results[0])
    model_3: selection.NMFModelSelectionResults = [x for x in results[0].results if x.k == 3][0]

    print('TEST ORDER')
    w_order: pd.DataFrame = affinity_order(model_3.w, axis = 1)
    print(w_order)

    print('TEST HMAP')
    heatmap_plot(model_3, flip=False, axes=[])

    # print('TARA DATA')
    # t_data = pd.read_csv('data/last_tara_run_data.csv', index_col=0)
    # import normalise as norm
    # selector = selection.NMFModelSelection(t_data, k_min=2, k_max=7, solver='mu', beta_loss='kullback-leibler',
    #                                        iterations=100, nmf_max_iter=10000, filter=norm.variance_filter,
    #                                        filter_threshold=0.75, normalise=norm.map_maximum)
    # tara: selection.NMFResults = selection.NMFResults.load_results('data/last_tara_run.results')
    # tara: selection.NMFResults = selector.run()
    # tara.write_results('data/last_tara_run.results')
    # res = [x for x in tara.results if x.k == 2][0]
    # heatmap_plot(res, log=True)
    # Attempt some feature selection
    # import signature as sig
    # dat = res.data.loc[res.w.index, res.h.columns]
    # correlations = sig.Correlation(res, dat).select()
    # correlations['max'] = correlations.max(axis=1)
    # correlations = correlations.sort_values(by=['max'], ascending=False)
    # # Let's see what it looks like to take the top 500
    # idx_top = correlations.iloc[:100].index
    # trimmed_w = res.w.loc[idx_top]
    # print(trimmed_w)
    # new_res = selection.NMFModelSelectionResults(res.k, res.cophenetic_corr, res.connectivity, trimmed_w, res.h,
    #                                              res.model, res.data)
    # heatmap_plot(new_res, log=True)
    # from selection import NMFResults
    #
    # with open('data/last_tara_run.results', 'rb') as f:
    #     exp = pickle.load(f)
    # exp = selection.NMFResults.load_results('/home/hal/Dropbox/PHD/FunctionalAbundance/nmf/data/metatranscriptome_k6.model')
    # exp = pd.read_csv('~/Dropbox/PHD/h_no_map.csv', index_col=0)
    #
    # # Load station metadata for lat / lon
    # metadata = pd.read_csv('~/Dropbox/PHD/FunctionalAbundance/nmf/data/tara/ERP001736.csv', index_col='Run')
    # # Filter metadata to only runs in our index
    # runs = exp.index.map(lambda x: x.split(' ')[0].strip())
    # metadata = metadata.loc[runs][['Latitude Start', 'Longitude Start']]
    #
    # # heatmap_plot(next(x for x in exp.results if x.k == 4), axes=[0,1], optorder_axes=(True, False))
    # # Fix index of model (remove station names)
    # modh = exp
    # modh.index = modh.index.map(lambda x: x.split(' ')[0].strip())
    # #  Testing RGB model with 0 values for weight of components
    # # model_to_map(exp, metadata, lat_lon=('Latitude Start', 'Longitude Start'),
    # #              split=lambda x: 'all', scale=True)[0].show()
    # model_to_piemap(modh, metadata, lat_lon=('Latitude Start', 'Longitude Start'), latlon_margin=(2, 9), fig_width=6,
    #                 colors=px.colors.qualitative.Light24)
    # plt.show()
    # plt.close('all')
    # heatmap_plot(exp, ordering='heirarchical', optorder_axes=[False, False])
    # plt.show()
    # import pickle
    # with open('../tests/data/test_multires.res', 'rb') as f:
    #     res = pickle.load(f)
    # multiselect_plot(res).show()
