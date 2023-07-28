# Metagenome NMF
This package provides tools for applying Non-negative Matrix Factorisation (NMF) to metagenomic data. 
Briefly, NMF methods factorise matrix $X$ with dimensions $m \times n$ to $W$ with dimensions $m \times k$ 
and $H$ with dimensions $k \times n$, such that $WH \approx X$, with the constraint that no entry in $W$ or $H$ can be 
negative.
The value $k$ is the rank of the decomposition; for application to metagenomics identifying a low rank decomposition 
can provide dimensionality reduction
The underlying implementation of NMF is provided by ```scikit-learn```. 
This pacakge provides additional tools in two main areas: selecting appropriate an appropriate rank for decompositions,
 and interpreting features in the resulting decomposition.
 Visualisation methods related to these are also included.
  
Here high level overviews of the different tools are given, before more detailled usage. 
Additionally, a Jupyter notebook with a worked analysis is provided.

 ## Rank Selection
Seven critera for evaluating the suitability for values of $k$ for decompositions of $X$.
A majority of these rank selection methods take some measurement from multiple decompositions of rank $k$ with random 
initialisations of $W$ and $H$, and as such are the most computationally expensive step in applying NMF.

For data expected to have a non-discrete underlying structure (each sample may be a mixture of underlying modules), 
our testing found the *concordance index* to perform best.
In cases where the structure is expected to be more discrete (each sample can be assigned to a single module), the 
connectivity based *dispersion* and *cophenetic correlation* methods perform well.

In practice, if computationally feasible given resources, we suggest running all methods other than *mse* and *mad* 
(due to long execution time) and identifying values of $k$ for which the *concordance index* and other methods show 
peaks.
Visualisation of candidate ranks, as well as those within +/-1, can help identify a suitable rank for the decomposition.

The rank selection methods provided are:
- *Concordance Index*; Jiang, X., Weitz, J. S. & Dushoff, J. A non-negative matrix factorization framework
    for identifying modular patterns in metagenomic profile data. J. Math. Biol. 64, 697–711 (2012).
- *Cophenetic Correlation*; Brunet, J.-P., Tamayo, P., Golub, T. R. & Mesirov, J. P. Metagenes and molecular pattern 
    discovery using matrix factorization. Proc. Natl. Acad. Sci. U.S.A. 101, 4164–4169 (2004).
- *Dispersion*; Kim, H. & Park, H. Sparse non-negative matrix factorizations via alternating 
    non-negativity-constrained least squares for microarray data analysis. Bioinformatics 23, 1495–1502 (2007).
- *Split-Half*; Muzzarelli L, Weis S, Eickhoff SB, Patil KR. Rank Selection in Non-negative Matrix
    Factorization: systematic comparison and a new MAD metric. 2019 International Joint Conference on Neural Networks
    (IJCNN). 2019. pp 1–8.
- *Permutation*; Muzzarelli et a. 2019
- *Imputation Mean Squared Error*; Muzzarelli et al. 2019
- *Imputation Median Absolute Deviation*; Muzzarelli et al. 2019

## Feature Interpretation
Given a decomposition $WH \approx X$, matrix $H$ gives the weight of the $n$ features in each of the $k$ modules.
In typical metagenomic data, $n$ will be very large, and interpreting $H$ an important step in generating biological 
understanding of the identified modules.
Two approaches to this problem are provided: 
providing measures to evaluate the importance of each feature to each module, 
and an implementation of Gene Set Enrichment Analysis to identify genesets which are enriched in a module, using 
```GSEApy```.

In data with features expected to be shared among multiple modules, our testing showed LOOCD (Leave-One-Out Correlation 
Decrease) to best identify features which are shared among modules in simulated data.

## Usage
### Rank Selection
The most straightforward way to use rank selection is through the ```NMFMultiSelect``` class. This allows you to run 
multiple (or just a single) rank selection criteria with the same settings, and returns their results.

```python
import pandas as pd
from mg_nmf.nmf.selection import NMFMultiSelect

# Load metagenome data
data = pd.read_csv('mg_data.csv')

# Create an object to perform multiple rank selection methods
# Here, methods takes a list of the methods to run, ranks is a collection of the ranks for which to run each method
# Other parameters are covered in the docstring
rank_selection = NMFMultiSelect(
    methods=['coph', 'disp', 'conc', 'perm', 'split'], ranks=range(2, 21), 
    iterations=100, beta_loss='kullback-leibler'
)

# Run rank selection methods for the loaded data
rank_selection_results = rank_selection.run(data)

# This returns a dictionary with keys being the method used, and value the results from that method as an 
# NMFModelSelection object. These results can be visualised as line plots using a provided method
import mg_nmf.nmf.visualise as visualise
visualise.multiselect_plot(rank_selection_results)
```

### Feature Interpretation
#### Gene Set Enrichment Analysis
Given matrices $W$ and $H$, row $i$ in $H$ is the weight of feature in the $i-th$ module, 
and column $i$ in $W$ the weight of the module in each sample.
From this, it can be useful to determine if sets of features with known biological relationships are enriched in the 
module.
We provide a wrapper for the ```GSEApy``` Prerank method, using as input for each module correlation between column
$W_i$ and each feature in $X$, with the intuition that the closer a features profile in the input data is to the module 
weights the more representative that feature is of the module.

Methods to fetch some standard genesets are provided, as well as to provide custom sets. The standard genesets include 
KEGG Orthologs grouped into KEGG Pathways, InterPro Accessions grouped into Gene Ontology Terms, and Pfams grouped into 
Gene Ontology Terms.

```python
# First obtain or define a geneset
from mg_nmf.nmf import enrichment

# Retrieve mapping of KEGG Orthologs to Pathways, fetches from the KEGG API online and stores locally for future use
# limit_pathways is an argument which can be to filter out some sets, here we remove any drug development related 
# pathways as these are likely not relevant to metagenomics
gs = enrichment.GeneSets()
geneset = gs.geneset_ko2pathway(limit_pathways=lambda x: 'map07' not in x)
# When visualising results, some metadata describing features and pathways is helpful, and can also be loaded by 
# provided methods
geneset_names, geneset_md = gs.geneid_names_ko(), gs.geneset_metadata_kegg_pathways()

# To perform enrichment analysis, we create an NMFGeneSetEnrichment object. The label parameter determines the column 
# name for features in outputs. The names and metadata can be omitted, but make reading results simpler.
enr = enrichment.NMFGeneSetEnrichment(model=model, data=model.data, gene_sets=geneset, 
    gene_set_metadata=geneset_md, gene_names=geneset_names, label='ko')
# The analysis is run the first time the results methods is called. Subsequent calls to results will not rerun the 
# analysis. Significance is the treshold for FDR which is used for multiple test correction. 
enr_res = enr.results(significance=0.05)

# This returns a table with the enriched pathways, and their normalised enrichment score
# This can be visualised as a heatmap using the provided method. group parameter takes a column in the results table to 
# divide the plot into multiple groups - primarily useful for dividing up GO namespaces
enr.plot_enrichment(enr_res)

# It may be of interest to examine two associated plots for each set of genes: the GSEA diagram, and scatter plots 
# showing the correlation underlyign that GSEA diagram
# In this example, we look at plots for enrichment of the Riboflavin Metabolism pathway (map00740) in module m1
enr.plot_gsea(component='m1', gene_set_id='map00740', ofname='output_file.png')
# This returns a plotly plot, which can be shown or written to disk
fig = enr.plot_geneset_correlation(component='m1', gene_set_id='map00740', cols='4')
fig.show()
fig.write_image('output_file.pdf')
```

#### Feature Importance
For each feature, we can seek to identify how important it is to each module. 
In the GSEA method above, we quantify that using correlation between columns of $H$ and $X$.
However, this can underestimate the relevance of genes which are shared between modules, and so we provide alternate 
methods of evaluating the importance of each feature to each module.
*Leave-One-Out Correlation Decrease* performs best for features which are shared, in addition to identifying features 
which are unique, and so we suggest using this measure.
All methods are implemented as classes inheriting from ```nmf.signature.FeatureSelection```.
A description of the methods provided is given below

- Leave-One-Out Correlation Decrease (```LeaveOneOut```): Based on the idea that the correspondence 
between feature $f$ in input data and the model should become measurably worse if we omit a module $i$ which is important in 
describing a feature. If $W_i$ is $W$ with the column for $i$ removed, and $H_i$ with the row for $i$ removed, the 
product $W_iH_i$ still has the same dimensions as $X$. 
This method proposes that if feature $f$ is important to module $i$, then the correlation between $f$ in the input data 
$X$ in the complete model $WH$ should be greater than the correlation between $f$ in $X$ and $W_iH_i$.  
- Correlation (```Correlation```): The Pearson correlation between the columns for feature $f$ in $X$ and $W$. Where 
the profile of a feature in the data is closely correlated to that profile of module, we take that as evidence that it 
is representative of the module. However where a feature is described by two modules, the correlation to each one will 
be lower.
- Specificity (```Specificity```): The extent to which a feature has weight in only one module. Suitable for finding 
unique features, obviously not suitable for shared features.
- Permutation (```Permutation```): If a feature is genuinely important to a module, it should have a higher weight in 
our model than on models we learn from data which is randomly permuted. This method learns data from permuted data a 
number of times, then fits a normal distribution to each feature weight, and takes the probability that the observed 
weight originates from that distribution. This method is computationally expensive, and we found little benefit 
compared to LOOCD.

```python
from mg_nmf.nmf import signature
# All classes share the same signature, here we will use LOOCD but others are used in the same way
loocd = signature.LeaveOneOut(model=model, model=model.data)
loocd_res, _ = loocd.select()
# This will return a table with the importance measure for each feature in each module, in this case how much the 
# correlation decreases by when each module is left out. A lower value indicates greater importance for LOOCD.
# The method returns a tuple, as some classes also return a table of significances, but LOOCD does not currently.
```

#### Feature Assignment
From a measure of feature importance, we want to be able to identify from those measures which features can be 
considered to belong to a module - essentially classifying whether a feature is part of a module or not.
We provide three methods of doing this, and recommend using a combination of ```LeaveOneOut``` and the simple 
```ThresholdAssignment``` with the default cutoff (-0.05).
Each approach is implemented as a class which inherits from the ```FeatureAssignment``` class, and can be used in the 
same way.
* Threshold (```ThresholdAssignment```): Assign any feature with a value above (or below) a selected threshold value to 
the module. The paramater ```cut_low``` is a boolean which determines whether to look for values above (```false```) 
or below (```true```) the threshold value.
* KDE Based (```KDEAssignment```): Based on the observation correlation and permutation measures of feature importance 
form a bimodal distribution, with features in the module tending to be on one side of this.
Identifying a minima in the centre can be used as a cutting point for classification.
This method does not function with LOOCD as it does not have this bimodal distribution.
* Greedy Assignment (```GreedyCorrelationAssignment```): A greedy algorithm, which for each feature $f$ iteratively 
assigns this to modules $m$ so long as the model restricted to only those modules $WH_m$ has improved correlation to 
$X$ above a certain threshold. This method performed poorly compared to those above in our tests.

```python
from mg_nmf.nmf import signature
# Using the LOOCD results we obtained earlier, perform assignment using the default threshold
assign = signature.ThresholdAssignment(measure=loocd_res, cut_low=True, threshold=-0.05)
assign_res = assign.assign()
# This returns a boolean table, indicate for each feature which modules it has been assigned to
```
