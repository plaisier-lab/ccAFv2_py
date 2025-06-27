# ccAFv2: Cell cycle classifier for Python and scanpy
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

This repository is for the Python package for the cell cycle classifier ccAFv2. The input for the ccAFv2 classifier is single cell, nuclei, or spatial RNA-seq data.  The features of this classifier are that it classifies six cell cycle states (G1, Late G1, S, S/G2, G2/M, and M/Early G1) and a quiescent-like G0 state, and it incorporates a tunable parameter to filter out less certain classifications. This package is implemented in Python so that it can be used in [scanpy](https://scanpy.readthedocs.io/en/stable/) analysis workflows. We provide examples of how to install, run ccAFv2 on scanpy objects (sc/snRNA-seq), and plot and use results.

## Table of Contents

- [Install](#install)
    - [Requirements](#requirements)
        - [Dependencies](#dependencies)
        - [Docker image](#docker-image)
    - [Installing ccAFv2](#installing-ccafv2)
- [Classifying single cell or nuclei RNA-seq](#classifying-single-cell-or-nuclei-rna-seq)
    - [Input for classification](#input-for-classification)
    - [Test data](#test-data)
    - [Cell cycle classification](#cell-cycle-classification)
    - [Plotting cell cycle states](#plotting-cell-cycle-states)
        - [Plotting a UMAP with cell cycle states](#plotting-a-umap-with-cell-cycle-states)
- [Maintainers](#maintainers)
- [Contributing](#contributing)

## Install

### Requirements

It is strongly suggested that users utilize the docker images we provide on DockerHub as they contain all dependencies needed to run ccAFv2.

#### Dependencies
There are four dependencies that must be met for ccAF to classify cell cycle states:
1. [numpy](https://numpy.org/) - ([install](https://numpy.org/install/))
2. [scipy](https://www.scipy.org/index.html) - ([install](https://www.scipy.org/install.html))
3. [scanpy](https://scanpy.readthedocs.io/en/latest/) - ([install](https://scanpy.readthedocs.io/en/latest/installation.html))
4. [h5py](https://www.h5py.org/) - ([install](https://docs.h5py.org/en/latest/build.html))

##### Python dependency installation commands
> **NOTE!**  pip may need to be replaced with pip3 depending upon your setup.

#### Installation of ccAF classifier
The ccAFv2 classifier can be installed with the following command:

```sh
pip install ccAFv2
```

#### Docker image

We facilitate the use of ccAFv2 by providing a Docker Hub container [cplaisier/ccafv2](https://hub.docker.com/r/cplaisier/ccafv2) which has all the dependencies and libraries required to run the ccAFv2 classifier. To see how the Docker container is configured plaese refer to the [Dockerfile](https://github.com/plaisier-lab/docker_ccafv2/blob/master/Dockerfile). Please [install Docker](https://docs.docker.com/get-docker/) and then from the command line run:

```sh
docker pull cplaisier/ccafv2_py
```

Then run the Docker container using the following command (replace <path to scRNA-seq profiles directory> with the directory where you have the scRNA-seq data to be classified):

```sh
docker run -it -v '<path to scRNA-seq profiles directory>:/files' cplaisier/ccafv2_py
```

This will start the Docker container in interactive mode and will leave you at a command prompt. You will then want to change directory to where you have your scRNA-seq or trasncriptome profiling data.

## Classifying single cell or nuclei RNA-seq

### Input for classification

It is expected that the input for the ccAFv2 classifier will be a scanpy AnnData object that has been thorougly quality controlled. Is is preferred that the data in the object be SCTransformed; however, the standard approach for normalization only applies to the highly variable genes. This can exclude genes needed for the accurate classification of the cell cycle. During the running of the ccAFv2 classifier it will tell you how many genes overlap with the classifier marker genes.

### Test data

The human neural stem cells (hNSCs) from a human fetus 8 weeks post-conception (PCW8) [(Zeng et al., 2023)](https://pubmed-ncbi-nlm-nih-gov.ezproxy1.lib.asu.edu/37192616/) is available for use as a testing dataset:
- [PCW8 hNSCs h5ad file](https://zenodo.org/records/10968634/files/W8-1_normalized_ensembl.h5ad?download=1)

Download this file and place it into the directory in which you wish to run the ccAFv2 tutorial below. This data has been QC'd and normalized using SCTransform in Seurat following our best practices [here](https://github.com/plaisier-lab/ccafv2_R/blob/main/README.md#input-for-classification).

### Cell cycle classification

Classification is as easy as two lines that can be added to any Seurat workflow. First the library must be loaded and then the PredictCellCycle function is run:

```python
# Load packages
import pandas as pd
import scanpy as sc
import ccAFv2

# Load up test dataset
pwc8_scdata = sc.read_h5ad('W8-1_normalized_ensembl.h5ad')

# Run ccAFv2 to predict cell labels
labels, predictions = ccAFv2.predict_labels(pwc8_scdata , species='human', gene_id='ensembl')
```
When the classifier is running it should look something like this:

```python
Running ccAFv2:
    Preparing data for classification...
    Marker genes present in this dataset: 845
    Missing marker genes in this dataset: 16
  Predicting cell cycle state probabilities...
  Choosing cell cycle state...
Done.
```

It is important to look at how many marker genes were present in the dataset. We found that when less than 689 marker genes (or 80%) were found in the dataset that this led significantly less accurate predictions.

There are several options that can be passed to the PredictCellCycle function:

```python
ccAFv2.predict_labels(scanpy_obj,
                      threshold=0.5,
                      species='human',
                      gene_id='ensembl')
```
- **scanpy_obj**: a scanpy object must be supplied to classify, no default
- **threshold**: the value used to threshold the likelihoods, default is 0.5
- **species**: from which species did the samples originate, either 'human' or 'mouse', defaults to 'human'
- **gene_id**: what type of gene ID is used, either 'ensembl' or 'symbol', defaults to 'ensembl'

### Cell cycle classification results

The results of the cell cycle classification are stored in the 'labels' variable of the 'ccAFv2.predict_labels' output, and the likelihoods are stored in the 'predictions' variable.

```python
labels
```
The labels variable returns the following:

```python
AAACCTGTCAGTTCGA-1-1-0-0-0           G1
AAACCTGTCTCTTATG-1-1-0-0-0            S
AAACGGGAGGGTCTCC-1-1-0-0-0    Neural G0
AAACGGGGTTCGAATC-1-1-0-0-0           G1
AAACGGGGTTCGCGAC-1-1-0-0-0           G1
                                ...
TTTATGCGTTGTCGCG-3-1-0-0-0         G2/M
TTTGCGCGTCAAGCGA-3-1-0-0-0    Neural G0
TTTGGTTAGTCAAGGC-3-1-0-0-0      Late G1
TTTGGTTCAGTAAGAT-3-1-0-0-0    Neural G0
TTTGGTTGTGGACGAT-3-1-0-0-0            S
Name: Cell State, Length: 2562, dtype: category
Categories (8, object): ['G1', 'G2/M', 'Late G1', 'M/Early G1', 'Neural G0', 'S', 'S/G2',
                         'Unknown']
```

```python 
predictions
```
The predictions variable returns the following:

```python
array([[9.9654096e-01, 2.7795029e-05, 1.6239237e-03, ..., 1.3492864e-03,
        3.3741508e-04, 5.0000000e-01],
       [2.3844824e-03, 9.5887110e-05, 4.4037965e-03, ..., 7.6215446e-01,
        1.0546359e-01, 5.0000000e-01],
       [3.0904070e-05, 1.4728287e-06, 3.9923802e-06, ..., 1.6501120e-07,
        8.8521841e-07, 5.0000000e-01],
       ...,
       [2.7292622e-03, 3.5920220e-03, 9.8504561e-01, ..., 1.0077821e-04,
        4.4624828e-04, 5.0000000e-01],
       [1.4344385e-01, 1.6131684e-03, 2.9916968e-03, ..., 1.0987800e-04,
        3.2776382e-04, 5.0000000e-01],
       [7.0566205e-07, 1.2742312e-09, 1.1380459e-07, ..., 9.9999607e-01,
        2.8096736e-06, 5.0000000e-01]], shape=(2562, 8), dtype=float32)
```

In the code Below we demonstrate how the classifications can be added to the metadata. After adding the column to the .obs metadata, the classification for each cell would then found in the column 'ccAFv2', and is a categorical variable which helps with plotting.

```python
# Save into scanpy object
pwc6_scdata.obs['ccAFv2'] = labels
```

### Plotting cell cycle states

We provide plotting functions that colorize the cell cycle states in the way used in our manuscripts. We strongly suggest using these functions when plotting if possible.

#### Plotting a UMAP with cell cycle states

Plotting cells using ther first two dimensions from a dimensionality reduction method (e.g., PCA, tSNE, or UMAP) is a common way to represent single cell or nuclei RNA-seq data. Below we provide code to plot the cells colorized based on their called cell cycle state.

```python
# Run UMAP of U5 hNSCs
sc.pp.highly_variable_genes(pcw8_scdata, n_top_genes=2000)
sc.tl.pca(pcw8_scdata)
sc.pp.neighbors(pcw8_scdata)
sc.tl.umap(pcw8_scdata)

# Prepare a color mapping dictionary
cmap1 = {"": "#d9a428", "G1": "#f37f73", "Late G1": "#1fb1a9",  "S": "#8571b2", "S/G2": "#db7092", "G2/M": "#3db270" ,"M/Early G1": "#6d90ca",  "Unknown": "#d3d3d3"}

# Plot UMAP of U5 hNSCs
sc.pl.umap(pcw8_scdata, color=['ccAFv2'], palette=cmap1, save='ccAFv2_UMAP_PCW8.pdf')
```

In the figures folder you will find the PDF 'umapccAFv2_UMAP_PCW8.pdf'. Below is the UMAP for the hNSCs from a human fetus 8 weeks post-conception colorized using the cell cycle states. The expected flow of the cell cycle states can be seen in the UMAP.

![UMAP DimPlot colorized with ccAFv2 cell cycle states](https://github.com/plaisier-lab/ccAFv2_py/blob/main/figs/ccAFv2_UMAP_PCW8.png?raw=true)

## Maintainers

For issues or comments please contact:  [Chris Plaisier](mailto:plaisier@asu.edu)

And for other great packages from the Plaisier Lab please check here:  [@plaisier-lab](https://github.com/plaisier-lab).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/plaisier-lab/ccAFv2_R/issues/new) or submit PRs.

## Citation

1. **Citation for ccAFv2 (version 2)**:

[Classifying cell cycle states and a quiescent-like G0 state using single-cell transcriptomics](https://doi.org/10.1101/2024.04.16.589816) Samantha A. O’Connor, Leonor Garcia, Anoop P. Patel, Benjamin B. Bartelle, Jean-Philippe Hugnot, Patrick J. Paddison, Christopher L. Plaisier. bioRxiv [Preprint]. 2024 Apr 20:2024.04.16.589816. doi: [10.1101/2024.04.16.589816](https://doi.org/10.1101/2024.04.16.589816). PMID: [38659838](https://pubmed.ncbi.nlm.nih.gov/38659838/) 

2. **Citation for ccAF (version 1)**:

[Neural G0: a quiescent-like state found in neuroepithelial-derived cells and glioma.](https://www.embopress.org/doi/full/10.15252/msb.20209522) Samantha A. O'Connor, Heather M. Feldman, Chad M. Toledo, Sonali Arora, Pia Hoellerbauer, Philip Corrin, Lucas Carter, Megan Kufeld, Hamid Bolouri, Ryan Basom, Jeffrey Delrow, Jose L. McFaline-Figueroa, Cole Trapnell, Steven M. Pollard, Anoop Patel, Patrick J. Paddison, Christopher L. Plaisier.  Mol Syst Biol. 2021 Jun;17(6):e9522. doi: [10.15252/msb.20209522](https://doi.org/10.15252/msb.20209522). PMID: [34101353](https://pubmed.ncbi.nlm.nih.gov/34101353/)

