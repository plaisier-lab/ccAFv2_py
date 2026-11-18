##########################################################
## OncoMerge:  ccAFv2.py                                ##
##  ______     ______     __  __                        ##
## /\  __ \   /\  ___\   /\ \/\ \                       ##
## \ \  __ \  \ \___  \  \ \ \_\ \                      ##
##  \ \_\ \_\  \/\_____\  \ \_____\                     ##
##   \/_/\/_/   \/_____/   \/_____/                     ##
## @Developed by: Plaisier Lab                          ##
##   (https://plaisierlab.engineering.asu.edu/)         ##
##   Arizona State University                           ##
##   242 ISTB1, 550 E Orange St                         ##
##   Tempe, AZ  85281                                   ##
## @Author:  Chris Plaisier, Samantha O'Connor,         ##
#            Thurston Herricks                          ##
## @License:  GNU GPLv3                                 ##
##                                                      ##
## If this program is used in your analysis please      ##
## mention who built it. Thanks. :-)                    ##
##########################################################

###################
## Load pacakges ##
###################

import numpy  as np
import pandas as pd
import scanpy as sc
from importlib.resources import path
import warnings

from ._ccAFv2 import run_model
from anndata  import AnnData
from importlib.resources import files
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib.backends.backend_pdf import PdfPages

from scipy import sparse

from scipy.stats import vonmises
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
sc.settings.verbosity = 0


############################
## Load model information ##
############################

with path('ccAFv2', 'ccAFv2_genes.csv') as inPath:
    _genes_all = pd.read_csv(inPath, index_col=0, header=0)
with path('ccAFv2', 'ccAFv2_classes.txt') as inPath:
    _classes = list(pd.read_csv(inPath, header=None)[0])


#####################################
## Load circular manifold loadings ##
#####################################

with path('ccAFv2', 'seurat_ref_U5_hNSC_SCT_7_29_2026.csv') as inPath:
    _u5_ref = pd.read_csv(inPath, index_col=0, header=0)
with path('ccAFv2', 'rownames_u5_ref.csv') as inPath:
    _rownames_u5_ref = pd.read_csv(inPath, header=0)


###############
## Functions ##
###############


# Scale data same way in R and Python
def _scale(df1):
    """
    scale takes in a pandas dataframe and applies scales the values into Z-scores across rows.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame of scRNA-seq data to be scaled.

    Returns
    -------
    pd.DataFrame
        DataFrame of scRNA-seq data that has been scaled.
    """
    return (df1.subtract(df1.mean(axis=1),axis=0)).div(df1.std(axis=1),axis=0)


# Integrate probability from von Mises distrubiton into equally spaced angular bins
def angles_to_vonmises_bins(angles, n_bins=10, kappa=4.0):
    """
    Convert angles (radians) into n-bin probability vectors using
    a von Mises distribution centered at each angle.

    Parameters
    ----------
    angles : array-like
        Angles in radians.
    n_bins : int
        Number of circular bins.
    kappa : float
        Concentration parameter of the von Mises distribution.
        Larger values produce narrower distributions.

    Returns
    -------
    pd.DataFrame
        Rows correspond to input angles.
        Columns are bin probabilities summing to 1.
    """
    angles = np.asarray(angles) % (2 * np.pi)

    # Bin edges
    edges = np.linspace(0, 2 * np.pi, n_bins + 1)
    prob_vectors = []
    for mu in angles:
        probs = []
        for left, right in zip(edges[:-1], edges[1:]):
            # Probability mass in this bin
            p = vonmises.cdf(right, kappa, loc=mu) - \
                vonmises.cdf(left, kappa, loc=mu)
            # Handle wrap-around numerically
            if p < 0:
                p += 1.0
            probs.append(p)
        probs = np.array(probs)
        probs /= probs.sum()  # normalize
        prob_vectors.append(probs)
    return pd.DataFrame(prob_vectors, columns=[f"bin_{i}" for i in range(n_bins)])


# Project transcriptomic data into reference embedding
def project_cycle_from_matrix(expr, ref_m, center_pc1=0, center_pc2=0):
    """
    Project expression data onto reference component loadings.

    Parameters
    ----------
    expr : pandas.DataFrame
        Expression matrix with genes as rows and cells as columns.
    ref_m : pandas.DataFrame
        Reference loadings with genes as rows and components as columns.
    center_pc1, center_pc2 : float
        Coordinates to subtract from the first two projected components.

    Returns
    -------
    dict
        'projection': cells × components DataFrame.
        'position': Series containing each cell's angle in radians.
    """
    if not expr.index.is_unique or not ref_m.index.is_unique:
        raise ValueError("Gene names must be unique in expr and ref_m.")

    if ref_m.shape[1] < 2:
        raise ValueError("The reference must contain at least two components.")

    # Preserve the order of overlapping genes in expr.
    genes = expr.index[expr.index.isin(ref_m.index)]

    if len(genes) == 0:
        raise ValueError(
            "No overlapping genes between the expression matrix "
            "and the reference."
        )

    print(
        f"    The number of projection genes found in the new data "
        f"is {len(genes)}."
    )

    reference = ref_m.loc[genes].copy()

    # Transpose to cells × genes and center each gene across cells.
    expression = expr.loc[genes].to_numpy(dtype=float).T
    expression = expression - expression.mean(axis=0, keepdims=True)

    projection = expression @ reference.to_numpy(dtype=float)

    projection[:, 0] -= center_pc1
    projection[:, 1] -= center_pc2

    # Counterclockwise angle from the positive PC1 axis.
    theta = np.mod(
        np.arctan2(projection[:, 1], projection[:, 0]),
        2 * np.pi,
    )

    projection = pd.DataFrame(
        projection,
        index=expr.columns,
        columns=reference.columns,
    )
    projection.attrs["rotation"] = reference

    position = pd.Series(
        theta,
        index=expr.columns,
        name="position",
    )

    return {
        "projection": projection,
        "position": position,
    }


# Project transcriptomic data into reference embedding
def project_cycle_from_anndata( adata, ref_m, reduction_name="cc_Projection", position_col="thetaPos", center_pc1=0,center_pc2=0):
    """
    Project AnnData expression data onto a reference embedding.

    Uses adata.X when layer=None, otherwise adata.layers[layer].
    Modifies and returns adata.

    Results are stored in:
        adata.obsm[reduction_name]
        adata.obs[position_col]
        adata.uns[reduction_name]
    """
    if not adata.var_names.is_unique or not ref_m.index.is_unique:
        raise ValueError("Gene names must be unique in adata and ref_m.")

    genes = adata.var_names[adata.var_names.isin(ref_m.index)]

    if len(genes) == 0:
        raise ValueError(
            "No overlapping genes between AnnData and the reference."
        )

    # Subset before converting sparse data to a dense matrix.
    subset = adata[:, genes]
    expression = subset.X

    if sparse.issparse(expression):
        expression = expression.toarray()

    expr = pd.DataFrame(
        np.asarray(expression).T,
        index=genes,
        columns=adata.obs_names,
    )

    result = project_cycle_from_matrix(
        expr=expr,
        ref_m=ref_m,
        center_pc1=center_pc1,
        center_pc2=center_pc2,
    )

    adata.obsm[reduction_name] = result["projection"].to_numpy()

    # Preserve the exact rotation/reflection formula from your R code.
    rotation = 0.225 * np.pi
    adata.obs[position_col] = (
        2 * np.pi
        - np.mod(result["position"] - rotation, 2 * np.pi)
    )

    adata.uns[reduction_name] = {
        "rotation": result["projection"].attrs["rotation"],
        "components": ref_m.columns.astype(str).tolist(),
    }

    return adata


# Use ccAFv2 classifier to calculate class probabilities
def _predict_classes(prep_data, classes = _classes):
    """
    _predict_classes takes in a pandas dataframe and the trained ccAFv2 model.

    Parameters
    ----------
    prep_data : pd.DataFrame
        DataFrame of scRNA-seq data to be classified.
        rows need to be cells and columns need to be the 861 classifer genes
        The gene order is given in the file ccAFv2_gens.csv

    Returns
    -------
    out_preds: flaot32 numpy array of probabilities
        rows are the cell samples
        columns are classifiers given by the order in ccAFv2_classes.txt

    """
    if prep_data.shape[1] != 871:
        raise RuntimeError('Input data frame not the right size for predictions.  Please check the integrity of the gene list')

    if not prep_data.shape[0]:
        raise RuntimeError('There are no cells in the input data')

    print('  Predicting cell cycle state probabilities...')
    # Create output array to receive data.
    oup_preds = np.zeros((prep_data.shape[0], 7), dtype = 'float32')

    # Change the pandas dataframe to a numpy array of float32
    inp_data = prep_data.to_numpy()
    inp_data = np.ascontiguousarray(inp_data, dtype = np.float64)

    # Loop through array to make predictions
    for ind in range(inp_data.shape[0]):
        oup_preds[ind,:] = run_model(inp_data[ind,:].reshape(1,871))

    prob_df = pd.DataFrame(oup_preds)
    prob_df.index = prep_data.index
    prob_df.columns = classes

    return prob_df


# Prepare test data for predicting
def _prep_predict_data(data_scanpy, genes):
    """
    prep_predict_data takes in a pandas dataframe and the trained ccAFv2 model.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of scRNA-seq data to be classified.
    model : keras.models.sequential
        Trained ccAFv2 sequential keras model.

    Returns
    -------
    pd.Series
        Series of labels for each single cell.

    """
    print('    Preparing data for classification...')

    # Uniquify the gene names
    data_scanpy.var_names_make_unique()

    # Remove all genes with zero counts
    sc.pp.filter_genes(data_scanpy, min_cells=1)

    # Restrict data to classifier genes
    in_both = list(set(genes).intersection(data_scanpy.var_names))
    if(len(in_both)>0):
        print('    Marker genes present in this dataset: '+str(len(in_both)))
        print('    Missing marker genes in this dataset: '+str(len(set(genes))-len(in_both)))

        # Scale the data subset with all available genes
        data_subset = data_scanpy[:,in_both]
        if sparse.isspmatrix(data_scanpy.X):
            data_subset = pd.DataFrame(data_subset.X.todense(), index = data_subset.obs_names, columns = data_subset.var_names)
        else:
            data_subset = pd.DataFrame(data_subset.X, index = data_subset.obs_names, columns = data_subset.var_names)
        data_df = pd.DataFrame(_scale(data_subset), index = data_subset.index, columns = data_subset.columns)

        # Add minimum values for missing genes
        missing = list(set(genes).difference(data_df.columns))
        if len(missing)>0:
            data_df_wMissing = pd.concat([data_df, pd.DataFrame(data_df.values.min(), index=data_df.index, columns = missing)], axis=1)
            return data_df_wMissing[list(genes)]
        else:
            return data_df
    else:
        raise RuntimeError('Check species and gene_id, because there is no overlap between input genes and classifier genes!')

# Predict labels
def predict_labels(new_data, species='human', gene_id='ensembl', threshold=0, include_g0=False, genes_all=_genes_all, classes=_classes, u5_ref = _u5_ref, rownames_u5_ref = _rownames_u5_ref):
    """
    predict_new_data takes in a pandas dataframe and the trained ccAFv2 model.

    Parameters
    ----------
    new_data : annData object
         New scRNA-seq dataset to be classified.
    species: string
         Species of the cells to be classified, currently supports 'human' and 'mouse'.
    gene_id: string
         Gene IDs for the scRNA-seq dataset, currently supports 'ensembl' and 'symbol'.
    threshold : float
        The threshold for likelihoods from the neural network classifier model.
    include_g0 : bool
        Whether or not to provide G0, G1, and Late G1 or to collapse them into a G0/G1 state. Best practice is to set to True if not applying to neuroepithelial derived cells.

    Returns
    -------
    pd.Series
        Series of labels for each single cell.

    """
    print('Running ccAFv2:')

    ## Prepare gene exprssion data for 861 features
    # Get a list of all genes based on species and gene ID
    genes = _genes_all[species+'_'+gene_id]

    # Prepare gene expression data for prediction
    prep_data = _prep_predict_data(new_data, genes)


    ## Add 10 angular bins
    # 1. Load up the U5 refrence loadings
    u5_ref_rownames = rownames_u5_ref[species+'_'+gene_id]
    u5_ref.index = u5_ref_rownames

    # 2. Project new_data onto reference
    new_data = project_cycle_from_anndata(new_data,
                                          u5_ref)

    projection = new_data.obsm["cc_Projection"]
    theta = new_data.obs["thetaPos"]

    # 3. Calculate angular bins
    n_bins = 10
    tmp_df = angles_to_vonmises_bins(list(theta), n_bins=n_bins, kappa=4.0)
    tmp_df.index = new_data.obs_names
    tmp_df_scaled = _scale(tmp_df.T).T
    prep_data_angBins = pd.concat([prep_data, tmp_df_scaled], axis=1)


    ## Predict probabilities for each class using classifier
    probs_df = _predict_classes(prep_data_angBins, classes = classes)

    print('  Choosing cell cycle state...')

    # Generate labels using argmax for the class
    labels = np.array([classes[np.argmax(i)] for i in probs_df.to_numpy()])

    # Threshold the probabilites, and set to unknown if not certain enough
    labels[np.where([np.max(i) < threshold for i in probs_df.to_numpy()])] = 'Unknown'

    # If include_g0 is set to False then combine Neural G0, G1, and Late G1
    if not include_g0:
        labels[np.where(labels=='Neural G0')] = 'G0/G1'
        labels[np.where(labels=='G1')] = 'G0/G1'
        labels[np.where(labels=='Late G1')] = 'G0/G1'
        labels = pd.Categorical(labels, categories=['G0/G1', 'S', 'S/G2', 'G2/M', 'M/Early G1', 'Unknown'], ordered=True)
    else:
        labels = pd.Categorical(labels, categories=['Neural G0', 'G1', 'Late G1', 'S', 'S/G2', 'G2/M', 'M/Early G1', 'Unknown'], ordered=True)
    print('Done.')
    return labels, probs_df


# Plot Umap of data using scanpy umap plot function
def plot_UMAP(data_scnpy,
              n_top_genes = 2000,
              show_figure = False,
              fig_save_path = None):

    """
    Plot UMAP of classified data

    Parameters
    ----------
    data_scnpy : annData object
        New scRNA-seq dataset to be classified.
    n_top_genes: int
        The number of genes to consider for pca.
    show_figure: bool
        If False then returns an axis object, if True then shows plot and doesn't return axis object.
    fig_save_path: string or pathlib object
        Path of figure or image to save.  File type is infered from the file extension (pdf, jpg, png, etc)

    Returns
    -------
    Generates Matplotlib figure
    Saves that figure
    """

    # Run UMAP of U5 hNSCs
    sc.pp.highly_variable_genes(data_scnpy, n_top_genes = n_top_genes)
    sc.tl.pca(data_scnpy)
    sc.pp.neighbors(data_scnpy)
    sc.tl.umap(data_scnpy)

    if fig_save_path is None:
        print('Overwriting previous plot')
        fig_save_path = '_ccAFv2_plot.pdf'

    # Prepare a color mapping dictionary
    plot_cmap = {"Neural G0": "#d9a428",
                     "G0/G1": "#FF6600",
                        "G1": "#f37f73",
                   "Late G1": "#1fb1a9",
                         "S": "#8571b2",
                      "S/G2": "#db7092",
                      "G2/M": "#3db270",
                "M/Early G1": "#6d90ca",
                   "Unknown": "#d3d3d3"}

    # Plot UMAP of U5 hNSCs
    sc.pl.umap(data_scnpy, color=['ccAFv2'], palette=plot_cmap, show = show_figure, save=fig_save_path)

    return None


# Plot circular manifold with cell cycle labels
def plot_circ_manifold(adata,
                       fig_save_path = None):

    edges = np.linspace(0, 2*np.pi, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    ccAFv2_colors['Neural G0'] = ccAFv2_colors['G0']
    ordered = {'G0':0,'G1':1,'Late G1':2,'S':3,'S/G2':4,'G2/M':5, 'M/Early G1':6}

    meta_data = adata.obs

    with PdfPages(fig_save_path) as pdf:
        fig1 = plt.figure(figsize=(6,6))
        plt.axis('off')
        # Unit circle
        theta = np.linspace(0, 2*np.pi, 500)
        plt.plot(np.cos(theta), np.sin(theta))

        # Bin boundaries
        for edge in edges[:-1]:
            plt.plot([0, np.cos(edge)],
                     [0, np.sin(edge)],
                     linestyle='--')

        # Points
        for class_name in ordered:
            if class_name=='G0':
                class_name2 = 'Neural G0'
            else:
                class_name2 = class_name
            print(class_name2)

            label_radius = (10-ordered[class_name])/10
            angles = np.array(meta_data.loc[meta_data['ccAFv2']==class_name2, 'theta'])
            df = pd.DataFrame({'angle': angles})
            # count duplicates within each angle
            df['rank'] = df.round(1).groupby('angle').cumcount()
            # spread from radius 0.9 to 1.1
            df['r'] = label_radius + (df['rank'] - df.groupby('angle')['rank'].transform('max')/2) * 0.0015
            df['x'] = df['r'] * np.cos(df['angle'])
            df['y'] = df['r'] * np.sin(df['angle'])

            plt.scatter(df['x'], df['y'], s=5, c=ccAFv2_colors[class_name], alpha=0.6)

        # Bin labels
        label_radius = 1.2
        for i, center in enumerate(centers):
            x = label_radius * np.cos(center)
            y = label_radius * np.sin(center)

            plt.text(
                x, y,
                f"Bin {i}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight='bold'
            )

        # Labels as multiples of p
        label_radius = 1.15

        for angle in edges:
            multiple = angle / np.pi
            plt.text(
                label_radius * np.cos(angle),
                label_radius * np.sin(angle),
                f"{multiple:.2f}$\pi$",
                ha="center",
                va="center"
            )

        plt.gca().set_aspect('equal')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)

        pdf.savefig(fig1)
        plt.close()

