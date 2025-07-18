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

import numpy  as np
import pandas as pd
import scanpy as sc
import pathlib
import warnings

from typing import Optional, Literal, Dict, List

from ._ccAFv2            import run_model
from anndata             import AnnData
from importlib.resources import files
import matplotlib.pyplot as plt

from scipy.stats           import zscore
from sklearn.preprocessing import StandardScaler
sc.settings.verbosity = 0

in_path = files("ccAFv2").joinpath("ccAFv2_genes.csv")
_genes_all = pd.read_csv(in_path, index_col=0, header=0)

in_path = files("ccAFv2").joinpath("ccAFv2_classes.txt")
_pred_classes = tuple(pd.read_csv(in_path, header=None)[0])

###############
## Functions ##
###############

# Scale data for classification
def _scale(data):
    """
    Standardize or normalize numeric data using scikit-learn's StandardScaler.

    Parameters:
    - data: 2D NumPy array or list of lists

    Returns:
    - Scaled data (NumPy array)
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Prepare test data for predicting
def _normalize_data(data_scnpy, genes_classified):
    """
    prep_predict_data takes in a pandas dataframe and the trained ccAFv2 model.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame of scRNA-seq data to be classified.
    genes_classified :  List 
        List of genes and order of the genes the classifier is trained on.

    Returns
    -------
    pd.Series
        Series of labels for each single cell.

    """

    min_genes = 689

    print('    Preparing data for classification...')
   # Make indicies unique just incase there are doubling up on gene names.  
    data_scnpy.var_names_make_unique()
    
    # Remove all genes with zero counts.
    sc.pp.filter_genes(data_scnpy, min_cells=1)

    # Create dense pandas arrays using the cells observerd and the union of the classified genes and genes in the dataset.
    data_index = data_scnpy.obs_names
    data_cols  = list(set(data_scnpy.var_names) & set(genes_classified))

    num_marker_genes = len(data_cols)
    num_missing_genes = len(genes_classified)-len(data_cols)

    print(f'    Marker genes present in this dataset: {len(data_cols)}')
    print(f'    Missing marker genes in this dataset: {len(genes_classified)-len(data_cols)}')
    
    # Raise a warning if too few genes are present in the dataset.
    if min_genes >= num_marker_genes:
        warnings.warn("Less than 80% of marker genes are present in this dataset.  The predictions may not be accurate", UserWarning)

    # Check to make sure that at lease 80% of classifier genes are present.
    if len(data_cols) > 0:

        # Z-score the data that is observed and overlaping with the gene_set data. 
        scaled_data = _scale(data_scnpy[:,data_cols].X.toarray())
        data_temp   = pd.DataFrame(scaled_data, index = data_index, columns = data_cols)
      
        # Reindex the temporary array to add in missing columns/genes and ensure the same column order as the original classifer.  
        # Fill the missing values with the minimum values of the dataset.  
        data_zscored  = data_temp.reindex(columns = genes_classified, fill_value = data_temp.values.min())
    
        return data_zscored

    else:
        raise RuntimeError('Check species and gene_id, because there is no overlap between input genes and classifier genes!\n Or too few genes are present in this dataset')

# Create the correct class_map that includes g0 or excludes g0
def _create_classmap(include_g0 = False):
    '''
    Creates a dictionary class_map to relate indicies of the ccAFv2 classifier output vector to 
    values returned from a numpy argmax function. Since there are only 7 classes and then one 
    threshold value.  

    Parameters
    ----------
    include_g0:  bool default to False.  

    Returns
    ----------
    class_map:  Dict dictionary mapping values to class labels 
    '''
    if include_g0:
        class_map       = { 0 : 'G1',
                            1 : 'G2/M',
                            2 : 'Late G1',
                            3 : 'M/Early G1',
                            4 : 'Neural G0',
                            5 : 'S',
                            6 : 'S/G2',
                            7 : 'Unknown'}
        
    else:
        class_map      = {  0 : 'G0/G1',
                            1 : 'G2/M',
                            2 : 'G0/G1',
                            3 : 'M/Early G1',
                            4 : 'G0/G1',
                            5 : 'S',
                            6 : 'S/G2',
                            7 : 'Unknown'}
      
    return class_map

# Check the figure files to make sure that a path exists.  
def _check_path(file_name):
    '''
    Checks if path is an absolute path.  If not it assumes a single file name, creates a figure 
    directory, and then returns a Path for saving the file.  
    
    Parameters:
        file_name (str): The name of the file (e.g., 'plot.png').
    
    Returns:
        Path: Either absolute path provided as Path object or path of file in 'figures' directory.
    '''

    temp_path =  pathlib.Path(file_name)
    # Check if path is absolute.  
    if temp_path.is_absolute():
        # Return the temp_path as a complete path
        return temp_path
    else:
        # Otherwise create a figure directory 
        current_path = pathlib.Path.cwd()
        # Define the figures directory
        figure_dir = current_path / 'figures'

        figure_dir.mkdir(exist_ok = True)

        return figure_dir / file_name
        
# Use ccAFv2 classifier to calculate class probabilities
def _predict_classes(data_scnpy):
    """
    _predict_classes takes in a pandas dataframe and the trained ccAFv2 model.

    Parameters
    ----------
    data_scnpy : pd.DataFrame
        DataFrame of scRNA-seq data to be classified.  
        rows need to be cells and columns need to be the 861 classifer genes
        The gene order is given in the file ccAFv2_gens.csv
    
    Returns
    -------
    out_preds: flaot32 numpy array of probabilities 
        rows are the cell samples
        columns are classifiers given by the order in ccAFv2_classes.txt

    """
    if data_scnpy.shape[1] != 861:
        raise RuntimeError('Input data frame not the right size for predictions.  Please check the integrity of the gene list')
    
    if not data_scnpy.shape[0]:
        raise RuntimeError('There are no cells in the input data')

    print('  Predicting cell cycle state probabilities...')
    # Create output array to receive data.
    oup_preds = np.zeros((data_scnpy.shape[0], 7), dtype = 'float32')

    # Change the pandas dataframe to a numpy array of float32 
    inp_data = data_scnpy.to_numpy()
    inp_data = np.ascontiguousarray(inp_data, dtype = np.float64)

    # Loop through array to make predictions
    for ind in range(inp_data.shape[0]):
        oup_preds[ind,:] = run_model(inp_data[ind,:].reshape(1,861))
    
    return oup_preds

# Predict labels from class probabilities with rejection threshold
def predict_labels(data_scnpy: Optional[AnnData] = None, 
                    species: Literal['human', 'mouse'] = 'human', 
                    gene_id: Literal['ensembl', 'symbol'] = 'ensembl', 
                    threshold: float = 0.5, 
                    include_g0: bool = False,  
                    genes_all: Dict[str, List[str]] = _genes_all):
    """
    predict_data_scnpy takes in a pandas dataframe and the trained ccAFv2 model.

    Parameters
    ----------
    data_scnpy : annData object
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
    labels:  pd.Series:
        Series of labels for each single cell.
    
    probs: # samples x 8 float64 np.array
        Probabilies of classes with the last row being the 'Unknown' threshold catagory

    """

    class_map = _create_classmap(include_g0)
    
    # Determin the number of classes and add one more for the unknown value
    n_classes = len(class_map)

    # Create array fpr probability values.  initialize the array to the unknown class threshold value.  
    probs_thresh = np.full((data_scnpy.shape[0], n_classes), fill_value = threshold, dtype = 'float32')

    print('Running ccAFv2:')

    # Select the correct species
    genes = genes_all[f'{species}_{gene_id}'].tolist()

    # Normalize data, organize gene order, and fill in empty values.
    pred_data = _normalize_data(data_scnpy, genes)

    # Make predictions from the data using the ccAFv2 classifier.
    probs = _predict_classes(pred_data)
    
    # Fill in the predicted probabilities.
    probs_thresh[:, :n_classes-1] = probs

    print('  Choosing cell cycle state...')

    # Find the column of the highest probability value.  
    # Argmax gives the equivalent of >= thresh since multiple values will return the lower index.
    probs_call = np.argmax(probs_thresh, axis = 1)

    # Create a pandas series and map the values to labels using the class_map dictionary.
    labels = pd.Series(data = probs_call, 
                       index = data_scnpy.obs_names, 
                       name = 'Cell State').map(class_map).astype('category')

    print('Done.')
    return labels, probs

# Get indicies of cells to order them in the cell cycle.
def cellcycle_order(class_probs: np.ndarray | None = None,
                    clock_offset: float = None,
                    include_g0: bool = False):
    """
    predict cells order in cell cycle using classifier prediction probabilities

    Parameters
    ----------
    class_probs : numpy float array 
        The class probabilites (predictions) generated from predict_labels function
   
    clock_offset : float 
        The offset to rotate the clock vectors so that G0 or G0/G1 state is not split betweem the begining 
        and end of the dataset.  

    include_g0 : bool
        Whether or not to provide G0, G1, and Late G1 or to collapse them into a G0/G1 state. Best practice is to set to True if not applying to neuroepithelial derived cells.

    Returns
    -------
    cell_ordered_ind:  numpy int array
        The indicies of the ordered cells.  This doesn't return the sorted data but the indicies to order the data.

    """

    if (clock_offset is None) and include_g0:
        clock_offset = np.pi/25

    elif (clock_offset is None) and not include_g0:
        clock_offset = np.pi/12.5


    if include_g0:

        # Initialize arrays for calculating cell angle
        cycle_order = np.array([4,0,2,5,6,1,3], dtype = 'int')
        thetas = np.tile( np.arange(0,2*np.pi, 2*np.pi/7) + clock_offset, [class_probs.shape[0], 1])
        xy_vecs = np.zeros((class_probs.shape[0], 2), dtype = 'float')
        
        # Re-order the predictions 
        vec_r = class_probs[:,cycle_order] 

        # Calculate the xy ordinates for clock positions.
        vec_x = vec_r * np.cos(thetas)
        vec_y = vec_r * np.sin(thetas)
        xy_vecs[:,0] = np.sum(vec_x, axis = 1)
        xy_vecs[:,1] = np.sum(vec_y, axis = 1)

        # Find the theta angle from 0 to 2pi.
        ang = np.arctan2(xy_vecs[:,1], xy_vecs[:,0]) 
        ang[ang<0] = ang[ang<0]+2*np.pi

        # Sort the angles for returning the indicies.  
        cell_ordered_inds = np.argsort(ang)

    else:

        # Combine Neural G0, G1 and Late G1 into a single class
        condensed_probs = np.zeros((class_probs.shape[0],5), dtype = class_probs.dtype)
        condensed_probs[:,0]  = np.sum(class_probs[:,[0,2,4]], axis = 1)
        condensed_probs[:,1:] = class_probs[:,[5,6,1,3]]

        # Initialize arrays for calculating cell angle
        thetas = np.tile( np.arange(0,2*np.pi, 2*np.pi/5) + clock_offset, [class_probs.shape[0], 1])
        xy_vecs = np.zeros((class_probs.shape[0], 2), dtype = 'float')

        # Calculate the xy ordinates for clock positions.       
        vec_r = condensed_probs
        vec_x = vec_r * np.cos(thetas)
        vec_y = vec_r * np.sin(thetas)
        xy_vecs[:,0] = np.sum(vec_x, axis = 1)
        xy_vecs[:,1] = np.sum(vec_y, axis = 1)

        # Find the theta angle from 0 to 2pi.
        ang = np.arctan2(xy_vecs[:,1], xy_vecs[:,0]) 
        ang[ang<0] = ang[ang<0]+2*np.pi

        # Sort the angles for returning the indicies.  
        cell_ordered_inds = np.argsort(ang)

    return cell_ordered_inds 

# Plot Umap of data using scanpy umap plot function
def plot_UMAP(data_scnpy: Optional[AnnData] = None,
              n_top_genes: int = 2000,
              show_figure: bool = False,
              fig_save_path: pathlib.Path | str | None = None, ):

    """
    Plot UMAP of classified data

    Parameters
    ----------
    data_scnpy : annData object
        New scRNA-seq dataset to be classified.
    fig_save_path: string or pathlib object
        Path of figure or image to save.  File type is infered from the file extension (pdf, jpg, png, etc)
    n_top_genes: int 
        The number of genes to consider for pca.  
   
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

# Plot class frequency against prediction minnimum threshold values
def plot_threshold(class_probs: np.ndarray | None = None, 
                   threshold_levels: np.ndarray | None = None, 
                   include_g0: bool = False,  
                   show_figure: bool = False,
                   fig_save_path: pathlib.Path | str | None = None):

    """
    Generate threshold plot of class frequency vs probability threshold

    Parameters
    ----------
    class_probs : float numpy array of num samples by num classes of class probabilities 
        This numpy array is generated from 

    threshold_levels: float numpy array of range 0 to 1 or None  
        Path of figure or image to save.  File type is infered from the file extension (pdf, jpg, png, etc)
    
    include_g0: bool  
        The number of genes to consider for pca.  
    
    file_path:  None, string, or pathlib object
   
    Returns
    -------
    Generates Matplotlib figure
    Saves that figure 

    """

    # Create default threshold level vector
    if threshold_levels is None:
        threshold_levels = np.arange(0, 1, 0.1)    

    fig_save_path = "ccAGv2_threshold_plot.pdf" if fig_save_path is None else fig_save_path
    # Check the file path for saving figures 
    file_path = _check_path(fig_save_path)

    # create class_map
    class_map = _create_classmap(include_g0)

    # find number of samples 
    n_samples = class_probs.shape[0]

    # Determin the number of classes and add one more for the unknown value
    n_classes = len(class_map)

    # Create array fpr probability values.  initialize the array to the unknown class threshold value.  
    probs_thresh = np.empty((n_samples, n_classes), dtype = 'float64')
    probs_call   = np.empty((n_samples, threshold_levels.shape[0]), dtype = 'int')
    
    class_counts = np.zeros((n_classes, threshold_levels.shape[0]), dtype = 'int')
    # Load in class probabilities for compairison to thresholds
    probs_thresh[:,:-1] = class_probs[:,:]
    threshold_labels = []

    # loop through threshold levels 
    for ind, threshold in enumerate(threshold_levels, start = 0):
        # set probabilities
        probs_thresh[:,-1] = threshold

        # find which class has highest probability 
        probs_call[:, ind] = np.argmax(probs_thresh, axis =1)

        # Use np.unique values and counts.  This could be faster using fasthistrogram histogram1d
        vals,  counts = np.unique(probs_call[:, ind], return_counts = True)
        
        class_counts[vals,ind] = counts

        threshold_labels.append(f'{threshold:.1f}')

    # Calculate class percentages
    class_perc = class_counts / n_samples
    
    # Since we can have duplicate labels in the class_map (if we ignore g0) we need to make a list 
    class_labels = list(set([item for item in class_map.values()]))
    class_labels.sort(reverse = True)

    weight_counts = { label : np.zeros((threshold_levels.shape[0]), dtype = 'float') for label in class_labels }
   
    # Consolidate weight percentages 
    for vals, label in class_map.items():
        weight_counts[label] += class_perc[vals,:]

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
    width = 0.8

    fig, ax = plt.subplots(layout='constrained')
    bottom  = np.zeros(threshold_levels.shape[0])

    for label, weight_count in weight_counts.items():
        bar_color = plot_cmap[label]
        p = ax.bar(threshold_labels, weight_count, width, label=label, bottom=bottom, color = bar_color)
        bottom += weight_count

    ax.set_xlabel('Threshold Level')
    ax.set_ylabel('Class frequency')
    ax.set_title("Class frequency at threshold level")
    fig.legend(loc="outside center right", frameon=False, reverse = True  )

    fig.savefig(file_path)

    if show_figure:
        plt.show()

    return None

# Plot Cell-cycle clock to show progress of cells through 
def plot_cellcycleclock(class_probs: np.ndarray,
                        labels: List[str], 
                        clock_offset: float | None = None,
                        include_g0: bool = False,
                        show_figure: bool = False,
                        fig_save_path: pathlib.Path | str | None = None):

    """
    predict cells order in cell cycle using classifier prediction probabilities

    Parameters
    ----------
    class_probs : numpy float array 
        predictions (preds) ourput from predict_labels function
   
    clock_offset : float 
        The offset to rotate the clock vectors so that G0 or G0/G1 state is not split betweem the begining 
        and end of the dataset.  

    include_g0 : bool
        Whether or not to provide G0, G1, and Late G1 or to collapse them into a G0/G1 state. Best practice is to set to True if not applying to neuroepithelial derived cells.

    Returns
    -------
    cell_ordered_ind:  numpy int array
        The indicies of the ordered cells.  This doesn't return the sorted data but the indicies to order the data.

    fig: Matplotlib figure handle 
        Matplotlib figure handle for saving the figure if needed.
    """
    
    
    fig_save_path = "ccAFv2_clock_plot.pdf" if fig_save_path is None else fig_save_path
    # Check the file path for saving figures 
    file_path = _check_path(fig_save_path)

    if (clock_offset is None) and include_g0:
        clock_offset = np.pi/25

    elif (clock_offset is None) and not include_g0:
        clock_offset = np.pi/12.5

    if include_g0:

        lbl_array   = ["Neural G0", "G1", "Late G1", "S","S/G2", "G2/M", "M/Early G1"]
        offsets = [[-0.2,-0.1],
                    [0,0],
                    [-0.1,0],
                    [-0.05, 0],
                    [-0.05,-.05],
                    [-0.05,-0.05],
                    [0,0]]

        # Initialize arrays for calculating cell angle
        cycle_order = np.array([4,0,2,5,6,1,3], dtype = 'int')
        thetas = np.tile( np.arange(0,2*np.pi, 2*np.pi/7) + clock_offset, [class_probs.shape[0], 1])
        xy_vecs = np.zeros((class_probs.shape[0], 2), dtype = 'float')
        
        # Re-order the predictions 
        vec_r = class_probs[:,cycle_order] 

        # Calculate the xy ordinates for clock positions.
        vec_x = vec_r * np.cos(thetas)
        vec_y = vec_r * np.sin(thetas)
        xy_vecs[:,0] = np.sum(vec_x, axis = 1)
        xy_vecs[:,1] = np.sum(vec_y, axis = 1)

        # Find the theta angle from 0 to 2pi.
        ang = np.arctan2(xy_vecs[:,1], xy_vecs[:,0]) 
        ang[ang<0] = ang[ang<0]+2*np.pi

        # Sort the angles for returning the indicies.  
        cell_ordered_inds = np.argsort(ang)

    else:

        lbl_array   = ["G0/G1", "S", "S/G2", "G2/M", "M/Early G1"]
        offsets = [[-0.2,-0.1],
                [0,0],
                [-0.1,0],
                [-0.05, 0],
                [-0.05,-.05],
                [-0.05,-0.05],
                [0,0]]

        cycle_order = [0,5,6,1,5]

        # Combine Neural G0, G1 and Late G1 into a single class
        condensed_probs = np.zeros((class_probs.shape[0],5), dtype = class_probs.dtype)
        condensed_probs[:,0]  = np.sum(class_probs[:,[0,2,4]], axis = 1)
        condensed_probs[:,1:] = class_probs[:,[5,6,1,3]]

        # Initialize arrays for calculating cell angle
        thetas = np.tile( np.arange(0,2*np.pi, 2*np.pi/5) + clock_offset, [class_probs.shape[0], 1])
        xy_vecs = np.zeros((class_probs.shape[0], 2), dtype = 'float')

        # Calculate the xy ordinates for clock positions.       
        vec_r = condensed_probs
        vec_x = vec_r * np.cos(thetas)
        vec_y = vec_r * np.sin(thetas)
        xy_vecs[:,0] = np.sum(vec_x, axis = 1)
        xy_vecs[:,1] = np.sum(vec_y, axis = 1)

        # Find the theta angle from 0 to 2pi.
        ang = np.arctan2(xy_vecs[:,1], xy_vecs[:,0]) 
        ang[ang<0] = ang[ang<0]+2*np.pi

        # Sort the angles for returning the indicies.  
        cell_ordered_inds = np.argsort(ang)


    plot_cmap = {"Neural G0": "#d9a428",
                     "G0/G1": "#FF6600", 
                        "G1": "#f37f73", 
                   "Late G1": "#1fb1a9", 
                         "S": "#8571b2", 
                      "S/G2": "#db7092", 
                      "G2/M": "#3db270",
                "M/Early G1": "#6d90ca",  
                   "Unknown": "#d3d3d3"}

    cmap_list = []
    for lbl in labels:
        cmap_list.append(plot_cmap[lbl])

    # Plot the cell cycle clock
    fig, ax = plt.subplots(figsize = [8,8])
    ax.set_title('ccAFv2 Clock Plot', fontweight = 'bold')

    # Loop through intividual points and plot them.
    for n in range(xy_vecs.shape[0]):
        ax.plot(xy_vecs[n,0], xy_vecs[n,1], '.', color = cmap_list[n])

    ax.set_xlim(-1.25,1.25)
    ax.set_ylim(-1.25,1.25)

    symbol_list = ['.','.', 's', '^', 'v', 's', 'v']

    for n, (angle, text_label) in enumerate(zip(thetas[0,:], lbl_array), start = 0):
        ax.plot([0,np.cos(angle)], [0, np.sin(angle)], '-', color = [0,0,0,0.4])
        ax.plot(np.cos(angle), np.sin(angle), marker = symbol_list[n], color = plot_cmap[text_label])

        x =  1.05*np.cos(angle)+offsets[n][0]
        y =  1.05*np.sin(angle)+offsets[n][1]
        ax.text(x,y, text_label)

    fig.savefig(file_path)   
    if show_figure: 
        plt.show()
    
    return cell_ordered_inds, fig