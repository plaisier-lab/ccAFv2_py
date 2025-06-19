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

from ._ccAFv2            import run_model
from importlib.resources import files

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
def _normalize_data(data, genes_classified):
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
    data.var_names_make_unique()
    
    # Remove all genes with zero counts.
    sc.pp.filter_genes(data, min_cells=1)

    # Create dense pandas arrays using the cells observerd and the union of the classified genes and genes in the dataset.
    data_index = data.obs_names
    data_cols  = list(set(data.var_names) & set(genes_classified))

    print(f'    Marker genes present in this dataset: {len(data_cols)}')
    print(f'    Missing marker genes in this dataset: {len(genes_classified)-len(data_cols)}')
    
    # Check to make sure that at lease 80% of classifier genes are present.
    if len(data_cols) > 0:

        # Z-score the data that is observed and overlaping with the gene_set data. 
        scaled_data = _scale(data[:,data_cols].X.toarray())
        data_temp   = pd.DataFrame(scaled_data, index = data_index, columns = data_cols)
      
        # Reindex the temporary array to add in missing columns/genes and ensure the same column order as the original classifer.  
        # Fill the missing values with the minimum values of the dataset.  
        data_zscored  = data_temp.reindex(columns = genes_classified, fill_value = data_temp.values.min())
    
        return data_zscored

    else:
        raise RuntimeError('Check species and gene_id, because there is no overlap between input genes and classifier genes!\n Or too few genes are present in this dataset')

# Use ccAFv2 classifier to calculate class probabilities
def _predict_new_data(new_data):
    """
    _predict_new_data takes in a pandas dataframe and the trained ccAFv2 model.

    Parameters
    ----------
    new_data : pd.DataFrame
        DataFrame of scRNA-seq data to be classified.  
        rows need to be cells and columns need to be the 861 classifer genes
        The gene order is given in the file ccAFv2_gens.csv
    
    Returns
    -------
    out_preds: flaot32 numpy array of probabilities 
        rows are the cell samples
        columns are classifiers given by the order in ccAFv2_classes.txt

    """
    if new_data.shape[1] != 861:
        raise RuntimeError('Input data frame not the right size for predictions.  Please check the integrity of the gene list')
    
    if not new_data.shape[0]:
        raise RuntimeError('There are no cells in the input data')

    print('  Predicting cell cycle state probabilities...')
    # Create output array to receive data.
    oup_preds = np.zeros((new_data.shape[0], 7), dtype = 'float32')

    # Change the pandas dataframe to a numpy array of float32 
    inp_data = new_data.to_numpy()
    inp_data = np.ascontiguousarray(inp_data, dtype = np.float64)

    # Loop through array to make predictions
    for ind in range(inp_data.shape[0]):
        oup_preds[ind,:] = run_model(inp_data[ind,:].reshape(1,861))
    
    return oup_preds

# Predict labels from class probabilities with rejection threshold
def predict_labels( new_data, species = 'human', 
                    gene_id = 'ensembl', 
                    threshold = 0.5, 
                    include_g0 = False,  
                    genes_all =_genes_all, 
                    pred_classes =_pred_classes):
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
    labels:  pd.Series:
        Series of labels for each single cell.
    
    probs: # samples x 8 float64 np.array
        Probabilies of classes with the last row being the 'Unknown' threshold catagory

    """

    if include_g0:
        class_map      = {  0 : 'G0/G1',
                            1 : 'G2/M',
                            2 : 'G0/G1',
                            3 : 'M/Early G1',
                            4 : 'G0/G1',
                            5 : 'S',
                            6 : 'S/G2',
                            7 : 'Unknown'}
    else:
        class_map       = { 0 : 'G1',
                            1 : 'G2/M',
                            2 : 'Late G1',
                            3 : 'M/Early G1',
                            4 : 'Neural G0',
                            5 : 'S',
                            6 : 'S/G2',
                            7 : 'Unknown'}


    # Determin the number of classes and add one more for the unknown value
    n_classes = len(class_map)

    # Create array fpr probability values.  initialize the array to the unknown class threshold value.  
    probs = np.full((new_data.shape[0], n_classes), fill_value = threshold, dtype = 'float32')

    print('Running ccAFv2:')

    # Select the correct species
    genes = genes_all[f'{species}_{gene_id}'].tolist()

    # Normalize data, organize gene order, and fill in empty values.
    pred_data = _normalize_data(new_data, genes)

    # Make predictions from the data using the ccAFv2 classifier.
    probs_raw = _predict_new_data(pred_data)
    
    # Fill in the predicted probabilities.
    probs[:, :n_classes-1] = probs_raw[:,:]

    print('  Choosing cell cycle state...')

    # Find the column of the highest probability value.
    probs_call = np.argmax(probs, axis = 1)

    # Create a pandas series and map the values to labels using the class_map dictionary.
    labels = pd.Series(data = probs_call, 
                       index = new_data.obs_names, 
                       name = 'Cell State').map(class_map).astype('category')

    print('Done.')
    return labels, probs