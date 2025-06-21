##########################################################
## OncoMerge:  ccAFve_pytest.py                         ##
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
## @Author:  Chris Plaisier, Samantha O'Connor          ##
## @License:  GNU GPLv3                                 ##
##                                                      ##
## If this program is used in your analysis please      ##
## mention who built it. Thanks. :-)                    ##
##########################################################

# Run this file to test ccAFv2 installation and scanpy functions


##########################################
## Load Python packages for classifiers ##
##########################################

import ccAFv2
import h5py
import numpy  as np
import scanpy as sc
import pathlib

import matplotlib.pyplot as plt


def _getHDF5str(str_array):
    '''Generate a string array from individual bchar values'''

    return ''.join(c.astype(str)  for c in str_array.squeeze())

def _getHDF5Data(hdf5Obj):
    """
    Recursively load data from HDF5 file.  
    """
    d = {}
    for key, item in hdf5Obj.items():
 
        if   isinstance(item, h5py.Dataset):
            if item.dtype == 'S10':
                d[key] = _getHDF5str(item[()])
            else: 
              
                d[key] = item[()]
          
        elif isinstance(item, h5py.Group):
            d[key] = _getHDF5Data(item)

    return d

def loadData(filename):

    """
    Load data from HDF5 file into a python Dictionary
    This function will attempt to load MATLAB *.mat files based on thier 
    file names.
    There maybe problems with the dictionary returned in that it may need 
    to be squeezed.
    """
    d = {}
    with h5py.File(filename, 'r') as hdf5Obj:
        d =  _getHDF5Data(hdf5Obj)
        
    return d

if __name__ == "__main__":
    lbl_encoder_dict    = { 0 : 'G1',
                            1 : 'G2/M',
                            2 : 'Late G1',
                            3 : 'M/Early G1',
                            4 : 'Neural G0',
                            5 : 'S',
                            6 : 'S/G2',
                            7 : 'Unknown'}

    print('Loading test data and comparison data...')

    # Load hdf5 input and output data to test the classifier installation
    oup_data_path = pathlib.Path("./tests/R_ccAFv2_out.hdf5")
    oup_data = loadData(oup_data_path)

    inp_data_path = pathlib.Path("./tests/R_ccAFv2_inp.hdf5")
    inp_data = loadData(inp_data_path)

 
    print('Running model...')
    # run the loop.  Hey who knew by default hdf5 files were C_contiguous
    oup_test = np.zeros(oup_data['U5'].shape, dtype = 'float32')
    for ind in range(inp_data['U5'].shape[0]):
        
        oup_test[ind,:] = ccAFv2.run_model(inp_data['U5'][ind,:].reshape(1,861))
    
    print('Calculating model difference in predictions...')
    predict_delta = oup_data['U5']-oup_test

    # take the mean and standard deviation of the output data and the test data
    mean_diff = np.mean(predict_delta, axis = 0)
    sdev_diff = np.std(predict_delta, axis = 0)

    # Make a formated string array to print the data we just calculated
    formatted_array = np.array([f'{val.strip()} stage mean difference = {mn:.2e} \u00B1 {sd:.2e}' for (key, val), mn, sd in zip(lbl_encoder_dict.items(), mean_diff, sdev_diff)])

    print('Difference between predictions \nR model and python model\n')
    for n in formatted_array:
        print(f'{n}\n')

    if not any(mean_diff > 0.03):
        print('All test predictions are satisfactory')
    else:
        print('The test predictions did not match previous predictions.')

    #  Load up test dataset sorry I have an old path in here you may need to change that.
    data_path = pathlib.Path("../Data/W8-1_normalized_ensembl.h5ad")
    pwc8_scdata = sc.read_h5ad(data_path)

    # Run ccAFv2 to predict cell labels
    labels, predictions = ccAFv2.predict_labels(pwc8_scdata, species='human', gene_id='ensembl', include_g0 = True)

    print('Adding predicted labels to dataset')
    # Save into scanpy object
    pwc8_scdata.obs['ccAFv2'] = labels

    print('Performing UMAP plotting')
    ccAFv2.plot_UMAP(pwc8_scdata)

    print('Generating Threshold Plot')
    ccAFv2.plot_threshold(class_probs = predictions, include_g0 = True)



    