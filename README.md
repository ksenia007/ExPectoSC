# Atlas of primary cell-type specific sequence models of gene expression and variant effects

## Code

### Main files 

→ python `add_new_dataset.py` <params> 
  - Preprocess .h5ad single cell expression file into the accepted format. User can specify groupKey (name of cell type column) and minimum number of cells the cell type needs to have.


→ python `train_new_model.py` <params>
  - Train a new set of models (one per cell type) using the provided dataset (pre-processed by `add_new_dataset.py`)


→ python `get_effect_predictions.py`  <params>
  - Get predictions for the set of variants. Full code includes running the deep learning encoder Module 1, which could be time consuming. 



## Background scripts

*main_train* - functions to train data, called by the `train_new_model.py` 

*condense_data* - functions to condense the data into dataX and dataY matrices

*data_utils* - additional data scripts

*encoder_for_variants* - all the scripts needed to run Beluga, such as running it on regions around TSS


## Overview

__Pulling TSS locations for genes__

Used GENCODE database and CAGE peaks databases to pull TSS locaitons. Used only protein coding genes, as noted in the Gencode. 
    - For CAGE used only *p1* peaks

Note: Used hg19 for both CAGE and Gencode. See below for more information on the datasources.

__Expression values__

`.h5ad` files are used for the group truth. Cell types with less than 100 cells are dropped (value can be adjusted).


__Running Beluga on variants for train__

Script uses 2 locations: one to save raw encodings, and another to save condensed encodings. Files are in `.hdf5` format, one file per gene (named as *"gene name"_"tss"* to allow for multiple TSS locations). The encodings run on the reference genome are named as *full_tss*, and the rest are saved with mutation information *location_alt*. File attributes are used to save additional information, such as TSS location, strand etc.  



__Module 2: Train regularized models__

Train cell type specific model, one per cell type. There are different training options: 

- Cross validation, test holdout (chromosomes 8 and 9) and full training. Cross validation is used to find the best parameters; evaluation of the existing models is done using the test dataset. Final models are trained with all the available data. 

- Subsetting is available for debugging purposes and to evaluate the effects of the size of the dataset on the results. When subsetting is used a fraction of the data is dropped randomly from the train. 

If the gene encodings do not exist prior to training those genes are dropped. 

