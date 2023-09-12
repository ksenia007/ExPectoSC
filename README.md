# Atlas of primary cell-type specific sequence models of gene expression and variant effects

## Quick setup

Download and unzip the pre-requisite file (note this might take a couple of minutes):

```
cd resources
wget https://humanbase.s3.us-west-2.amazonaws.com/clever/deepsea.beluga.pth
cd ..
```

Please note that you will also need .fasta file for your build (`hg19.fa` needs to be downloaded and put into `resources/reference_files/`)


Setup the environment. Make sure you have a working cuda-enabled pytorch. See below for an example

```
python3.6 -m venv expectosc_env
source expectosc_env/bin/activate
pip install -r requirements.txt
pip install selene-sdk
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

### Variant effect prediction
See `resources/test.csv` for an example .csv file. See `example_shell.sh` for getting predictions for the .csv with variants. 


## More details

### Main files 

→ python `add_new_dataset.py` <params> 
  - Preprocess .h5ad single cell expression file into the accepted format. User can specify groupKey (name of cell type column) and minimum number of cells the cell type needs to have.


→ python `train_new_model.py` <params>
  - Train a new set of models (one per cell type) using the provided dataset (pre-processed by `add_new_dataset.py`)


→ python `get_effect_predictions.py`  <params>
  - Get predictions for the set of variants. Full code includes running the deep learning encoder Module 1, which could be time consuming. This assumes the data is formatted as : \[chrom, snp_chromStart, snp_chromEnd, ref, alt, SNP, chrom_gene, tss, chromEnd, strand, gene_name, cage_used, dist\]. Note that snp_chromStart is the location of the variant; SNP is the ID of choice, for example chrom_variantLoc_ref_alt. 




## Background scripts

*main_train* - functions to train data, called by the `train_new_model.py` 

*condense_data* - functions to condense the data into dataX and dataY matrices

*data_utils* - additional data scripts

*encoder_for_variants* - all the scripts needed to run Beluga, such as running it on regions around TSS


## Overview

__Pulling TSS locations for genes__

Used GENCODE database and CAGE peaks databases to pull TSS locations (only protein coding genes, as noted in the Gencode). 
    - For CAGE used only *p1* peaks

Note: Used hg19 for both CAGE and Gencode. See below for more information on the datasources.

__Expression values__

`.h5ad` files are used for the group truth. Cell types with less than 100 cells are dropped (value can be adjusted).


__Running encoder on variants for train__

Script uses 2 locations: one to save raw encodings, and another to save condensed encodings. Files are in `.hdf5` format, one file per gene (named as *"gene name"_"tss"* to allow for multiple TSS locations). The encodings run on the reference genome are named as *full_tss*, and the rest are saved with mutation information *location_alt*. File attributes are used to save additional information, such as TSS location, strand etc. The weights for the model can be downloaded [here](https://humanbase.readthedocs.io/en/latest/clever.html#download)



__Module 2: Train regularized models__

Train cell type specific model, one per cell type. There are different training options: 

- Cross validation, test holdout (chromosomes 8 and 9) and full training. Cross validation is used to find the best parameters; evaluation of the existing models is done using the test dataset. Final models are trained with all the available data. 

- Subsetting is available for debugging purposes and to evaluate the effects of the size of the dataset on the results. When subsetting is used a fraction of the data is dropped randomly from the train. 

If the gene encodings do not exist prior to training those genes are dropped. 

