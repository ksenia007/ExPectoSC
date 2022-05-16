# CLEVER: Predicting single-cell-resolved gene expression from sequence

CLEVER = (Cell LEVel ExpRession)

## Running scripts

### Main files 

python `add_new_dataset.py` <params> 
   - Preprocess .h5ad single cell expression file into the accepted format. User can specify groupKey (name of cell type column) and miniumu number of cells the cell type needs to have.


python `train_new_model.py` <params>
    - Train a new set of models (one per cell type) using the provided dataset (pre-processed by `add_new_dataset.py`)


python `get_effect_predictions.py`  <params>
    - Get predictions for the set of variants. Full code includes running the deep learning encoder model, which could be time consuming. 



## Background scripts

*main_train* - functions to train data, called by the `train_new_model.py` 

*condense_data* - functions to condense the data into dataX and dataY matrices

*data_utils* - additional data scripts

*DS_for_variants* - all the scripts needed to run Beluga, such as running it on regions arouns TSS



# Need to check overview below (might be outdated)

## Overview

__Pulling TSS locations for genes__

Used GENCODE database and CAGE peaks databases to pull TSS locaitons. Used only protein coding genes, as noted in the Gencode. 
    - For CAGE used only *p1* peaks

To analyze effect of adding CAGE, created 2 datasets: **base dataset** and **only Gencode**. Base dataset uses TSS from CAGE where available, and dropped genes for which there was disagreement on chromosome or strand between CAGE/Gencode locations.

Note: Used hg19 for both CAGE and Gencode. See below for more information on the datasources.

__Expression values__

*Class_label* and *subclass_label* from sample cluster assignments were used as available groups. Three groups (Pericyte, Endothelial, VLMC) were dropped as there were less than 100 samples associated with them. Counts (exon+intron) were normalized as CPM: 10^6 * count detected/sum of counts in the sample, and mean was taken across samples in the same group. Resulting heatmap for the genes in the *base dataset*:

<p align="center">
  <img src="/images/scRNAseq_expressions/CPM_base_gene_set.png" width="700" title="hover text">
</p>

Pseudocount of 0.001, followed by log2 was used later. For such values, distribution of expressions by group were:

<p align="center">
  <img src="/images/scRNAseq_expressions/CPM_dist_by_group.png" width="450" title="hover text">
</p>

There are 165 genes that have 0 expression for every group. In the previous iternation we've shown that these genes are also lowly expressed in the GTEx brain data. Keep them for now. 

__Running DeepSea (DS) on variants for train__

Save files as *hdf5*. DeepSea predictions are saved as *full_tss* dataset in the file, and attributes are used to save additional information. Can use *load_gene()* from *condense_data.py* to get this value. Each file is named as *"gene name"_"tss"* to allow for multiple TSS locations. DS is ran forward and backward and then averaged. 

__Condense DS outputs__

Used exponential condense for DS outputs, in the spirit of Expecto. 

__Train xgboost and holdout performance__

Split the dataset 80-20, based on the gene names. 20% is used for validation. Even without grid search and fixed train (stopped before convergence), results on this holdout valiation are good. 

<p align="center">
  <img src="/images/xgboost_results/spearman_first_xgboost_run.png" width="500" title="hover text">
</p>

<p align="center">
  <img src="/images/xgboost_results/pearson_first_xgboost_run.png" width="450" title="hover text">
  <img src="/images/xgboost_results/scatter_GABAergic.png" width="350" title="hover text">
</p>


# TODO

❗️❗️❗️ Evaluation other than holdout

❗️Run DS for evaluation datasets (takes time)

❗️Move to Simons server if needed

❗️Run AMBER on the new data

❗️Other way to condense, try to use non-condensed as it gives richer representation

# Notes

Running bedtools: bedtools closest -d -a hg19.baselineLD_All.20180423.bed -b fantom_middle_peak_bed.bed > cage_closest_all.txt

# Datasources:

CAGE:
https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/extra/CAGE_peaks/
hg38_fair+new_CAGE_peaks_phase1and2.bed.gz
Access: Sept 15

Gencode:
https://www.gencodegenes.org/human/
Comprehensive gene annotation gff3
Release 35 (GRCh38.p13)
Access: Sept 15

Fasta hg38.p13 [not used right now]
https://www.gencodegenes.org/human/ 
assembly, genomic fasta (.fna)
Access: Sept 15

Single-cell datasets:

Downloaded from:
https://portal.brain-map.org/atlases-and-data/rnaseq/human-multiple-cortical-areas-smart-seq

Access date: Sept 15, 2020

Data source:
https://portal.brain-map.org/atlases-and-data/rnaseq/human-m1-10x

Access date: Sept 15, 2020


# Quick links:

* Presentation made in lab (February 27th) https://docs.google.com/presentation/d/191W60VRJS1NQb20LId0jEU4WXXNWAlN_zJpZbaVnHg0/edit#slide=id.p 

* Dataset webpage https://portal.brain-map.org/atlases-and-data/rnaseq

* Expression bulk tissue lookup https://amp.pharm.mssm.edu/archs4/

* How GTEx works etc (documentation) https://www.gtexportal.org/home/documentationPage

* Using h5py to read .tome files https://community.brain-map.org/t/loading-data-from-transcrip-tome-file-in-python-with-h5py/310 


