"""
Main file for adding new datasets. 

Assume .h5ad files, with per-cell non-condensed expression and cell type assignments in file (groupKey)
"""

import argparse
import logging
import sys
import os
import pickle
import pandas as pd
from data_utils import *
import scanpy as sc

parser = argparse.ArgumentParser()

parser.add_argument('--file', default=None, type=str,
                    help="Path to file with expression. Currently support only h5ad files")
parser.add_argument('--save-location', default=None, type=str,
                    help="Path to folder to save results")
parser.add_argument('--new-filename', default=None, type=str,
                    help="Filename for the new expressions file")
parser.add_argument('--groupKey', default='CellType', type=str,
                    help="Name of the column containing cell type assignments")

parser.add_argument('--min_number_cells', default=100, type=int,
                    help="Minimum number of cells a cell type should contain.")


def process_h5ad_dataset(adata_path, groupKey, logging,
                         min_number_cells=100,
                         save_counts=True, save_counts_path=''):
    """Process given dataset, assuming groupKey contains the cell type assignment"""
    adata = sc.read_h5ad(adata_path)
    if adata.shape[1] < 50:
        # possibly dealing with umap instead of full expression values
        print('Warning, small dimnesionality of the dataset')
        logging.warning(
            'Warning, small dimnesionality of the dataset. Possibly UMAP is provided')

    means = grouped_obs_mean(adata, groupKey)
    counts = count_ncells_celltype(adata, groupKey)
    if save_counts:
        counts.to_csv(save_counts_path+'/number_cells_per_group.csv')

    means = means[counts[counts.n_counts > min_number_cells].index.values]
    means['micro_mean'] = np.array(adata.X.mean(0)).flatten()

    return means

args = parser.parse_args()

if not os.path.exists(args.save_location):
    os.makedirs(args.save_location)

logging.basicConfig(filename=args.save_location+'/INFO.log', filemode='w',
                    format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.info('Start processing {}'.format(args.file))

means = process_h5ad_dataset(args.file, args.groupKey, logging, save_counts_path = args.save_location)

logging.info('Collected means for the dataset')
logging.info('Collected a total of {} genes and {} cell types'.format(means.shape[0], means.shape[1]))
logging.info(means.columns.values)

# for downstream we want transposed dataframe
means_t = means.transpose()

means_t.to_csv(args.save_location+'/'+args.new_filename)

logging.info('DONE')
logging.info('***')