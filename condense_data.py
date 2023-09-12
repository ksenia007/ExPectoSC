# condense scripts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List

from data_utils import *


def condense_gene_exponential(DS_predictions):
    """Condense DS predictions for the given gene."""
    shifts = np.array(list(range(-19000, 21200, 200)))/200
    weights = np.vstack([[np.exp([(-p*np.abs(shifts))
                                  for p in [0.2, 0.1, 0.05, 0.02, 0.01, 0.01, 0.02, 0.05, 0.1, 0.2, ]])]])
    return np.dot(DS_predictions, weights[0].transpose()).flatten()


def load_gene(gene_name: str, tss: int, folder: str, dataset: str = 'full_tss'):
    """Load gene data for the given gene and TSS in the folder.
    dataset - name of the dataset in the h5py file, allows to call for specific mutation"""

    filepath = generate_filepath(gene_name, tss, folder)

    if not os.path.exists(filepath):
        print('Gene not found: ', filepath)
        return None

    try:
        data = h5py.File(filepath, "r")
        return data[(dataset)][0]
    except:
        print('Error loading file: ', filepath)
        return None

    return None


def condense_mutations_list(condense_gene_fn, gene_name: str, tss: int, ds_noncondensed_folder: str,
                            mutation_dataset_names: List[str]):
    """Return dictionary of condensed DS per mutation.
        - mutation_dataset_names should have mutation as dataset is names in h5py"""
    results = {}

    filepath = generate_filepath(gene_name, tss, ds_noncondensed_folder)

    if not os.path.exists(filepath):
        return None
    try:
        data = h5py.File(filepath, "r")
    except:
        return None

    full_tss = data[('full_tss')][0]
    results['full_tss'] = condense_gene_fn(full_tss)

    for mut in mutation_dataset_names:
        try:
            mut_DS = data[(mut)][0]
        except:
            continue
        results[mut] = condense_gene_fn(mut_DS)

    return results


def condense_and_save(condense_gene_fn, filenames: List[str], location_noncondesed: str,
                      location_save: str):
    """Function to create condesed genes files with the same structure as non-condensed."""

    if not os.path.exists(location_save):
        print('Create destination folder')
        os.makedirs(location_save)
    
    count_total_condensed = 0
    for i, file_name in enumerate(filenames):

        if i % 100 == 99:
            print('Genes:', i)

        filepath = location_noncondesed + file_name
        try:
            data = h5py.File(filepath, "r")
        except:
            continue

        if os.path.exists(location_save + 'condensed_'+file_name):
            try: 
                f = h5py.File(location_save + 'condensed_'+file_name, "r+")
            except:
                print(file_name)
                continue
        else:
            f = h5py.File(location_save + 'condensed_'+file_name, "w")
            condense_base = condense_gene_fn(data[('full_tss')][0])
            count_total_condensed += 1
            f.create_dataset("full_tss", data=condense_base,
                             compression='gzip')
            f.attrs['gene_name'] = data.attrs['gene_name']
            try:
                f.attrs['tss'] = data.attrs['tss']
            except:
                print('ERROR with TSS', file_name)
                f.attrs['tss'] = file_name.str.split('_')[1]
            f.attrs['chrom'] = data.attrs['chrom']
            f.attrs['strand'] = data.attrs['strand']
            try:
                f.attrs['cage_used'] = data.attrs['cage_used']
            except:
                print('Cage used not available', file_name)
        
        try:
            datasets = list(data.keys())
        except RuntimeError:
            print('runtime error reading data keys for', filepath)
            f.close()
            data.close()
            continue


        for dataset_name in datasets:
            if count_total_condensed % 1000 == 999:
                print('Condensed:', count_total_condensed)
                print('Genes:', i, 'Total needed:', len(filenames))

            if dataset_name in list(f.keys()):
                # already exists
                continue
            condense_mut = condense_gene_fn(data[(dataset_name)][0])
            f.create_dataset(dataset_name, data=condense_mut,
                             compression='gzip')
            count_total_condensed += 1

        f.close()
        data.close()


def condense_genes_from_file(condense_gene_fn, gene_tss_filename: str, ds_condensed_folder: str,
                             save: bool = False, save_loc_name: str = ''):
    """Condense genes into dictionary format, no mutations. Used for training."""
    condensed_dict = {}
    genes_locs = pd.read_csv(gene_tss_filename)

    for i, row in genes_locs.iterrows():
        filepath = ds_condensed_folder + 'condensed_' + row.gene_name+'_'+str(int(row.tss))+'.hd5f'

        if not os.path.exists(filepath):
            print('Gene not found: ', filepath)
            continue

        try:
            data = h5py.File(filepath, "r")
            data = data[('full_tss')][:]
        except:
            print('Error loading file: ', filepath)
            continue

        if data is None:
            continue

        condensed_dict[row.gene_name] = data

        if i % 5000 == 0:
            print(i)

    # add metadata and save
    if save:
        pickle.dump(condensed_dict, open(save_loc_name, 'wb'))

    return condensed_dict


def collect_condense_genes_from_csv(condensed_location: str, gene_tss_pd):
    """Collect already pre-condensed files into dictionary for training.
        Inputs: 
            condensed_location - location of the condensed hd5f files 
            gene_tss_pd - DataFrame with genes [gene_name] and tss [tss] columns
        Outputs:
            Dictionary: gene-name - condensed matrix
            Int: Number of genes that did not have a corresponding file
    """
    condensed_dict = {}
    n_genes_not_found = 0

    for i, row in gene_tss_pd.iterrows():
        if i%1000==0:
            print('Collected ', i)
        filepath = condensed_location + 'condensed_' + row.gene_name+'_'+str(int(row.tss))+'.hd5f'

        if not os.path.exists(filepath):
            n_genes_not_found += 1
            continue
        try:
            data = h5py.File(filepath, "r")
            data = data[('full_tss')][:]
        except:
            print('Error loading file: ', filepath)
            continue

        if data is None:
            continue

        condensed_dict[row.gene_name] = data

    return condensed_dict, n_genes_not_found
