"""General data utils file"""
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.3)
sns.set_style("whitegrid")
from sklearn.metrics import roc_curve, auc
from scipy.stats import rankdata


def get_files_in_folder(folder: str) -> bool:
    """Function returns files present in the folder."""
    return os.listdir(folder)


def generate_filepath(gene_name: str, tss: int, folder: str) -> str:
    return folder + gene_name+'_'+str(int(tss))+'.hd5f'


def check_gene_TSS_exists(gene_name: str, tss: int, folder: str) -> bool:
    """Check if gene with the specified TSS had been computed."""
    filepath = generate_filepath(gene_name, tss, folder)
    return os.path.exists(filepath)

def convert_dict_df(dict_formatted_scores, col_name):
    """Convert dict into pandas DF used for merging"""
    formatted_scores = pd.DataFrame.from_dict(dict_formatted_scores, orient='index')
    formatted_scores.columns = [col_name]
    formatted_scores['SNP'] = formatted_scores.index.values
    return formatted_scores

def process_ref_alt(ref, alt, columns_models):
    """Compute score effects from ref and alt"""
    ref_exp = np.exp2(ref[columns_models])
    alt_exp = np.exp2(alt[columns_models])
    fraction = (alt_exp+1)/(ref_exp+1)
    fraction_log = np.log2(fraction) 
    scaled = ref.copy()
    scaled[columns_models] = fraction_log
    return scaled

def collect_truth_pred_df(truth_preds_stratified):
    """ Collect one truth/preds dataframe from the folds """
    folds = list(truth_preds_stratified.keys())
    groups = truth_preds_stratified[0]['groups']
    genes0 = truth_preds_stratified[0]['genes']
    truth_expr_raw_logs = pd.DataFrame(
        truth_preds_stratified[0]['truth'], columns=groups, index=genes0)
    pred_expr_raw_logs = pd.DataFrame(
        truth_preds_stratified[0]['preds'], columns=groups, index=genes0)

    for i, fold in enumerate(folds):
        if i == 0 or fold == 'genes':
            continue  # already added
        if not np.array_equal(groups, truth_preds_stratified[fold]['groups']):
            raise ValueError('different group permutation')
        truth_temp = pd.DataFrame(truth_preds_stratified[fold]['truth'],
                                  columns=groups, index=truth_preds_stratified[fold]['genes'])
        pred_temp = pd.DataFrame(truth_preds_stratified[fold]['preds'],
                                 columns=groups, index=truth_preds_stratified[fold]['genes'])
        truth_expr_raw_logs = truth_expr_raw_logs.append(truth_temp)
        pred_expr_raw_logs = pred_expr_raw_logs.append(pred_temp)

    return truth_expr_raw_logs, pred_expr_raw_logs


def grouped_obs_mean(adata, group_key, layer=None):
    """Collect means for the clusters, where group_key is the cluster assignment.
        adata: AnnData objecy w/ group_key observation
        Source reference: https://github.com/theislab/scanpy/issues/181#issuecomment-534867254
        """
    if layer is not None:
        def getX(x): return x.layers[layer]
    else:
        def getX(x): return x.X

    grouped = adata.obs.groupby(group_key)
    out = pd.DataFrame(
        np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
        columns=list(grouped.groups.keys()),
        index=adata.var_names
    )

    for group, idx in grouped.indices.items():
        X = getX(adata[idx])
        out[group] = np.ravel(X.mean(axis=0, dtype=np.float64))
    return out


def count_ncells_celltype(adata, group_key):
    """ Count how many cells there are in each cell cluster"""
    n_groups = pd.DataFrame(adata.obs.groupby([group_key]).apply(len))
    n_groups.columns = ['n_counts']
    n_groups = n_groups.sort_values(by='n_counts', ascending=False)
    return n_groups

