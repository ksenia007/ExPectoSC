"""Code to train and run models."""
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

#import xgboost as xgb
#import lightgbm as lgb
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from scipy.stats import rankdata

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from scipy.stats.stats import pearsonr
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

import re
import h5py

from data_utils import *


from DS_for_variants import *


def collect_dataX_from_condense(genes, condensed_dict, condensed_length=20020):
    """Given a list of required genes & condensed dicitonary, create dataX numpy matrix.

    genes: dataframe containing gene_name as column
    condensed_dict: Dict of form [gene_name: np.array (20020)], where np array is the condensed DS

    Note: if gene in 'genes' is not present in condensed_dict raises Error.
    """
    dataX = np.zeros((genes.shape[0], condensed_length))
    counter = 0
    for _, row in genes.iterrows():
        try:
            dataX[counter, :] = condensed_dict[row.gene_name]
        except:
            print('missing', row.gene_name)
            raise ValueError(
                'Requested gene does not exist: incompatible gene tss file and condense')
        counter += 1

    return dataX


def collect_dataY(genes, expressions_df_all):
    expr = np.zeros((genes.shape[0], expressions_df_all.shape[0]))
    print(expr.shape)
    counter = 0
    for _, row in genes.iterrows():
        expr[counter, :] = expressions_df_all[row.gene_name].values
        counter += 1
    group_names = expressions_df_all.index.values
    return expr, group_names


def collect_dataXY(genes, condensed_dict, expressions_df_all, condensed_length=20020):
    dataX = collect_dataX_from_condense(
        genes, condensed_dict, condensed_length)
    dataY, metadata = collect_dataY(genes, expressions_df_all)
    return dataX, dataY, metadata


def get_weights(expressions, percetile_cutoff_top=70, percetile_cutoff_bottom=10):
    """Assign genes weights based on variance"""
    expressions_copy = expressions.copy()
    # normalize with mean
    expressions_copy = expressions_copy-expressions_copy.mean()
    weights = np.zeros(expressions.shape[0])
    weights = weights+1
    top_percentile_value = np.percentile(
        expressions_copy.std(1), percetile_cutoff_top)
    weights[expressions_copy.std(1) >= top_percentile_value] += 1
    bottom_percentile_value = np.percentile(
        expressions_copy.std(1), percetile_cutoff_bottom)
    weights[expressions_copy.std(1) < bottom_percentile_value] -= 0.5
    return weights


def save_train_val_matrices(genes_tss_file_loc, condensed_values_file_loc,
                            expressions_all_file_loc, save_loc, condensed_length=20020):
    """ Read genes and split into train and test as done before training"""
    genes_tss = pd.read_csv(genes_tss_file_loc)
    genes_train, genes_val = train_test_split(
        genes_tss, test_size=0.1, random_state=3)

    condensed_dict = pickle.load(open(condensed_values_file_loc, 'rb'))
    expressions_df_all = pd.read_csv(expressions_all_file_loc, index_col=0)

    dataX, dataY, group_names = collect_dataXY(
        genes_train, condensed_dict, expressions_df_all, condensed_length)
    dataX_val, dataY_val, _ = collect_dataXY(
        genes_val, condensed_dict, expressions_df_all, condensed_length)

    train_data = {'X': dataX, 'Y': dataY, 'groups': group_names}
    val_data = {'X': dataX_val, 'Y': dataY_val, 'groups': group_names}
    pickle.dump(train_data, open(save_loc+'train.pkl', 'wb'))
    pickle.dump(val_data, open(save_loc+'valid.pkl', 'wb'))


def eval_model(model, dataX, dataY_one_group, dataX_val, dataY_one_group_val,
               save_preds=False, save_preds_loc_base='', group_name='', run_train_data = True):
    """Evaluate predictions on the train and validation. Return results + pred on validation"""
    results = {}
    if run_train_data:
        preds = model.predict(dataX)
        c, p = spearmanr(dataY_one_group, preds)
        results['train'] = np.median(c)

    preds = model.predict(dataX_val)
    c, p = spearmanr(dataY_one_group_val, preds)
    results['valid'] = np.median(c)

    if save_preds:
        if not os.path.exists(save_preds_loc_base):
            os.makedirs(save_preds_loc_base)
        pickle.dump(preds, open(save_preds_loc_base +
                                re.sub(r'[^\w]', '_', group_name+'_preds')+'.p', 'wb'))
        pickle.dump(dataY_one_group_val, open(
            save_preds_loc_base+re.sub(r'[^\w]', '_', group_name)+'.p', 'wb'))

    results['rmse'] = np.sqrt(mean_squared_error(dataY_one_group_val, preds))
    results['r2'] = r2_score(dataY_one_group_val, preds)
    results['expl_variance'] = explained_variance_score(
        dataY_one_group_val, preds)

    c, p = pearsonr(dataY_one_group_val, preds)
    results['pearson'] = np.median(c)

    return results, preds


def train_group_linear(dataX, dataY_one_group, dataX_val, dataY_one_group_val,
                       group_name, params, valid=True,
                       save_preds=False, save_preds_loc_base='', save_models=False,
                       save_models_loc='', weighted=False, weights=[], eval=True, 
                       dataX_test=None, dataY_test=None):
    """ 
    Train a Linear model for a given group w/ provided parameters, record eval values
    """
    print('Cell type: ', group_name)

    print(dataX.shape)
    print(dataY_one_group.shape)

    bst = linear_model.Ridge(**params)
    if weighted:
        bst.fit(dataX, dataY_one_group, sample_weight=weights)
    else:
        bst.fit(dataX, dataY_one_group)
    if save_models:
        pickle.dump(bst, open(save_models_loc +
                              re.sub(r'[^\w]', '_', group_name)+'.txt', 'wb'))


    if not eval:
        # full train
        return None, None
    
    if eval:
        if valid:
            results, preds = eval_model(bst, dataX, dataY_one_group, dataX_val,
                                    dataY_one_group_val, save_preds, 
                                    group_name = group_name,
                                    save_preds_loc_base = save_preds_loc_base)
            print('Validation median Spearman w/ log transform', results['valid'])
            return results, preds
        results_test, preds_test = eval_model(bst, dataX, dataY_one_group, dataX_test,
                                    dataY_test, save_preds, 
                                    save_preds_loc_base = save_preds_loc_base+'test/', 
                                    group_name = group_name, run_train_data=False)
        print('Test Spearman w/ log transform', results_test['valid'])
        print('***')

    return results_test, preds_test



def cross_val_predict_all_groups(function_train, genes_tss, condensed_dict, expressions_df_all,
                                 train_params, pseudocount, random_state=3,
                                 save_preds=True, save_preds_locs='', use_log=True,
                                 weighted=False, condensed_length=20020, folds=5, subset_size=False, fraction_drop=1):
    """Cross validation to get the predictions for the whole dataset"""
    print('seed:', random_state)
    if save_preds and not os.path.exists(save_preds_locs):
        os.makedirs(save_preds_locs)

    genes_tss = genes_tss[genes_tss.chrom != 'X']
    genes_tss = genes_tss[genes_tss.chrom != 'Y']

    genes_tss_test = genes_tss[(genes_tss.chrom == '8') | (genes_tss.chrom == '9')]
    genes_tss_test = check_existence(genes_tss_test, condensed_dict, expressions_df_all)

    genes_tss = genes_tss[(genes_tss.chrom != '8') & (genes_tss.chrom != '9')]
    genes_tss = check_existence(genes_tss, condensed_dict, expressions_df_all)
    

    if subset_size:
        # only use part of the dataset to check for the effect data size has
        genes_tss, _ = train_test_split(
            genes_tss, test_size=fraction_drop, random_state=random_state)

    dataX, dataY, group_names = collect_dataXY(
        genes_tss, condensed_dict, expressions_df_all, condensed_length)
    dataX_test, dataY_test, group_names_test = collect_dataXY(
        genes_tss_test, condensed_dict, expressions_df_all, condensed_length)

    print('*Compiled one large datset*')
    print('Train: ', dataX.shape)
    print('Test: ', dataX_test.shape)

    # Instantiate the CV and RF
    folds_splits = KFold(n_splits=folds, shuffle=True,
                         random_state=random_state)

    fold_curr = 0
    results = {}
    truth_preds = {}

    for train_idx, test_idx in folds_splits.split(dataY):
        print('Starting fold:', fold_curr)

        results[fold_curr] = {}
        dataX_fold = dataX[train_idx, :]
        dataY_fold = dataY[train_idx, :]
        dataX_fold_test = dataX[test_idx, :]
        dataY_fold_test = dataY[test_idx, :]

        print(dataX_fold.shape)
        print(dataY_fold.shape)
        print(dataY_fold_test.shape)

        preds_fold = np.zeros((dataY_fold_test.shape[0], len(group_names)))

        if weighted:
            weights = get_weights(dataY_fold)
        else:
            weights = []

        for i, group_name in enumerate(group_names):
            # separate model for each group
            if use_log:
                dataY_one_group = np.log2(dataY_fold[:, i]+pseudocount)
                dataY_fold_one_group_test = np.log2(
                    dataY_fold_test[:, i]+pseudocount)
                dataY_one_group_test = np.log2(
                    dataY_test[:, i]+pseudocount)
            else:
                dataY_one_group = dataY_fold[:, i]
                dataY_fold_one_group_test = dataY_fold_test[:, i]
                dataY_one_group_test = dataY_test[:, i]

            group_results, preds = function_train(dataX_fold, dataY_one_group, dataX_fold_test, dataY_fold_one_group_test,
                                                  group_name, train_params,
                                                  save_preds=save_preds, save_preds_loc_base=save_preds_locs,
                                                  save_models=False,
                                                  weighted=weighted, weights=weights,
                                                  dataX_test = dataX_test, dataY_test = dataY_one_group_test)

            results[fold_curr][group_name] = group_results
            preds_fold[:, i] = preds

        # dataframe of results
        if use_log:
            dataY_fold_test_np = np.log2(dataY_fold_test+pseudocount)
        else:
            dataY_fold_test_np = dataY_fold_test

        truth_preds[fold_curr] = {'truth': dataY_fold_test_np, 'preds': preds_fold,
                                  'groups': group_names,
                                  'genes': genes_tss.gene_name.values[test_idx]}

        fold_curr += 1

    truth_preds['genes'] = genes_tss.gene_name
    print('Done with all folds')
    return truth_preds, results


def evals_train_test(function_train, genes_tss, condensed_dict, expressions_df_all,
                                 train_params, pseudocount, random_state=3,
                                 save_preds=True, save_preds_locs='', use_log=True,
                                 weighted=False, condensed_length=20020, folds=5, subset_size=False, fraction_drop=1):
    """Cross validation to get the predictions for the whole dataset"""
    print('seed:', random_state)
    if save_preds and not os.path.exists(save_preds_locs):
        os.makedirs(save_preds_locs)

    genes_tss = genes_tss[genes_tss.chrom != 'X']
    genes_tss = genes_tss[genes_tss.chrom != 'Y']

    genes_tss_test = genes_tss[(genes_tss.chrom == '8') | (genes_tss.chrom == '9')]
    genes_tss_test = check_existence(genes_tss_test, condensed_dict, expressions_df_all)

    genes_tss = genes_tss[(genes_tss.chrom != '8') & (genes_tss.chrom != '9')]
    genes_tss = check_existence(genes_tss, condensed_dict, expressions_df_all)
    

    if subset_size:
        # only use part of the dataset to check for the effect data size has
        genes_tss, _ = train_test_split(
            genes_tss, test_size=fraction_drop, random_state=random_state)

    dataX, dataY, group_names = collect_dataXY(
        genes_tss, condensed_dict, expressions_df_all, condensed_length)
    dataX_test, dataY_test, group_names_test = collect_dataXY(
        genes_tss_test, condensed_dict, expressions_df_all, condensed_length)

    print('*Compiled one large datset*')
    print('Train: ', dataX.shape)
    print('Test: ', dataX_test.shape)
   
    results = {}
    truth_preds = {}

    print(dataX.shape)
    print(dataY.shape)

    if weighted:
        weights = get_weights(dataY)
    else:
        weights = []

    for i, group_name in enumerate(group_names):
        # separate model for each group
        if use_log:
            dataY_one_group = np.log2(dataY[:, i]+pseudocount)
            dataY_one_group_test = np.log2(
                dataY_test[:, i]+pseudocount)
        else:
            dataY_one_group = dataY[:, i]
            dataY_one_group_test = dataY_test[:, i]

        group_results, _ = function_train(dataX, dataY_one_group, [], [],
                                                group_name, train_params,
                                                save_preds=save_preds, save_preds_loc_base=save_preds_locs,
                                                save_models=False, valid=False,
                                                weighted=weighted, weights=weights,
                                                dataX_test = dataX_test, dataY_test = dataY_one_group_test)

        results[group_name] = group_results

        # dataframe of results
        if use_log:
            dataY_test_np = np.log2(dataY_test+pseudocount)
        else:
            dataY_test_np = dataY_test

        
    print('Done with test')
    return results



def check_existence(genes_tss, condensed_dict, expressions_df_all):
    exist = pd.DataFrame({'gene_name': list(condensed_dict.keys())})
    genes_tss = genes_tss.merge(exist, on='gene_name', how='inner')
    exist_expr = pd.DataFrame({'gene_name': expressions_df_all.columns.values})
    genes_tss = genes_tss.merge(exist_expr, on='gene_name', how='inner')
    return genes_tss


def train_all_groups(function_train, genes_tss, condensed_dict, expressions_df_all,
                     train_params, pseudocount, save_models=False, save_models_loc='',
                     save_preds=True, save_preds_locs='', use_log=True,
                     log_odds_norm=False, weighted=False, condensed_length=20020, subset_size=False, fraction_drop=1,
                     fraction_valid=0.1, run_eval=True):
    """Train all groups given the function_train that trains one model per group"""
    if save_preds and not os.path.exists(save_preds_locs):
        os.makedirs(save_preds_locs)
    if save_models and not os.path.exists(save_models_loc):
        os.makedirs(save_models_loc)

    genes_tss = genes_tss[genes_tss.chrom != 'X']
    genes_tss = genes_tss[genes_tss.chrom != 'Y']

    try:
        genes_tss.index = genes_tss.gene_name.values
    except:
        print('could not assign gene name as index')

    genes_tss = check_existence(genes_tss, condensed_dict, expressions_df_all)

    if subset_size:
        genes_tss, _ = train_test_split(
            genes_tss, test_size=fraction_drop, random_state=3)

    if run_eval:
        genes_train, genes_val = train_test_split(
            genes_tss, test_size=fraction_valid, random_state=3)
    else:
        genes_train = genes_tss
        genes_val = pd.DataFrame({'gene_name': []})

    dataX, dataY, group_names = collect_dataXY(
        genes_train, condensed_dict, expressions_df_all, condensed_length)
    dataX_val, dataY_val, _ = collect_dataXY(
        genes_val, condensed_dict, expressions_df_all, condensed_length)

    if weighted:
        weights = get_weights(dataY)
    else:
        weights = []

    # add macro-average dataset
    mean_expr = np.transpose(np.array([dataY.mean(1)]))
    print(mean_expr.shape)
    dataY = np.concatenate((dataY, mean_expr), 1)
    group_names = np.concatenate((group_names, ['macro_mean']))
    mean_expr_val = np.transpose(np.array([dataY_val.mean(1)]))
    dataY_val = np.concatenate((dataY_val, mean_expr_val), 1)

    print('*Compiled datasets*')
    print('Train size: ', dataX.shape, ' Validation: ', dataX_val.shape)

    preds_all = np.zeros((dataX_val.shape[0], len(group_names)))

    results = {}

    # train each of the groups separately
    for i, group_name in enumerate(group_names):

        if log_odds_norm:
            dataY_one_group = np.log2(
                dataY[:, i]+pseudocount) - np.log2(np.mean(dataY[:, i])+pseudocount)
            dataY_one_group_val = np.log2(
                dataY_val[:, i]+pseudocount) - np.log2(np.mean(dataY[:, i])+pseudocount)
        elif use_log:
            dataY_one_group = np.log2(dataY[:, i]+pseudocount)
            dataY_one_group_val = np.log2(dataY_val[:, i]+pseudocount)
        else:
            dataY_one_group = dataY[:, i]
            dataY_one_group_val = dataY_val[:, i]

        group_results, preds = function_train(dataX, dataY_one_group, dataX_val, dataY_one_group_val,
                                              group_name, train_params,
                                              save_preds=save_preds, save_preds_loc_base=save_preds_locs,
                                              save_models=save_models, save_models_loc=save_models_loc,
                                              weighted=weighted, weights=weights, eval=run_eval)
        results[group_name] = group_results

        preds_all[:, i] = preds

    # save csv files of results
    if save_preds:
        preds_df = pd.DataFrame(preds_all, columns=group_names,
                                index=genes_val.gene_name.values)
        if use_log:
            truth_df = pd.DataFrame(np.log2(dataY_val+pseudocount),
                                    columns=group_names, index=genes_val.gene_name.values)
        else:
            truth_df = pd.DataFrame(dataY_val,
                                    columns=group_names, index=genes_val.gene_name.values)
    else:
        preds_df = []
        truth_df = []

    return results, preds_df, truth_df


def model_inference(model, condensed_DS):
    """Return prediction for the given model."""
    return model.predict(condensed_DS[:].reshape(1, 20020))[0]


def predict_diff_models_all_gene(models_locations, gene_info, file_location, full_gene_name=''):
    """Predict mutation-full_tss effects for the given model, 
    all present mutations, all genes in list."""

    # load file
    if full_gene_name:
        filename = file_location+full_gene_name
    else:
        filename = file_location+'condensed_'+gene_info.gene_name + \
            '_'+str(int(gene_info.tss))+'.hd5f'
    try:
        file = h5py.File(filename, "r")
    except:
        raise ValueError('File not found', filename)

    # got models info
    models_all = os.listdir(models_locations)
    models_names = [i.split('.')[0] for i in models_all]
    print(models_names)

    results = pd.DataFrame(np.zeros((len(list(file.keys())), len(models_names))),
                           columns=models_names, index=list(file.keys()))

    # loop over models
    problems = []
    for i, model_name in enumerate(models_names):
        print(model_name)
        preds = {}
        if i == 0:
            diff_ds = {}
        model_loc = models_locations+model_name+'.txt'
        model = pickle.load(open(model_loc, 'rb'))
        full_tss = file['full_tss']
        # get reference as it does not change
        reference_pred = model_inference(model, full_tss)
        preds['full_tss'] = reference_pred
        for m_i, mut in enumerate(list(file.keys())):
            if mut[:3] != 'mut':
                continue
            try:
                mutation_cond = file[mut]
            except:
                problems.append(mut)
                continue
            if i == 0:
                diff_ds[mut] = np.linalg.norm(file[mut][:]-full_tss[:])
            effect_mut = model_inference(model, mutation_cond)
            preds[mut] = effect_mut

        model_res = pd.DataFrame.from_dict(
            preds, orient='index', columns=[model_name])
        results.update(model_res)

    print('Problems:', problems)
    print('Mutations checked:', results.index)

    return results, problems, reference_pred, diff_ds


def predict_diff_model(model, variants_df, file_location,
                       use_diff=False, predict_A1A2=False,
                       tss_col_name='cl_chromStart', A1='A1', A2='A2', compute_delta=False):
    """Predict A2 & A1 effects for the given model."""

    diff_preds = {}
    diff_preds_A2 = {}

    delta_ds = {}
    problems = []
    running_gene_name = ''
    for i, variant in variants_df.iterrows():
        # if i % 100000 == 0:
        #     print(i)
        #     print('problems', len(problems))
        if running_gene_name != variant.gene_name:
            running_gene_name = variant.gene_name
            filename = file_location+'condensed_'+running_gene_name + \
                '_'+str(int(variant[tss_col_name]))+'.hd5f'
            if not os.path.exists(filename):
                print(filename)
                problems.append(variant)
                continue
            try:
                file = h5py.File(filename, "r")
            except:
                raise ValueError('File not found', filename)

            reference_pred = model_inference(model, file['full_tss'])

        # check if A1 or A2 are references
        if check_if_reference(variant[A1], variant.snp_chromStart, 'chr'+str(variant.chrom)):
            effect_1 = reference_pred
            if compute_delta:
                ds_A1 = file['full_tss'][:]
        else:
            if 'A1' == 'ref':
                raise Exception("A1 should be ref but it is not")
            effect_1_name = "mut_" + \
                str(variant.snp_chromStart)+'_'+str(variant[A1])
            try:
                cond = file[effect_1_name]
            except:
                problems.append(variant)
                continue
            effect_1 = model_inference(model, cond)

            if compute_delta:
                ds_A1 = cond[:]

        if check_if_reference(variant[A2], variant.snp_chromStart, 'chr'+str(variant.chrom)):
            effect_2 = reference_pred
            if compute_delta:
                ds_A2 = file['full_tss'][:]
        else:
            effect_2_name = "mut_" + \
                str(variant.snp_chromStart)+'_'+str(variant[A2])
            try:
                cond = file[effect_2_name]
            except:
                problems.append(variant)
                continue
            effect_2 = model_inference(model, cond)

            if compute_delta:
                ds_A2 = cond[:]

        if use_diff:
            diff_preds[variant.SNP] = effect_2-effect_1
        elif predict_A1A2:
            diff_preds[variant.SNP] = effect_1
            diff_preds_A2[variant.SNP] = effect_2
        else:
            diff_preds[variant.SNP] = effect_2/effect_1

        if compute_delta:
            delta_ds[variant.SNP] = np.linalg.norm(ds_A2-ds_A1)

    return diff_preds, problems, delta_ds, diff_preds_A2


def collect_predictions_models(models_base, variants, base_save_loc, compressed_encodings_loc, logging,
                               save_filename_base_name, models_inside_folder='models_full_train',
                               tss_col_name='tss',
                               A1='ref', A2='alt'):
    """ 
    Predict effects for variants provided for all models in the folder (one organ). Save ref, alt, deltaDS and final scores. 
    """
    models_loc = models_base+'/'+models_inside_folder+'/'
    models_all = os.listdir(models_loc)
    models_names = [i.split('.')[0] for i in models_all]
    print(models_names)
    logging.info(models_names)
    print('Read all files')

    use_diff = False
    predict_A1A2 = True

    for i, model_name in enumerate(models_names):
        print(model_name)
        model_loc = models_loc+model_name+'.txt'
        model = pickle.load(open(model_loc, 'rb'))

        if i == 0:
            dict_preds_REF, problems, delta_ds, dict_preds_ALT = predict_diff_model(model, variants, file_location=compressed_encodings_loc,
                                                                                    compute_delta=True, use_diff=use_diff, predict_A1A2=predict_A1A2,
                                                                                    tss_col_name=tss_col_name, A1=A1, A2=A2)
        else:
            dict_preds_REF, _, _, dict_preds_ALT = predict_diff_model(model, variants, file_location=compressed_encodings_loc,
                                                                      compute_delta=False, use_diff=use_diff, predict_A1A2=predict_A1A2,
                                                                      tss_col_name=tss_col_name, A1=A1, A2=A2)
        # convert into pandas df
        preds_REF = convert_dict_df(dict_preds_REF, model_name)
        preds_ALT = convert_dict_df(dict_preds_ALT, model_name)
        if i == 0:
            results_ref = preds_REF
            results_alt = preds_ALT
        else:
            results_ref = results_ref.merge(
                preds_REF, on='SNP', how='left').fillna(0)
            results_alt = results_alt.merge(
                preds_ALT, on='SNP', how='left').fillna(0)

        if i == 0:
            delta_ds = convert_dict_df(delta_ds, 'delta_ds')

    logging.info('There were problems with {} variants'.format(len(problems)))

    results_ref.to_csv(base_save_loc+'/intermediate/' +
                       save_filename_base_name+'_REF_A1.csv')
    results_alt.to_csv(base_save_loc+'/intermediate/' +
                       save_filename_base_name+'_ALT_A2.csv')
    delta_ds.to_csv(base_save_loc+'/intermediate/' +
                    save_filename_base_name+'_deltaDS.csv')

    # create a file with scores
    scores = process_ref_alt(ref=results_ref, alt=results_alt, columns_models=[
                             i for i in results_ref.columns if i != 'SNP'])
    scores.index = scores.SNP
    scores = scores.drop('SNP', axis=1)
    scores.to_csv(base_save_loc+'/'+save_filename_base_name+'.csv')

def find_immune(column_names):
    """Find which columns are not immune cells"""
    immune_cell_name_fragments = ['bcell', 'inflammatorymacs', 'nklike', 'abtcell', 'gdtcell',
                              'cd4tcell', 'cd8tcell', 'monocytederived', 'mnpcdendriticcell',
                              'nktcell', 'nkcell', 'b_cells', 'dc_', 'macrophage', 'monocyte',
                              'nk_', 't_cd4', 't_cd8', 't_r', 'mast_', 'lymphoid', 'nk_lung'
                             ]
    immune = []
    nonimmune = []

    for cell_type in column_names:
        immune_found=False
        for immune_fragm in immune_cell_name_fragments:
            if immune_fragm in re.sub(r'[^\w]', '_', cell_type).lower() and not immune_found:
                immune.append(cell_type)
                immune_found = True
        if not immune_found:    
            nonimmune.append(cell_type)
    return nonimmune

def find_ambiguous_groups(column_names):
    """Find which columns are not ambigious" cells"""
    drop_groups = ['unannotated', 'unspecified', 'NotAssigned', 'Unclassified', 'doublets', '_mean']
    ambiguous = []
    normal = []

    for cell_type in column_names:
        amb_found=False
        for ambig_fragm in drop_groups:
            if ambig_fragm in re.sub(r'[^\w]', '_', cell_type).lower() and not amb_found:
                ambiguous.append(cell_type)
                amb_found = True
        if not amb_found:    
            normal.append(cell_type)
    return normal

def collect_ldscore_chrom(path_ldscore_SNPs, base_path_effects_computed,
                          model_subfolders, chrom, save_location, 
                          remove_immune=True, remove_ambiguous=True):
    """Collect all ldscore effect from all models and keep the order correct"""

    if not os.path.exists(save_location):
        os.makedirs(save_location)
    
    if os.path.exists(save_location+'chrom.'+str(chrom)+'.csv'):
        print('Existed', save_location+'chrom.'+str(chrom)+'.csv')
        return

    try:
        file_ldscore_order = pd.read_csv(
            path_ldscore_SNPs+'baselineLD.'+str(chrom)+'.annot', sep='\t')
    except:
        raise ValueError('Chrom {} does not exist'.format(chrom))

    results_chrom_final_order = file_ldscore_order[['CHR', 'SNP', 'BP']]
    results_chrom_final_order.index = results_chrom_final_order.SNP

    print('final size', results_chrom_final_order.shape)

    for i, group in enumerate(model_subfolders):
        print('Working on', group)
        variant_effects = pd.read_csv(
            base_path_effects_computed+'/'+group+'/ldsc/chrom.'+str(chrom)+'.csv', index_col=0)
        variant_effects = np.abs(variant_effects)
        
        variant_effects.columns = variant_effects.columns+'_'+group

        # drop immune cell types
        if remove_immune:
            non_immune = find_immune(variant_effects.columns)
            variant_effects = variant_effects[non_immune]
        if remove_ambiguous:
            good_groups = find_ambiguous_groups(variant_effects.columns)
            variant_effects = variant_effects[good_groups]

        
        if i == 0:
            variant_effects_res = variant_effects
        else:
            variant_effects_res = pd.concat(
                [variant_effects_res, variant_effects], axis=1)
        print(variant_effects_res.shape)

    results_chrom_final_order = results_chrom_final_order.merge(variant_effects_res,
                                                                left_index=True, right_index=True, 
                                                                how='left').fillna(0)

    print(results_chrom_final_order.shape)
    results_chrom_final_order.to_csv(save_location+'chrom.'+str(chrom)+'.csv', index=False)
    return results_chrom_final_order

