"""
Main file for training models from the new dataset. 

Options:
 - CV training to check performance
 - Full train for prediction
 - Provide train parameters or estimate alpha for training first
"""
import argparse
import logging
import sys
import pickle
import pandas as pd
from main_train import *
from condense_data import *
from data_utils import *


def binarize_truth_on_off(truth_expr, genes_info, percentile_cutoff=30):
    """Get binarized truth/preds and on/off genes for given percentile"""
    truth_expr = truth_expr.transpose()
    binarized_high_truth = truth_expr > np.percentile(truth_expr, 100-percentile_cutoff, axis=0)
    binarized_low_truth = truth_expr <= np.percentile(truth_expr, percentile_cutoff, axis=0)
    on_off_genes = np.logical_and(binarized_high_truth.sum(1)>0, binarized_low_truth.sum(1)>0)
    
    genes_keep = list(truth_expr.index.values[on_off_genes])
    genes_keep_test = list(genes_info[(genes_info.chrom == '8') | (genes_info.chrom == '9')].gene_name.values)
    print(genes_keep)
    print(genes_keep_test)

    genes_keep_test.extend(genes_keep)
    
    return list(set(genes_keep_test))


def get_genes_tss_pd(genes_list, gene_tss_reference: str, merge_on='gene_name'):
    """ Get TSS locations for the provided genes.
    Inputs:
        genes_list: List[str], list of gene names or other gene IDs
        gene_tss_reference: .csv with  "merge_on" gene ID columns 
    Return:
        genes_TSS: merged genes_list and gene_tss_reference
        lost_genes_N: number of rows that were lost in merge
    """
    gene_TSS_ref = pd.read_csv(gene_tss_reference)
    genes_TSS = pd.DataFrame({merge_on: genes_list})
    genes_TSS = genes_TSS.merge(gene_TSS_ref, on=merge_on, how='inner')
    # number of genes for which TSS not found
    lost_genes_N = len(genes_list)-genes_TSS.shape[0]
    return genes_TSS, lost_genes_N


def run_evaluations(truth_preds, args, save_evals_loc):
    # save eval results

    if not os.path.exists(save_evals_loc):
        os.makedirs(save_evals_loc)

    # convert to one large dataframe
    truth_expr, pred_expr = collect_truth_pred_df(truth_preds)

    # save truth and pred tables 
    truth_expr.to_csv(save_evals_loc+'ground_truth_expression.csv')
    pred_expr.to_csv(save_evals_loc+'predicted_expression.csv')

    # spearman correlation per groups on all folds
    spearman_scores_per_group = {}
    for gr in truth_expr.columns.values:
        spearman_scores_per_group[gr] = spearmanr(
            truth_expr[gr], pred_expr[gr])[0]
    spearman_scores_per_group = pd.DataFrame.from_dict(
        spearman_scores_per_group, orient='index', columns=['Spearman'])
    spearman_scores_per_group = spearman_scores_per_group.sort_values(
        by='Spearman', ascending=False)
    spearman_scores_per_group.to_csv(save_evals_loc+'spearman_per_group.csv')

    return spearman_scores_per_group, truth_expr, pred_expr


parser = argparse.ArgumentParser()

parser.add_argument('--expressions', default=None, type=str,
                    help="Path to .csv file with expression. Each column is a group, and each row is a gene. Gene ID is the index")
parser.add_argument('--save-location', default=None, type=str,
                    help="Path to folder to save results")
parser.add_argument('--alpha', default=0.1, type=float,
                    help="Alpha for training. Should fine-tune if possible.")
parser.add_argument('--find-alpha', default=False, action='store_true',
                    help="If used, alpha value is ignored and alpha is approximated based on the dataset")
parser.add_argument('--cv', default=False, action='store_true',
                    help="If listed, cross-validation is performed")
parser.add_argument('--test', default=False, action='store_true',
                    help="If listed, test is performed. Only works w/ specified alpha")
parser.add_argument('--full-train', default=False, action='store_true',
                    help="If listed, all available data is used to fit one model")


parser.add_argument('--overwrite-existing', default=False, action='store_true',
                    help="If folder/predictions df in it, that we expect to write to (CV folds) already exist we do not run this alpha unless this flag is used")
parser.add_argument('--use-subsetting', default=False, action='store_true',
                    help="If listed, only a fraction (radnom) of the dataset will be used for all train. Note: use with caution")
parser.add_argument('--use-subsetting-cv', default=False, action='store_true',
                    help="If listed, only a fraction (radnom) of the dataset will be used for CV phase. Combined with use-subsetting.")
parser.add_argument('--drop-fraction-subset', default=0.4, type=float,
                    help="If use subsetting, what fraction of the dataset to *drop*.")
parser.add_argument('--drop-fraction-cv', default=0.4, type=float,
                    help="If use subsetting, what fraction of the dataset to *drop* in CV, in addition to main drop.")
parser.add_argument('--condensed-location', default='/projects/TROYANSKAYA/sokolova/09_2020_predictions_condensed_exponential/', type=str,
                    help="Path to location of available X data")
parser.add_argument('--gene-tss-reference', default='resources/reference_files/compiled_CAGE_Gencode_tss_renamed.csv', type=str,
                    help="Path to reference file containing TSS for the genes. Should have gene id, column 'gene_name', 'chromosome' and 'tss' columns")
parser.add_argument('--gene-id-used', default='gene_name', type=str,
                    help="What gene ID is provided in reference, and is assume to be the type of index in gene expression")
parser.add_argument('--pseudocount', default=0.01, type=float,
                    help="Pseudocount to use for models")
parser.add_argument('--folds', default=5, type=int,
                    help="Number of folds if doing cross-validation")
# parser.add_argument('--save-models', default= True, type=bool,
#                     help="If True, models from each fold are saved.")
parser.add_argument('--save-preds', default=False, action='store_true',
                    help="If listed, predictions from each fold are saved.")
parser.add_argument('--not-weighted', default=True, action='store_false',
                    help="If True, the high variation genes are assigned higher weight")  # check for more details
parser.add_argument('--no-normalize', default=True, action='store_false',
                    help="If True, the normalize")  # check for more details
parser.add_argument('--random-state', default=321, type=int,
                    help="Random state.")

parser.add_argument('--on-off-drop-debug', default=False, action='store_true',
                    help="Drop on/off genes for evaluation purposes (not final train)")

# optional arguments that can be passed
# log_odds_norm = False but we are never using it, maybe should remove
# condensed_length

args = parser.parse_args()
model = train_group_linear
condensed_length = 20020
alpha_search = [0.01, 0.1, 1, 10, 100]

if not os.path.exists(args.save_location):
    os.makedirs(args.save_location)

logging.basicConfig(filename=args.save_location+'/INFO.log', filemode='w',
                    format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.info('Start run for {}'.format(args.expressions))

# Part 1: collect all the files required for training
expressions = pd.read_csv(args.expressions, index_col=0)
genes_list = expressions.columns.values
print(genes_list[0:10])

genes_tss_pd, lost_genes_N = get_genes_tss_pd(genes_list=genes_list,
                                              gene_tss_reference=args.gene_tss_reference, merge_on=args.gene_id_used)
print('N genes not found in TSS file: ', lost_genes_N)
print('N genes found in TSS file: ', genes_tss_pd.shape[0])
logging.info('{} genes not found in TSS file.'.format(lost_genes_N))
logging.info('{} genes found in TSS file.'.format(genes_tss_pd.shape[0]))

if 'gene_name' not in genes_tss_pd.columns:
    logging.error('gene_name column not found in the file.')
    raise ValueError('gene_name column not found in the file.')


if args.on_off_drop_debug:
    logging.info('Drop on/off genes for debugging purposes.')
    genes_list = binarize_truth_on_off(expressions, genes_tss_pd, percentile_cutoff=30)
    genes_tss_pd, lost_genes_N = get_genes_tss_pd(genes_list=genes_list,
                                              gene_tss_reference=args.gene_tss_reference, 
                                              merge_on=args.gene_id_used)
    print('on/off genes removed: N genes not found in TSS file: ', lost_genes_N)
    print('on/off genes removed: N genes found in TSS file: ', genes_tss_pd.shape[0])
    logging.info('on/off genes removed: {} genes not found in TSS file.'.format(lost_genes_N))
    logging.info('on/off genes removed: {} genes found in TSS file.'.format(genes_tss_pd.shape[0]))

if args.use_subsetting:
    # subset all available genes
    logging.info('Subsetting the whole dataset as requested by user (--use-subsetting flag).')
    print('Subsetting dataset')
    genes_tss_pd = genes_tss_pd.sample(frac = 1-args.drop_fraction_subset).reset_index(drop=True)
    print('New size of train dataset: ', genes_tss_pd.shape[0])
    logging.info('New size of dataset: {}'.format(genes_tss_pd.shape[0]))



print('Collecting condense dictionary, can take up to 20-30 minutes')
logging.info('Collecting condense dictionary, can take up to 20-30 minutes, depending on dataset size')
condensed_dict, n_genes_not_found = collect_condense_genes_from_csv(
    args.condensed_location, genes_tss_pd)
print('N genes without condense (dropped): ', n_genes_not_found)
logging.info('{} genes without condense (dropped)'.format(n_genes_not_found))

print('weighted', args.not_weighted)
logging.info('weighted: {}'.format(args.not_weighted))

print('norm', args.no_normalize)
logging.info('normalize: {}'.format(args.no_normalize))


# Part 2: CV

if args.cv and not args.find_alpha:
    logging.info(
        'Cross validation with user specified alpha {}'.format(args.alpha))
    params = {'alpha': args.alpha, 'normalize': args.no_normalize}

    save_loc_alpha = args.save_location + \
            '/CV_folds/evaluations_alpha_'+str(args.alpha)+'/'

    truth_preds, results = cross_val_predict_all_groups(model, genes_tss_pd, condensed_dict, expressions,
                                                        params, args.pseudocount,
                                                        folds=args.folds, random_state=args.random_state,
                                                        save_preds=args.save_preds,
                                                        save_preds_locs=save_loc_alpha,
                                                        weighted=args.not_weighted, condensed_length=condensed_length, use_log=True,
                                                        subset_size=args.use_subsetting_cv, fraction_drop=args.drop_fraction_cv)
    print('Performed CV fit.')
    logging.info('Performed CV fit.')
    save_evals_loc = args.save_location+'/CV_folds/evaluations/'
    spearman_scores, truth_expr, pred_expr = run_evaluations(truth_preds, args, save_evals_loc)
    logging.info('Computed Spearman correlations.')

    plot_AUC_high_low(truth_expr, pred_expr, '', save_evals_loc, logger = logging)

if args.test and not args.cv and not args.find_alpha:
    logging.info(
        'Only test with user specified alpha {}'.format(args.alpha))
    params = {'alpha': args.alpha, 'normalize': args.no_normalize}

    save_loc_alpha = args.save_location + \
            '/CV_folds/evaluations_test_alpha_'+str(args.alpha)+'/'

    results = evals_train_test(model, genes_tss_pd, condensed_dict, expressions,
                                                    params, args.pseudocount,
                                                    folds=args.folds, random_state=args.random_state,
                                                    save_preds=args.save_preds,
                                                    save_preds_locs=save_loc_alpha,
                                                    weighted=args.not_weighted, condensed_length=condensed_length, use_log=True,
                                                    subset_size=args.use_subsetting_cv, fraction_drop=args.drop_fraction_cv)
    print('Performed test fit.')
    logging.info('Performed test fit.')

    

if args.cv and args.find_alpha:
    logging.info('Cross validation with alpha search')
    best_mean_spearman = -1
    best_alpha = -1
    for alpha in alpha_search:
        params = {'alpha': alpha, 'normalize': args.no_normalize}
        save_loc_alpha = args.save_location + \
            '/CV_folds/evaluations_alpha_'+str(alpha)+'/'
        print(save_loc_alpha+'predicted_expression.csv')
        if os.path.exists(save_loc_alpha+'predicted_expression.csv') and not args.overwrite_existing:
            logging.info('{} already existed. Skipping.'.format(alpha))
            print('{} already existed. Skipping.'.format(alpha))
            continue
        truth_preds, results = cross_val_predict_all_groups(model, genes_tss_pd, condensed_dict, expressions,
                                                            params, args.pseudocount,
                                                            folds=args.folds, random_state=args.random_state,
                                                            save_preds=args.save_preds,
                                                            save_preds_locs=save_loc_alpha,
                                                            weighted=args.not_weighted, 
                                                            condensed_length=condensed_length, use_log=True,
                                                            subset_size=args.use_subsetting_cv, fraction_drop=args.drop_fraction_cv)
        print('Performed CV fit for alpha ', alpha)
        logging.info('Performed CV fit for alpha {}'.format(alpha))
        save_evals_loc = args.save_location + \
            '/CV_folds/evaluations_alpha_'+str(alpha)+'/'
        spearman_scores, truth_expr, pred_expr = run_evaluations(truth_preds, args, save_evals_loc)
        print('Mean Spearman: ', spearman_scores.Spearman.mean())
        logging.info('Mean Spearman for this alpha: {}'.format(spearman_scores.Spearman.mean()))
        if best_mean_spearman < spearman_scores.Spearman.mean():
            best_mean_spearman = spearman_scores.Spearman.mean()
            best_alpha = alpha

        plot_AUC_high_low(truth_expr, pred_expr, '', save_evals_loc, logger = logging)

    print('Best alpha among new models: ', best_alpha)
    logging.info('BEST ALPHA found is {} for mean spearman of {} among all the new groups'.format(
        best_alpha, best_mean_spearman))

    print('Check all available alphas')
    logging.info('Check all available alphas')
    avail_alphas = os.listdir(args.save_location+'/CV_folds/')
    print(avail_alphas)
    avail_alphas = [i for i in avail_alphas if 'evaluations_alpha' in i]
    avail_alphas = [i for i in avail_alphas if i[0]!='.']
    print(avail_alphas)
    logging.info(avail_alphas)
    best_mean_spearman = -1
    best_alpha = -1
    for avail_alpha in avail_alphas:
        spearman_alpha = pd.read_csv(args.save_location+'/CV_folds/'+avail_alpha+'/spearman_per_group.csv')
        print(avail_alpha, spearman_alpha.Spearman.mean())
        logging.info('for {} mean Spearman is {}'.format(avail_alphas, spearman_alpha.Spearman.mean()))
        if best_mean_spearman < spearman_alpha.Spearman.mean():
            best_mean_spearman = spearman_alpha.Spearman.mean()
            best_alpha = float(avail_alpha.split('_')[-1])

    logging.info('BEST ALPHA overall is {} for mean spearman of {} among all the new groups'.format(
        best_alpha, best_mean_spearman))
    logging.info('***')

# Part 3: Full train

if args.full_train:
    if args.find_alpha:
        params = {'alpha': best_alpha, 'normalize': args.no_normalize}
    else:
        params = {'alpha': args.alpha, 'normalize': args.no_normalize}

    logging.info(
        'Perform a full train using all the data, alpha is {}'.format(params['alpha']))
    print('Doing full train with alpha', params['alpha'])

    # subsetting is performed before this command here (if wanted)
    results, preds_df, truth_df = train_all_groups(model, genes_tss_pd, condensed_dict,
                                                   expressions, params, args.pseudocount,
                                                   save_preds=False,
                                                   save_models=True,
                                                   save_models_loc=args.save_location+'/models_full_train/',
                                                   weighted=args.not_weighted, condensed_length=condensed_length,
                                                   use_log=True, fraction_valid=0, run_eval=False,
                                                   subset_size=False, fraction_drop=0)
    logging.info('All models trained')
