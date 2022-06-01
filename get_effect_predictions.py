""" Main script to get effect predictions for the mutations """

import argparse
import logging
import os
import sys
import pandas as pd
from encoder_for_variants import *
from beluga_convert import *
from condense_data import *
from main_train import *


parser = argparse.ArgumentParser()

parser.add_argument('--variants', default='', type=str,
                    help="Path to .csv file with variants of interest. Columns: gene_name, tss, chrom, strand, cage_used,snp_chromStart, ref, alt ")

# Predictions
parser.add_argument('--cell-model-path', default=None, type=str,
                    help="Path to the folder with models. Should have a 'models-subfolde' with all models inside")
parser.add_argument('--save-location-preds', default=None, type=str,
                    help="Path to folder to save results")
parser.add_argument('--filename-output', default='predictions', type=str,
                    help="What name should be prepended to _ref and _alt predictions.")

parser.add_argument('--run-encoder', default=False, action='store_true',
                    help="Do we want to run encoder for the missing variants. Not recommended to run in this file for large amount of new variants.")
parser.add_argument('--use-cuda', default=False, action='store_true',
                    help="Use CUDA (when/if running encoder)")

parser.add_argument('--ldsc', default=False, action='store_true',
                    help="For LDSC we take abs() value")
parser.add_argument('--ldsc-location', default='', type=str,
                    help="LDSC (filtered to +/- 20kb location")                    

parser.add_argument('--models-subfolder', default='models_full_train', type=str,
                    help="Subfolder with models")                   
# Encoder locations
parser.add_argument('--encoder-model-path', default='resources/deepsea.beluga.pth', type=str,
                    help="Path to encoder")
parser.add_argument('--encoder-output-save-path', default='', type=str,
                    help="Where to save raw encodings")
parser.add_argument('--compressed-output-save-path', default='', type=str,
                    help="Where to save compressed encodings")

args = parser.parse_args()


if not os.path.exists(args.save_location_preds):
    os.makedirs(args.save_location_preds)

if not os.path.exists(args.save_location_preds+'/intermediate'):
    os.makedirs(args.save_location_preds+'/intermediate')


logging.basicConfig(filename=args.save_location_preds+'/intermediate/'+args.filename_output+'.log', filemode='w',
                    format='%(asctime)s - %(message)s', level=logging.DEBUG)
logging.info('Start run for {}'.format(args.variants))

if not args.ldsc:
    variants_df = pd.read_csv(args.variants)

if args.run_encoder and not args.ldsc:
    # load model
    model = deepsea_beluga_2002_cpu
    model = load_model_from_state_dict(torch.load(args.encoder_model_path), model)
    model = model.eval()
    print('Model loaded')
    logging.info('Encoder loaded')


    #############
    # Encode provided variants
    process_variants_with_ref(
        variants_file=variants_df, destination_folder=args.encoder_output_save_path, model=model, use_cuda=args.use_cuda)
    logging.info('Variants encoded')

    #############
    # Condense genes that were in that variants dataframe
    for index, row in variants_df.iterrows():
        filename = row['gene_name'] + '_'+str(int(row['tss']))+'.hd5f'
        condense_and_save(condense_gene_fn=condense_gene_exponential,
                        filenames=[
                            filename], location_noncondesed=args.encoder_output_save_path,
                        location_save=args.compressed_output_save_path)
    logging.info('Encoding condensed')

#############
# Predictions

if not args.ldsc:
    logging.info('*** Start predicting ***')
    collect_predictions_models(args.cell_model_path, variants_df, base_save_loc=args.save_location_preds,
                            compressed_encodings_loc=args.compressed_output_save_path, logging=logging, 
                            save_filename_base_name=args.filename_output,
                            models_inside_folder = args.models_subfolder)
    logging.info('*** DONE ***')
else:
    logging.info('*** LDSC predictions ***')
    chroms = range(1, 24)
    for chrom in chroms:
        ldsc_file = args.ldsc_location+'/variants.chr' + str(chrom)+'.closest.filtered.txt'
        if not os.path.exists(ldsc_file):
            logging.warning('File does not exist: {}'.format(ldsc_file))
            continue
        # check if this ldsc already run
        if os.path.exists(args.save_location_preds+'/'+'chrom.'+str(chrom)+'.csv'):
            logging.info('Chrom {} already existed'.format(chrom))
            continue
        variants_df = pd.read_csv(ldsc_file)
        collect_predictions_models(args.cell_model_path, variants_df, base_save_loc=args.save_location_preds,
                            compressed_encodings_loc=args.compressed_output_save_path, logging=logging, 
                            save_filename_base_name='chrom.'+str(chrom),
                            tss_col_name = 'cl_chromStart',
                            models_inside_folder = args.models_subfolder, A1='A1', A2='A2')
        logging.info('Completed chrom {}'.format(chrom))
    logging.info('*** DONE ***')
        

        