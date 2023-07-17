#!/bin/bash
#SBATCH --time=15:13:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH -o logs/runtime_%j.out
#SBATCH -e logs/runtime_%j.err

module load cudnn/cuda-8.0/6.0
module load anaconda3/5.3.1

source activate env

CHROM="22"
FILE="filtered.chr${CHROM}.csv"
file_temp="chr${CHROM}_20k"
python -u get_effect_predictions.py --variants "${FILE}" --cell-model-path /final_models/kidney --save-location-pred /outputs --filename-output "${file_temp}" --encoder-output-save-path /path-to-save/ --compressed-output-save-path /path-to-save/ --run-encoder --use-cuda
