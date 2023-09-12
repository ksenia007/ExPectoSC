#!/bin/bash
#SBATCH --time=00:09:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=20000
#SBATCH -o logs/runtime_%j.out
#SBATCH -e logs/runtime_%j.err


module load cudnn/cuda-11.x/8.2.0
module load anaconda3/2022.5
source expectosc_env/bin/activate

FILE="resources/test.csv"
file_temp="test_out"
python3.6 -u get_effect_predictions.py --variants "${FILE}" --cell-model-path ./trained_models/kidney --save-location-pred ./outputs --filename-output "${file_temp}" --encoder-output-save-path ./temp_out/ --compressed-output-save-path ./temp_out_compr/ --run-encoder --use-cuda
