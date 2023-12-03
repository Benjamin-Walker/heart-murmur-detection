#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --ntasks-per-node=28
#SBATCH --gres=gpu:v100:1
#SBATCH --clusters=htc
#SBATCH --time=00:10:00
#SBATCH --partition=devel
#SBATCH --job-name=test


# Load environment
module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet22
conda info --env

python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_5/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_5/test_data --dbres_output_directory data/d_xgboost_OutcomeBinary_outputs/cv_True_stratified_False/split_5/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_5/model_OutcomeBinary.pth --output_directory data/xgboost_OutcomeBinary_outputs/cv_True_stratified_False/split_5
