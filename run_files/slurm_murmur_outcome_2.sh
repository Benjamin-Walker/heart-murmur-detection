#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --ntasks-per-node=28
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_mem:32GB'
#SBATCH --clusters=htc
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --job-name=murout2

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk


# Load environment
module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet22
conda info --env


# BINARY OUTCOME ###############################################################
## Train
#python train_resnet.py --recalc_features --model_label OutcomeBinary --classes_name outcome_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_5/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_5/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_5 --model_dir data/c_models/cv_True_stratified_False/split_5
#python train_resnet.py --recalc_features --model_label OutcomeBinary --classes_name outcome_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_6/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_6/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_6 --model_dir data/c_models/cv_True_stratified_False/split_6
#python train_resnet.py --recalc_features --model_label OutcomeBinary --classes_name outcome_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_7/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_7/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_7 --model_dir data/c_models/cv_True_stratified_False/split_7
#python train_resnet.py --recalc_features --model_label OutcomeBinary --classes_name outcome_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_8/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_8/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_8 --model_dir data/c_models/cv_True_stratified_False/split_8
#python train_resnet.py --recalc_features --model_label OutcomeBinary --classes_name outcome_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_9/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_9/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_9 --model_dir data/c_models/cv_True_stratified_False/split_9

## Run and Evaluate
#python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_5/test_data --output_directory data/d_dbres_OutcomeBinary_outputs/cv_True_stratified_False/split_5 --model_binary_pth data/c_models/cv_True_stratified_False/split_5/model_OutcomeBinary.pth
#python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_6/test_data --output_directory data/d_dbres_OutcomeBinary_outputs/cv_True_stratified_False/split_6 --model_binary_pth data/c_models/cv_True_stratified_False/split_6/model_OutcomeBinary.pth
#python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_7/test_data --output_directory data/d_dbres_OutcomeBinary_outputs/cv_True_stratified_False/split_7 --model_binary_pth data/c_models/cv_True_stratified_False/split_7/model_OutcomeBinary.pth
#python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_8/test_data --output_directory data/d_dbres_OutcomeBinary_outputs/cv_True_stratified_False/split_8 --model_binary_pth data/c_models/cv_True_stratified_False/split_8/model_OutcomeBinary.pth
#python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_9/test_data --output_directory data/d_dbres_OutcomeBinary_outputs/cv_True_stratified_False/split_9 --model_binary_pth data/c_models/cv_True_stratified_False/split_9/model_OutcomeBinary.pth

## Run XGBoooost models
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_5/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_5/test_data --dbres_output_directory data/d_xgboost_OutcomeBinary_outputs/cv_True_stratified_False/split_5/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_5/model_OutcomeBinary.pth --output_directory data/xgboost_OutcomeBinary_outputs/cv_True_stratified_False/split_5
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_6/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_6/test_data --dbres_output_directory data/d_xgboost_OutcomeBinary_outputs/cv_True_stratified_False/split_6/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_6/model_OutcomeBinary.pth --output_directory data/xgboost_OutcomeBinary_outputs/cv_True_stratified_False/split_6
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_7/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_7/test_data --dbres_output_directory data/d_xgboost_OutcomeBinary_outputs/cv_True_stratified_False/split_7/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_7/model_OutcomeBinary.pth --output_directory data/xgboost_OutcomeBinary_outputs/cv_True_stratified_False/split_7
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_8/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_8/test_data --dbres_output_directory data/d_xgboost_OutcomeBinary_outputs/cv_True_stratified_False/split_8/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_8/model_OutcomeBinary.pth --output_directory data/xgboost_OutcomeBinary_outputs/cv_True_stratified_False/split_8
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_9/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_9/test_data --dbres_output_directory data/d_xgboost_OutcomeBinary_outputs/cv_True_stratified_False/split_9/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_9/model_OutcomeBinary.pth --output_directory data/xgboost_OutcomeBinary_outputs/cv_True_stratified_False/split_9

