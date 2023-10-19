#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --ntasks-per-node=28
#SBATCH --gres=gpu:1
#SBATCH --clusters=htc
#SBATCH --time=00:10:00
#SBATCH --partition=devel
#SBATCH --job-name=murxgbw

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk


# Load environment
module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet22
conda info --env



# BINARY MURMUR ################################################################
## Run XGBooost models
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_0/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_0/test_data --dbres_output_directory data/d_xgboost_MurmurBinary_outputs/cv_True_stratified_False/split_0/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_0/model_MurmurBinary.pth --output_directory data/d_xgboost_MurmurBinary_outputs_weighted/cv_True_stratified_False/split_0 --use_weights True
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_1/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_1/test_data --dbres_output_directory data/d_xgboost_MurmurBinary_outputs/cv_True_stratified_False/split_1/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_1/model_MurmurBinary.pth --output_directory data/d_xgboost_MurmurBinary_outputs_weighted/cv_True_stratified_False/split_1 --use_weights True
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_2/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_2/test_data --dbres_output_directory data/d_xgboost_MurmurBinary_outputs/cv_True_stratified_False/split_2/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_2/model_MurmurBinary.pth --output_directory data/d_xgboost_MurmurBinary_outputs_weighted/cv_True_stratified_False/split_2 --use_weights True
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_3/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_3/test_data --dbres_output_directory data/d_xgboost_MurmurBinary_outputs/cv_True_stratified_False/split_3/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_3/model_MurmurBinary.pth --output_directory data/d_xgboost_MurmurBinary_outputs_weighted/cv_True_stratified_False/split_3 --use_weights True
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_4/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_4/test_data --dbres_output_directory data/d_xgboost_MurmurBinary_outputs/cv_True_stratified_False/split_4/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_4/model_MurmurBinary.pth --output_directory data/d_xgboost_MurmurBinary_outputs_weighted/cv_True_stratified_False/split_4 --use_weights True
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_5/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_5/test_data --dbres_output_directory data/d_xgboost_MurmurBinary_outputs/cv_True_stratified_False/split_5/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_5/model_MurmurBinary.pth --output_directory data/d_xgboost_MurmurBinary_outputs_weighted/cv_True_stratified_False/split_5 --use_weights True
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_6/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_6/test_data --dbres_output_directory data/d_xgboost_MurmurBinary_outputs/cv_True_stratified_False/split_6/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_6/model_MurmurBinary.pth --output_directory data/d_xgboost_MurmurBinary_outputs_weighted/cv_True_stratified_False/split_6 --use_weights True
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_7/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_7/test_data --dbres_output_directory data/d_xgboost_MurmurBinary_outputs/cv_True_stratified_False/split_7/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_7/model_MurmurBinary.pth --output_directory data/d_xgboost_MurmurBinary_outputs_weighted/cv_True_stratified_False/split_7 --use_weights True
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_8/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_8/test_data --dbres_output_directory data/d_xgboost_MurmurBinary_outputs/cv_True_stratified_False/split_8/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_8/model_MurmurBinary.pth --output_directory data/d_xgboost_MurmurBinary_outputs_weighted/cv_True_stratified_False/split_8 --use_weights True
python xgboost_integration.py --train_data_directory data/a_splits/cv_True_stratified_False/split_9/train_data --test_data_directory data/a_splits/cv_True_stratified_False/split_9/test_data --dbres_output_directory data/d_xgboost_MurmurBinary_outputs/cv_True_stratified_False/split_9/dbres_output --model_binary_pth data/c_models/cv_True_stratified_False/split_9/model_MurmurBinary.pth --output_directory data/d_xgboost_MurmurBinary_outputs_weighted/cv_True_stratified_False/split_9 --use_weights True
