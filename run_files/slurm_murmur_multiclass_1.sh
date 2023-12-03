#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --ntasks-per-node=28
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_mem:32GB'
#SBATCH --clusters=htc
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --job-name=murmul1

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk


# Load environment
module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet22
conda info --env



# MULTICLASS MURMUR ############################################################
## Train
#python train_resnet.py --recalc_features --model_label BinaryPresent --classes_name binary_present --train_data_directory data/a_splits/cv_True_stratified_False/split_0/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_0/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_0 --model_dir data/c_models/cv_True_stratified_False/split_0
#python train_resnet.py --recalc_features --model_label BinaryPresent --classes_name binary_present --train_data_directory data/a_splits/cv_True_stratified_False/split_1/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_1/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_1 --model_dir data/c_models/cv_True_stratified_False/split_1
#python train_resnet.py --recalc_features --model_label BinaryPresent --classes_name binary_present --train_data_directory data/a_splits/cv_True_stratified_False/split_2/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_2/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_2 --model_dir data/c_models/cv_True_stratified_False/split_2
#python train_resnet.py --recalc_features --model_label BinaryPresent --classes_name binary_present --train_data_directory data/a_splits/cv_True_stratified_False/split_3/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_3/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_3 --model_dir data/c_models/cv_True_stratified_False/split_3
#python train_resnet.py --recalc_features --model_label BinaryPresent --classes_name binary_present --train_data_directory data/a_splits/cv_True_stratified_False/split_4/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_4/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_4 --model_dir data/c_models/cv_True_stratified_False/split_4

#python train_resnet.py --recalc_features --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/a_splits/cv_True_stratified_False/split_0/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_0/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_0 --model_dir data/c_models/cv_True_stratified_False/split_0
#python train_resnet.py --recalc_features --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/a_splits/cv_True_stratified_False/split_1/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_1/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_1 --model_dir data/c_models/cv_True_stratified_False/split_1
#python train_resnet.py --recalc_features --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/a_splits/cv_True_stratified_False/split_2/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_2/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_2 --model_dir data/c_models/cv_True_stratified_False/split_2
#python train_resnet.py --recalc_features --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/a_splits/cv_True_stratified_False/split_3/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_3/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_3 --model_dir data/c_models/cv_True_stratified_False/split_3
#python train_resnet.py --recalc_features --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/a_splits/cv_True_stratified_False/split_4/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_4/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_4 --model_dir data/c_models/cv_True_stratified_False/split_4

## Run and Evaluate
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_0/test_data --output_directory data/d_dbres_multiclass_outputs/cv_True_stratified_False/split_0 --model_binary_present_pth data/c_models/cv_True_stratified_False/split_0/model_BinaryPresent.pth --model_binary_unknown_pth data/c_models/cv_True_stratified_False/split_0/model_BinaryUnknown.pth
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_1/test_data --output_directory data/d_dbres_multiclass_outputs/cv_True_stratified_False/split_1 --model_binary_present_pth data/c_models/cv_True_stratified_False/split_1/model_BinaryPresent.pth --model_binary_unknown_pth data/c_models/cv_True_stratified_False/split_1/model_BinaryUnknown.pth
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_2/test_data --output_directory data/d_dbres_multiclass_outputs/cv_True_stratified_False/split_2 --model_binary_present_pth data/c_models/cv_True_stratified_False/split_2/model_BinaryPresent.pth --model_binary_unknown_pth data/c_models/cv_True_stratified_False/split_2/model_BinaryUnknown.pth
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_3/test_data --output_directory data/d_dbres_multiclass_outputs/cv_True_stratified_False/split_3 --model_binary_present_pth data/c_models/cv_True_stratified_False/split_3/model_BinaryPresent.pth --model_binary_unknown_pth data/c_models/cv_True_stratified_False/split_3/model_BinaryUnknown.pth
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_4/test_data --output_directory data/d_dbres_multiclass_outputs/cv_True_stratified_False/split_4 --model_binary_present_pth data/c_models/cv_True_stratified_False/split_4/model_BinaryPresent.pth --model_binary_unknown_pth data/c_models/cv_True_stratified_False/split_4/model_BinaryUnknown.pth
