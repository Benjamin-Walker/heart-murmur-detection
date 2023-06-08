#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=28
#SBATCH --gres=gpu:v100:1
#SBATCH --clusters=htc
#SBATCH --time=72:00:00
#SBATCH --partition=long
#SBATCH --job-name=murmul2

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk


# Load environment
module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet22
conda info --env



# MULTICLASS MURMUR ############################################################
## Train
python train_resnet.py --recalc_features --model_label BinaryPresent --classes_name binary_present --train_data_directory data/a_splits/cv_True_stratified_False/split_5/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_5/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_5 --model_dir data/c_models/cv_True_stratified_False/split_5
python train_resnet.py --recalc_features --model_label BinaryPresent --classes_name binary_present --train_data_directory data/a_splits/cv_True_stratified_False/split_6/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_6/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_6 --model_dir data/c_models/cv_True_stratified_False/split_6
python train_resnet.py --recalc_features --model_label BinaryPresent --classes_name binary_present --train_data_directory data/a_splits/cv_True_stratified_False/split_7/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_7/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_7 --model_dir data/c_models/cv_True_stratified_False/split_7
python train_resnet.py --recalc_features --model_label BinaryPresent --classes_name binary_present --train_data_directory data/a_splits/cv_True_stratified_False/split_8/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_8/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_8 --model_dir data/c_models/cv_True_stratified_False/split_8
python train_resnet.py --recalc_features --model_label BinaryPresent --classes_name binary_present --train_data_directory data/a_splits/cv_True_stratified_False/split_9/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_9/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_9 --model_dir data/c_models/cv_True_stratified_False/split_9

python train_resnet.py --recalc_features --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/a_splits/cv_True_stratified_False/split_5/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_5/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_5 --model_dir data/c_models/cv_True_stratified_False/split_5
python train_resnet.py --recalc_features --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/a_splits/cv_True_stratified_False/split_6/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_6/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_6 --model_dir data/c_models/cv_True_stratified_False/split_6
python train_resnet.py --recalc_features --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/a_splits/cv_True_stratified_False/split_7/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_7/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_7 --model_dir data/c_models/cv_True_stratified_False/split_7
python train_resnet.py --recalc_features --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/a_splits/cv_True_stratified_False/split_8/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_8/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_8 --model_dir data/c_models/cv_True_stratified_False/split_8
python train_resnet.py --recalc_features --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/a_splits/cv_True_stratified_False/split_9/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_9/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_9 --model_dir data/c_models/cv_True_stratified_False/split_9

## Run and Evaluate
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_5/test_data --output_directory data/d_dbres_multiclass_outputs/cv_True_stratified_False/split_5 --model_binary_present_pth data/c_models/cv_True_stratified_False/split_5/model_BinaryPresent.pth --model_binary_unknown_pth data/c_models/cv_True_stratified_False/split_5/model_BinaryUnknown.pth
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_6/test_data --output_directory data/d_dbres_multiclass_outputs/cv_True_stratified_False/split_6 --model_binary_present_pth data/c_models/cv_True_stratified_False/split_6/model_BinaryPresent.pth --model_binary_unknown_pth data/c_models/cv_True_stratified_False/split_6/model_BinaryUnknown.pth
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_7/test_data --output_directory data/d_dbres_multiclass_outputs/cv_True_stratified_False/split_7 --model_binary_present_pth data/c_models/cv_True_stratified_False/split_7/model_BinaryPresent.pth --model_binary_unknown_pth data/c_models/cv_True_stratified_False/split_7/model_BinaryUnknown.pth
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_8/test_data --output_directory data/d_dbres_multiclass_outputs/cv_True_stratified_False/split_8 --model_binary_present_pth data/c_models/cv_True_stratified_False/split_8/model_BinaryPresent.pth --model_binary_unknown_pth data/c_models/cv_True_stratified_False/split_8/model_BinaryUnknown.pth
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_9/test_data --output_directory data/d_dbres_multiclass_outputs/cv_True_stratified_False/split_9 --model_binary_present_pth data/c_models/cv_True_stratified_False/split_9/model_BinaryPresent.pth --model_binary_unknown_pth data/c_models/cv_True_stratified_False/split_9/model_BinaryUnknown.pth
