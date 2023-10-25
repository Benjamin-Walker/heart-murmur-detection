#!/bin/sh

#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --ntasks-per-node=28
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_mem:32GB'
#SBATCH --clusters=htc
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --job-name=res

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk

# Load environment
module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet22
conda info --env

model_label=MurmurBinary_Res
model_name=resnet50
dl_output=MurmurBinary_Res
xgb_output=MurmurBinary_Res

# SPLIT ########################################################################
#python data_splits.py --data_directory /Users/felixkrones/python_projects/data/physionet_challenge_2022/physionet.org/files/circor-heart-sound/1.0.3/training_data --vali_size 0.2 --test_size 0.2 --cv True

# BINARY MURMUR ResDrop ################################################################
## Train
python train_resnet.py --recalc_features --model_label MurmurBinary_Res --classes_name murmur_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_0/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_0/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_0 --model_dir data/c_models/cv_True_stratified_False/split_0 --model_name $model_name --disable-bayesian
python train_resnet.py --recalc_features --model_label MurmurBinary_Res --classes_name murmur_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_1/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_1/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_1 --model_dir data/c_models/cv_True_stratified_False/split_1 --model_name $model_name --disable-bayesian
python train_resnet.py --recalc_features --model_label MurmurBinary_Res --classes_name murmur_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_2/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_2/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_2 --model_dir data/c_models/cv_True_stratified_False/split_2 --model_name $model_name --disable-bayesian
python train_resnet.py --recalc_features --model_label MurmurBinary_Res --classes_name murmur_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_3/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_3/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_3 --model_dir data/c_models/cv_True_stratified_False/split_3 --model_name $model_name --disable-bayesian
python train_resnet.py --recalc_features --model_label MurmurBinary_Res --classes_name murmur_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_4/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_4/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_4 --model_dir data/c_models/cv_True_stratified_False/split_4 --model_name $model_name --disable-bayesian
python train_resnet.py --recalc_features --model_label MurmurBinary_Res --classes_name murmur_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_5/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_5/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_5 --model_dir data/c_models/cv_True_stratified_False/split_5 --model_name $model_name --disable-bayesian
python train_resnet.py --recalc_features --model_label MurmurBinary_Res --classes_name murmur_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_6/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_6/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_6 --model_dir data/c_models/cv_True_stratified_False/split_6 --model_name $model_name --disable-bayesian
python train_resnet.py --recalc_features --model_label MurmurBinary_Res --classes_name murmur_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_7/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_7/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_7 --model_dir data/c_models/cv_True_stratified_False/split_7 --model_name $model_name --disable-bayesian
python train_resnet.py --recalc_features --model_label MurmurBinary_Res --classes_name murmur_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_8/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_8/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_8 --model_dir data/c_models/cv_True_stratified_False/split_8 --model_name $model_name --disable-bayesian
python train_resnet.py --recalc_features --model_label MurmurBinary_Res --classes_name murmur_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_9/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_9/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_9 --model_dir data/c_models/cv_True_stratified_False/split_9 --model_name $model_name --disable-bayesian

## Run and Evaluate
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_0/test_data --output_directory data/d_dl_output/${dl_output}/cv_True_stratified_False/split_0 --model_binary_pth data/c_models/cv_True_stratified_False/split_0/model_${model_label}.pth --model_name $model_name --disable-bayesian
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_1/test_data --output_directory data/d_dl_output/${dl_output}/cv_True_stratified_False/split_1 --model_binary_pth data/c_models/cv_True_stratified_False/split_1/model_${model_label}.pth --model_name $model_name --disable-bayesian
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_2/test_data --output_directory data/d_dl_output/${dl_output}/cv_True_stratified_False/split_2 --model_binary_pth data/c_models/cv_True_stratified_False/split_2/model_${model_label}.pth --model_name $model_name --disable-bayesian
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_3/test_data --output_directory data/d_dl_output/${dl_output}/cv_True_stratified_False/split_3 --model_binary_pth data/c_models/cv_True_stratified_False/split_3/model_${model_label}.pth --model_name $model_name --disable-bayesian
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_4/test_data --output_directory data/d_dl_output/${dl_output}/cv_True_stratified_False/split_4 --model_binary_pth data/c_models/cv_True_stratified_False/split_4/model_${model_label}.pth --model_name $model_name --disable-bayesian
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_5/test_data --output_directory data/d_dl_output/${dl_output}/cv_True_stratified_False/split_5 --model_binary_pth data/c_models/cv_True_stratified_False/split_5/model_${model_label}.pth --model_name $model_name --disable-bayesian
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_6/test_data --output_directory data/d_dl_output/${dl_output}/cv_True_stratified_False/split_6 --model_binary_pth data/c_models/cv_True_stratified_False/split_6/model_${model_label}.pth --model_name $model_name --disable-bayesian
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_7/test_data --output_directory data/d_dl_output/${dl_output}/cv_True_stratified_False/split_7 --model_binary_pth data/c_models/cv_True_stratified_False/split_7/model_${model_label}.pth --model_name $model_name --disable-bayesian
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_8/test_data --output_directory data/d_dl_output/${dl_output}/cv_True_stratified_False/split_8 --model_binary_pth data/c_models/cv_True_stratified_False/split_8/model_${model_label}.pth --model_name $model_name --disable-bayesian
python dbres.py --recalc_output --data_directory data/a_splits/cv_True_stratified_False/split_9/test_data --output_directory data/d_dl_output/${dl_output}/cv_True_stratified_False/split_9 --model_binary_pth data/c_models/cv_True_stratified_False/split_9/model_${model_label}.pth --model_name $model_name --disable-bayesian
