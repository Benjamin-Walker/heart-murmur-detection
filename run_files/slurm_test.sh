#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --ntasks-per-node=28
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_mem:32GB'
#SBATCH --clusters=htc
#SBATCH --time=00:20:00
#SBATCH --partition=medium
#SBATCH --job-name=murbin1

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk

# Load environment
module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet22
conda info --env



# BINARY MURMUR ################################################################
## Train
python train_resnet.py --recalc_features --model_label MurmurBinary --classes_name murmur_binary --train_data_directory data/a_splits/cv_True_stratified_False/split_0/train_data --vali_data_directory data/a_splits/cv_True_stratified_False/split_0/vali_data --spectrogram_directory data/b_spectrograms/cv_True_stratified_False/split_0 --model_dir data/c_models/cv_True_stratified_False/split_0
