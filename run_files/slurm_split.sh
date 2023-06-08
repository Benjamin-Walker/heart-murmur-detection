#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=28
#SBATCH --time=00:30:00
#SBATCH --partition=short
#SBATCH --job-name=split

# Load environment
module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet22
conda info --env

# Split data in 10 fold
python data_splits.py --data_directory /data/inet-multimodal-ai/wolf6245/data/physionet_challenge_2022/physionet.org/files/circor-heart-sound/1.0.3/training_data --vali_size 0.2 --test_size 0.2 --cv True
