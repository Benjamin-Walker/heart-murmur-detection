#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem=180G
#SBATCH --ntasks-per-node=28
#SBATCH --gres=gpu:1
#SBATCH --clusters=htc
#SBATCH --time=00:10:00
#SBATCH --partition=devel
#SBATCH --job-name=m_ood

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=felix.krones@oii.ox.ac.uk


# Load environment
module load Anaconda3
source activate /data/inet-multimodal-ai/wolf6245/envs/physionet22
conda info --env


python dbres.py --recalc_output --recordings_file data/e_challenge_anaylsis/2016_challenge_annotations.csv --output_directory data/f_pyc2016/dbres_outputs --model_binary_pth data/c_models/cv_True_stratified_False/split_0/model_OutcomeBinary.pth
