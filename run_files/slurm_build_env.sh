#! /bin/bash

# Load the version of Anaconda you need
module load Anaconda3

# Create an environment in $DATA and give it an appropriate name
export CONPREFIX=$DATA/envs/physionet22
conda create --prefix $CONPREFIX python=3.9

# Activate your environment
source activate $CONPREFIX

# Install packages...
conda install -y pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch -c nvidia
pip install tqdm==4.64.1 scipy==1.9.2 resampy==0.4.2 scikit-learn==1.1.2 pandas==1.5.0 xgboost==1.6.2 librosa==0.9.2
