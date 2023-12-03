#!/bin/sh

# BINARY MURMUR Bayesian ################################################################
model_name=resnet50dropout
model_label=Outcome_Binary_Bayesian
dl_output=Outcome_Binary_Bayesian
classes_name=outcome_binary

## Train
#python train_resnet.py --recalc_features --model_label $model_label --classes_name $classes_name --train_data_directory data/a_splits/yaseen/train_data --vali_data_directory data/a_splits/yaseen/vali_data --spectrogram_directory data/b_spectrograms/yaseen --model_dir data/c_models/yaseen --model_name $model_name

## Run and Evaluate
python dbres.py --recalc_output --data_directory data/a_splits/yaseen/test_data --output_directory data/d_dl_output/${dl_output}/yaseen --model_binary_pth data/c_models/yaseen/model_${model_label}.pth --model_name $model_name
