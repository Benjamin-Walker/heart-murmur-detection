#!/bin/sh

# Split the data
#python stratified_data_split.py --data_directory /Users/felixkrones/python_projects/data/physionet_challenge_2022/physionet.org/files/circor-heart-sound/1.0.3/training_data --vali_size 0.2 --test_size 0.2 --cv True


# Train models
python train_resnet.py --recalc_features --model_name resnet50dropout --model_label BinaryPresent --classes_name binary_present --train_data_directory data/stratified_data/cv_True/split_0/train_data --vali_data_directory data/stratified_data/cv_True/split_0/vali_data --spectrogram_directory data/spectrograms/cv_True/split_0 --model_dir data/models/cv_True/split_0
python train_resnet.py --recalc_features --model_name resnet50dropout --model_label BinaryPresent --classes_name binary_present --train_data_directory data/stratified_data/cv_True/split_1/train_data --vali_data_directory data/stratified_data/cv_True/split_1/vali_data --spectrogram_directory data/spectrograms/cv_True/split_1 --model_dir data/models/cv_True/split_1
python train_resnet.py --recalc_features --model_name resnet50dropout --model_label BinaryPresent --classes_name binary_present --train_data_directory data/stratified_data/cv_True/split_2/train_data --vali_data_directory data/stratified_data/cv_True/split_2/vali_data --spectrogram_directory data/spectrograms/cv_True/split_2 --model_dir data/models/cv_True/split_2
python train_resnet.py --recalc_features --model_name resnet50dropout --model_label BinaryPresent --classes_name binary_present --train_data_directory data/stratified_data/cv_True/split_3/train_data --vali_data_directory data/stratified_data/cv_True/split_3/vali_data --spectrogram_directory data/spectrograms/cv_True/split_3 --model_dir data/models/cv_True/split_3
python train_resnet.py --recalc_features --model_name resnet50dropout --model_label BinaryPresent --classes_name binary_present --train_data_directory data/stratified_data/cv_True/split_4/train_data --vali_data_directory data/stratified_data/cv_True/split_4/vali_data --spectrogram_directory data/spectrograms/cv_True/split_4 --model_dir data/models/cv_True/split_4

python train_resnet.py --recalc_features --model_name resnet50dropout --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/stratified_data/cv_True/split_0/train_data --vali_data_directory data/stratified_data/cv_True/split_0/vali_data --spectrogram_directory data/spectrograms/cv_True/split_0 --model_dir data/models/cv_True/split_0
python train_resnet.py --recalc_features --model_name resnet50dropout --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/stratified_data/cv_True/split_1/train_data --vali_data_directory data/stratified_data/cv_True/split_1/vali_data --spectrogram_directory data/spectrograms/cv_True/split_1 --model_dir data/models/cv_True/split_1
python train_resnet.py --recalc_features --model_name resnet50dropout --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/stratified_data/cv_True/split_2/train_data --vali_data_directory data/stratified_data/cv_True/split_2/vali_data --spectrogram_directory data/spectrograms/cv_True/split_2 --model_dir data/models/cv_True/split_2
python train_resnet.py --recalc_features --model_name resnet50dropout --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/stratified_data/cv_True/split_3/train_data --vali_data_directory data/stratified_data/cv_True/split_3/vali_data --spectrogram_directory data/spectrograms/cv_True/split_3 --model_dir data/models/cv_True/split_3
python train_resnet.py --recalc_features --model_name resnet50dropout --model_label BinaryUnknown --classes_name binary_unknown --train_data_directory data/stratified_data/cv_True/split_4/train_data --vali_data_directory data/stratified_data/cv_True/split_4/vali_data --spectrogram_directory data/spectrograms/cv_True/split_4 --model_dir data/models/cv_True/split_4


# Run models
python dbres.py --recalc_output --data_directory data/stratified_data/cv_True/split_0/test_data --output_directory data/dbres_outputs/cv_True/split_0 --model_binary_present_pth data/models/cv_True/split_0/model_BinaryPresent.pth --model_binary_unknown_pth data/models/cv_True/split_0/model_BinaryUnknown.pth
python dbres.py --recalc_output --data_directory data/stratified_data/cv_True/split_1/test_data --output_directory data/dbres_outputs/cv_True/split_1 --model_binary_present_pth data/models/cv_True/split_1/model_BinaryPresent.pth --model_binary_unknown_pth data/models/cv_True/split_1/model_BinaryUnknown.pth
python dbres.py --recalc_output --data_directory data/stratified_data/cv_True/split_2/test_data --output_directory data/dbres_outputs/cv_True/split_2 --model_binary_present_pth data/models/cv_True/split_2/model_BinaryPresent.pth --model_binary_unknown_pth data/models/cv_True/split_2/model_BinaryUnknown.pth
python dbres.py --recalc_output --data_directory data/stratified_data/cv_True/split_3/test_data --output_directory data/dbres_outputs/cv_True/split_3 --model_binary_present_pth data/models/cv_True/split_3/model_BinaryPresent.pth --model_binary_unknown_pth data/models/cv_True/split_3/model_BinaryUnknown.pth
python dbres.py --recalc_output --data_directory data/stratified_data/cv_True/split_4/test_data --output_directory data/dbres_outputs/cv_True/split_4 --model_binary_present_pth data/models/cv_True/split_4/model_BinaryPresent.pth --model_binary_unknown_pth data/models/cv_True/split_4/model_BinaryUnknown.pth


# Combined
#python main.py --data_directory /Users/felixkrones/python_projects/data/physionet_challenge_2022/physionet.org/files/circor-heart-sound/1.0.3/training_data --vali_size 0.2 --test_size 0.2 --random_state 42 --recalc_features --model_name resnet50dropout --recalc_output


# Evaluate on pascal
#python dbres.py --recalc_output --recordings_file /Users/felixkrones/python_projects/data/physionet_challenge_2022/pascal/df_pascal.csv --output_directory data/pascal/dbres_outputs --model_binary_present_pth data/models/cv_True/split_0/model_BinaryPresent.pth --model_binary_unknown_pth data/models/cv_True/split_0/model_BinaryUnknown.pth
