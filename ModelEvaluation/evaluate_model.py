#!/usr/bin/env python
#
# BSD 2-Clause License
#
# Copyright (c) 2022 PhysioNet/Computing in Cardiology Challenges
# All rights reserved.
#
# Do *not* edit this script. Changes will be discarded so that we can process the models consistently.
#
# This file contains functions for evaluating models for the 2022 Challenge. You can run it as follows:
#
#   python evaluate_model.py labels outputs scores.csv
#
# where 'labels' is a folder containing files with the labels, 'outputs' is a folder containing
# files with the outputs from your model, and 'scores.csv' (optional) is a collection of
# scores for the model outputs.
#
# Each label or output file must have the format described on the Challenge webpage.
# The scores for the algorithm outputs include the area under the receiver-operating
# characteristic curve (AUROC), the area under the recall-precision curve (AUPRC), macro
# accuracy, a weighted accuracy score, and the Challenge score.

import os
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from DataProcessing.find_and_load_patient_files import load_patient_data
from DataProcessing.helper_code import compare_strings
from DataProcessing.label_extraction import get_murmur, get_outcome


# Evaluate the models.
def evaluate_model(label_folder, output_probabilities, output_labels, model_type, recordings_file: str="", output_directory: str=""):

    # Define murmur and outcome classes.
    if model_type == "murmur":
        class_options = ["Present", "Unknown", "Absent"]
        default_class = "Present"
    elif model_type == "outcome_binary":
        class_options = ["Abnormal", "Normal"]
        default_class = "Abnormal"
    elif model_type == "murmur_binary":
        class_options = ["Present", "Absent"]
        default_class = "Present"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Class options: {class_options} for model type {model_type}")

    if len(recordings_file) > 0:
        if model_type != "outcome_binary":
            raise NotImplementedError("Recordings file only supported for binary outcome classification.")
        print("Using recordings file for evaluation.")
        df_recordings = pd.read_csv(recordings_file)
        num_patients = len(df_recordings)
        num_classes = len(class_options)
        true_labels = np.zeros((num_patients, num_classes), dtype=np.bool_)
        for i in range(num_patients):
            label = df_recordings["label"].iloc[i]
            for j, x in enumerate(class_options):
                if compare_strings(label, x):
                    true_labels[i, j] = 1
    else:
        print("Using labels file for evaluation.")
        label_files = find_challenge_files(label_folder)
        if model_type == "murmur":
            true_labels = load_murmurs(label_files, class_options)
        elif model_type == "outcome_binary":
            true_labels = load_outcomes(label_files, class_options)
        elif model_type == "murmur_binary":
            true_labels = load_binary_murmurs(label_files, class_options)
        else:
            raise ValueError("Unknown model type: {}".format(model_type))

    # For each patient, set the 'Present' or 'Abnormal' class to positive if no
    # class is positive or if multiple classes are positive.
    true_labels = enforce_positives(true_labels, class_options, default_class)

    # Evaluate the murmur model by comparing the labels and model outputs.
    (
        auroc,
        auprc,
        auroc_classes,
        auprc_classes,
    ) = compute_auc(true_labels, output_probabilities)
    f_measure, f_measure_classes = compute_f_measure(
        true_labels, output_labels
    )
    accuracy, accuracy_classes = compute_accuracy(
        true_labels, output_labels
    )
    weighted_accuracy = compute_weighted_accuracy(
        true_labels, output_labels, class_options
    )  # This is the murmur scoring metric.
    confusion_matrix_ = compute_confusion_matrix(true_labels, output_labels)

    murmur_output_string = (
        "AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy"
        "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(
            auroc, auprc, f_measure, accuracy, weighted_accuracy
        )
    )
    confusion_matrix_string = (
        "Confusion Matrix\n"
        + ",".join(class_options)
        + "\n"
        + "\n".join(
            ",".join(str(x) for x in row) for row in confusion_matrix_
        )
        + "\n"
    )
    murmur_class_output_string = (
        "Classes,{}\nAUROC,{}\nAUPRC,{}\nF-measure,"
        "{}\nAccuracy,"
        "{}\n".format(
            ",".join(class_options),
            ",".join("{:.3f}".format(x) for x in auroc_classes),
            ",".join("{:.3f}".format(x) for x in auprc_classes),
            ",".join("{:.3f}".format(x) for x in f_measure_classes),
            ",".join("{:.3f}".format(x) for x in accuracy_classes),
        )
    )

    output_string = (
        "#Scores\n"
        + murmur_output_string
        + "\n#MScores (per class)\n"
        + murmur_class_output_string
        + "\n#Confusion Matrix\n"
        + confusion_matrix_string
    )

    # Create plots.
    if len(output_directory) > 0:
        # Define decision thresholds
        thresholds = np.linspace(0, 1, 100)

        accuracy_list = []
        fpr_list = []  # false positive rate
        fnr_list = []  # false negative rate

        # Check if both classes are contained. If not, add one row to true_labels that contains at position 0 True and at all other positions False.
        if (np.unique(true_labels[:, 0]).size == 1) and (true_labels[0, 0] == 0):
            true_labels_aux = np.vstack((true_labels, true_labels[-1]))
            true_labels_aux[-1, 0] = True
            true_labels_aux[-1, 1:] = False
            output_probabilities_aux = np.vstack((output_probabilities, output_probabilities[-1]))
            output_probabilities_aux[-1, 0] = True
            output_probabilities_aux[-1, 1:] = False
            print("Only one class contained in labels, adding dummy class 1.")
        else:
            true_labels_aux = true_labels
            output_probabilities_aux = output_probabilities

        # Calculate metrics for each threshold
        for threshold in thresholds:
            predicted_labels = output_probabilities_aux[:, 0]
            predicted_labels = (predicted_labels >= threshold).astype(int)

            # Create confusion matrix
            tn, fp, fn, tp = confusion_matrix(true_labels_aux[:, 0], predicted_labels).ravel()
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            fpr = fp / (fp + tn)  # false positive rate
            fnr = fn / (fn + tp)  # false negative rate

            # Add metrics to their respective lists
            accuracy_list.append(accuracy)
            fpr_list.append(fpr*-1)
            fnr_list.append(fnr*-1)

        # Generate the plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color = 'tab:blue'
        ax1.set_xlabel('Decision Threshold', fontsize=14)
        ax1.set_ylabel('Accuracy', color=color, fontsize=14)
        ax1.plot(thresholds, accuracy_list, color='tab:blue', label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color, labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('- False Positive Rate / - False Negative Rate', color=color, fontsize=14)
        ax2.plot(thresholds, fpr_list, color='tab:red', label='- False Positive Rate (FP / (FP + TN))')
        ax2.plot(thresholds, fnr_list, color='tab:orange', label='- False Negative Rate (FN / (FN + TP))')
        ax2.tick_params(axis='y', labelcolor=color, labelsize=12)

        fig.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust top margin
        plt.title('Metrics for different decision thresholds for the positive finding class', fontsize=16, pad=20)
        fig.legend(loc="center right", bbox_to_anchor=(0.95,0.5), bbox_transform=ax1.transAxes, fontsize=12)
        plt.grid(True)

        # Save the plots
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        plot_path_png = os.path.join(output_directory, "threshold_plot.png")
        plt.savefig(plot_path_png)
        plot_path_pdf = os.path.join(output_directory, "threshold_plot.pdf")
        plt.savefig(plot_path_pdf)

        # Close the figure
        plt.close()

    # Return the results.
    return output_string


# Find Challenge files.
def find_challenge_files(label_folder):
    label_files = list()
    for label_file in sorted(os.listdir(label_folder)):
        label_file_path = os.path.join(
            label_folder, label_file
        )  # Full path for label file
        if (
            os.path.isfile(label_file_path)
            and label_file.lower().endswith(".txt")
            and not label_file.lower().startswith(".")
        ):
            label_files.append(label_file_path)

    if label_files:
        return label_files
    else:
        raise IOError("No label or output files found.")


# Load murmurs from label files.
def load_murmurs(label_files, classes):
    num_patients = len(label_files)
    num_classes = len(classes)

    # Use one-hot encoding for the labels.
    labels = np.zeros((num_patients, num_classes), dtype=np.bool_)

    # Iterate over the patients.
    for i in range(num_patients):
        data = load_patient_data(label_files[i])
        label = get_murmur(data)
        for j, x in enumerate(classes):
            if compare_strings(label, x):
                labels[i, j] = 1

    return labels


# Load binary murmurs from label files.
def load_binary_murmurs(label_files, classes):
    labels = load_murmurs(label_files, ["Present", "Unknown", "Absent"])
    
    # Check if "Present" in classes is at index 0
    if classes[0] != "Present":
        raise ValueError("The first class must be 'Present'")

    # Combine Present and Unknown into a single class
    labels[:, 0] = np.logical_or(labels[:, 0], labels[:, 1])
    labels = np.delete(labels, 1, 1)

    return labels


# Load outcomes from label files.
def load_outcomes(label_files, classes):
    num_patients = len(label_files)
    num_classes = len(classes)

    # Use one-hot encoding for the labels.
    labels = np.zeros((num_patients, num_classes), dtype=np.bool_)

    # Iterate over the patients.
    for i in range(num_patients):
        data = load_patient_data(label_files[i])
        label = get_outcome(data)
        for j, x in enumerate(classes):
            if compare_strings(label, x):
                labels[i, j] = 1

    return labels


# For each patient, set a specific class to positive if no class is positive or multiple classes are positive.
def enforce_positives(outputs, classes, positive_class):
    num_patients, num_classes = np.shape(outputs)
    j = classes.index(positive_class)

    for i in range(num_patients):
        if np.sum(outputs[i, :]) != 1:
            outputs[i, :] = 0
            outputs[i, j] = 1
    return outputs


# Compute macro AUROC and macro AUPRC.
def compute_auc(labels, outputs):
    num_patients, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_patients and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float("nan")
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float("nan")
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float("nan")

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    # Compute macro AUROC and macro AUPRC across classes.
    if np.any(np.isfinite(auroc)):
        macro_auroc = np.nanmean(auroc)
    else:
        macro_auroc = float("nan")
    if np.any(np.isfinite(auprc)):
        macro_auprc = np.nanmean(auprc)
    else:
        macro_auprc = float("nan")

    return macro_auroc, macro_auprc, auroc, auprc


# Compute a binary confusion matrix, where the columns are the expert labels and
# the rows are the classifier labels.
def compute_confusion_matrix(labels, outputs):
    assert np.shape(labels)[0] == np.shape(outputs)[0]
    assert all(value in (0, 1, True, False) for value in np.unique(labels))
    assert all(value in (0, 1, True, False) for value in np.unique(outputs))

    num_patients = np.shape(labels)[0]
    num_label_classes = np.shape(labels)[1]
    num_output_classes = np.shape(outputs)[1]

    A = np.zeros((num_output_classes, num_label_classes))
    for k in range(num_patients):
        for i in range(num_output_classes):
            for j in range(num_label_classes):
                if outputs[k, i] == 1 and labels[k, j] == 1:
                    A[i, j] += 1

    return A


# Compute binary one-vs-rest confusion matrices, where the columns are the expert
# labels and the rows are the classifier labels.
def compute_one_vs_rest_confusion_matrix(labels, outputs):
    assert np.shape(labels) == np.shape(outputs)
    assert all(value in (0, 1, True, False) for value in np.unique(labels))
    assert all(value in (0, 1, True, False) for value in np.unique(outputs))

    num_patients, num_classes = np.shape(labels)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_patients):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                A[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                A[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                A[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                A[j, 1, 1] += 1

    return A


# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    num_patients, num_classes = np.shape(labels)

    A = compute_one_vs_rest_confusion_matrix(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, _ = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float("nan")

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float("nan")

    return macro_f_measure, f_measure


# Compute accuracy.
def compute_accuracy(labels, outputs):
    # Compute confusion matrix.
    assert np.shape(labels) == np.shape(outputs)
    num_patients, num_classes = np.shape(labels)
    A = compute_confusion_matrix(labels, outputs)

    # Compute accuracy.
    if np.sum(A) > 0:
        accuracy = np.trace(A) / np.sum(A)
    else:
        accuracy = float("nan")

    # Compute per-class accuracy.
    accuracy_classes = np.zeros(num_classes)
    for i in range(num_classes):
        if np.sum(A[:, i]) > 0:
            accuracy_classes[i] = A[i, i] / np.sum(A[:, i])
        else:
            accuracy_classes[i] = float("nan")

    return accuracy, accuracy_classes


# Compute accuracy.
def compute_weighted_accuracy(labels, outputs, classes):
    # Define constants.
    if classes == ["Present", "Unknown", "Absent"]:
        weights = np.array([[5, 3, 1], [5, 3, 1], [5, 3, 1]])
    elif (classes == ["Abnormal", "Normal"]) or (classes == ["Present", "Absent"]):
        weights = np.array([[5, 1], [5, 1]])
    else:
        raise NotImplementedError(
            "Weighted accuracy undefined for classes {}".format(", ".join(classes))
        )

    # Compute confusion matrix.
    assert np.shape(labels) == np.shape(outputs)
    A = compute_confusion_matrix(labels, outputs)

    # Multiply the confusion matrix by the weight matrix.
    assert np.shape(A) == np.shape(weights)
    B = weights * A

    # Compute weighted_accuracy.
    if np.sum(B) > 0:
        weighted_accuracy = np.trace(B) / np.sum(B)
    else:
        weighted_accuracy = float("nan")

    return weighted_accuracy


# Define total cost for algorithmic prescreening of m patients.
def cost_algorithm(m):
    return 10 * m


# Define total cost for expert screening of m patients out of a total of n total patients.
def cost_expert(m, n):
    return (25 + 397 * (m / n) - 1718 * (m / n) ** 2 + 11296 * (m / n) ** 4) * n


# Define total cost for treatment of m patients.
def cost_treatment(m):
    return 10000 * m


# Define total cost for missed/late treatement of m patients.
def cost_error(m):
    return 50000 * m


# Compute Challenge cost metric.
def compute_cost(labels, outputs, label_classes):
    # Define positive and negative classes for referral and treatment.
    positive_classes = ["Present", "Unknown", "Abnormal"]
    negative_classes = ["Absent", "Normal"]

    # Compute confusion matrix.
    A = compute_confusion_matrix(labels, outputs)

    # Identify positive and negative classes for referral.
    idx_label_positive = [
        i for i, x in enumerate(label_classes) if x in positive_classes
    ]
    idx_label_negative = [
        i for i, x in enumerate(label_classes) if x in negative_classes
    ]
    idx_output_positive = [
        i for i, x in enumerate(label_classes) if x in positive_classes
    ]
    idx_output_negative = [
        i for i, x in enumerate(label_classes) if x in negative_classes
    ]

    # Identify true positives, false positives, false negatives, and true negatives.
    tp = np.sum(A[np.ix_(idx_output_positive, idx_label_positive)])
    fp = np.sum(A[np.ix_(idx_output_positive, idx_label_negative)])
    fn = np.sum(A[np.ix_(idx_output_negative, idx_label_positive)])
    tn = np.sum(A[np.ix_(idx_output_negative, idx_label_negative)])
    total_patients = tp + fp + fn + tn

    # Compute total cost for all patients.
    total_cost = (
        cost_algorithm(total_patients)
        + cost_expert(tp + fp, total_patients)
        + cost_treatment(tp)
        + cost_error(fn)
    )

    # Compute mean cost per patient.
    if total_patients > 0:
        mean_cost = total_cost / total_patients
    else:
        mean_cost = float("nan")

    return mean_cost


def run_model_evaluate(label_folder, output_folder, output_file):
    murmur_scores = evaluate_model(label_folder, output_folder)

    (
        classes,
        auroc,
        auprc,
        auroc_classes,
        auprc_classes,
        f_measure,
        f_measure_classes,
        accuracy,
        accuracy_classes,
        weighted_accuracy,
    ) = murmur_scores
    murmur_output_string = (
        "AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy"
        "\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(
            auroc, auprc, f_measure, accuracy, weighted_accuracy
        )
    )
    murmur_class_output_string = (
        "Classes,{}\nAUROC,{}\nAUPRC,{}\nF-measure,"
        "{}\nAccuracy,"
        "{}\n".format(
            ",".join(classes),
            ",".join("{:.3f}".format(x) for x in auroc_classes),
            ",".join("{:.3f}".format(x) for x in auprc_classes),
            ",".join("{:.3f}".format(x) for x in f_measure_classes),
            ",".join("{:.3f}".format(x) for x in accuracy_classes),
        )
    )

    output_string = (
        "#Murmur scores\n"
        + murmur_output_string
        + "\n#Murmur scores (per class)\n"
        + murmur_class_output_string
    )

    if output_file is not None:
        with open(output_file, "w") as f:
            f.write(output_string)
    else:
        print(output_string)
