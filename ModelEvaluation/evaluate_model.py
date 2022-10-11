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

from DataProcessing.find_and_load_patient_files import load_patient_data
from DataProcessing.helper_code import compare_strings
from DataProcessing.label_extraction import get_murmur


# Evaluate the models.
def evaluate_model(label_folder, output_probabilities, output_labels):
    # Define murmur and outcome classes.
    murmur_classes = ["Present", "Unknown", "Absent"]

    # Load and parse label and model output files.
    label_files = find_challenge_files(label_folder)
    murmur_labels = load_murmurs(label_files, murmur_classes)

    # For each patient, set the 'Present' or 'Abnormal' class to positive if no
    # class is positive or if multiple classes are positive.
    murmur_labels = enforce_positives(murmur_labels, murmur_classes, "Present")

    # Evaluate the murmur model by comparing the labels and model outputs.
    (
        murmur_auroc,
        murmur_auprc,
        murmur_auroc_classes,
        murmur_auprc_classes,
    ) = compute_auc(murmur_labels, output_probabilities)
    murmur_f_measure, murmur_f_measure_classes = compute_f_measure(
        murmur_labels, output_labels
    )
    murmur_accuracy, murmur_accuracy_classes = compute_accuracy(
        murmur_labels, output_labels
    )
    murmur_weighted_accuracy = compute_weighted_accuracy(
        murmur_labels, output_labels, murmur_classes
    )  # This is the murmur scoring metric.
    murmur_scores = (
        murmur_classes,
        murmur_auroc,
        murmur_auprc,
        murmur_auroc_classes,
        murmur_auprc_classes,
        murmur_f_measure,
        murmur_f_measure_classes,
        murmur_accuracy,
        murmur_accuracy_classes,
        murmur_weighted_accuracy,
    )

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
    elif classes == ["Abnormal", "Normal"]:
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
