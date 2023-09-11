import numpy as np
from sklearn.metrics import confusion_matrix


def get_confusion_matrix(mask, prediction, threshold=None):
    conf_mat = confusion_matrix(mask, prediction)
    return conf_mat


def get_dice_coefficient(confusion_matrix: np.ndarray):
    TP, FP, TN, FN = get_tuple(confusion_matrix)
    return 2 * TP / (2 * TP + FN + FP)


def get_jaccard_index(confusion_matrix: np.ndarray):
    TP, FP, TN, FN = get_tuple(confusion_matrix)
    return TP / (TP + FN + FP)


def get_sensitivity(confusion_matrix: np.ndarray):
    TP, FP, TN, FN = get_tuple(confusion_matrix)
    return TP / (TP + FN)


def get_specificity(confusion_matrix: np.ndarray):
    TP, FP, TN, FN = get_tuple(confusion_matrix)
    return TN / (TN + FP)


def get_accuracy(confusion_matrix: np.ndarray):
    TP, FP, TN, FN = get_tuple(confusion_matrix)
    return (TP + TN) / tf.reduce_sum(confusion_matrix)


def get_tuple(confusion_matrix):
    """
    return: TP,FP,TN,FN
    """
    return confusion_matrix[1][1], confusion_matrix[0][1], confusion_matrix[0][0], confusion_matrix[1][0]
