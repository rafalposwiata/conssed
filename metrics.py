import numpy as np
from keras.utils import to_categorical


def calculate_metrics(predictions, ground):
    accuracy = calculate_accuracy(predictions, ground)
    micro_precision, micro_recall, micro_f1 = calculate_micro_metrics(predictions, ground)

    return accuracy, micro_precision, micro_recall, micro_f1


def calculate_f1_metrics(predictions, ground):
    happy_f1, sad_f1, angry_f1 = calculte_macro_f1_for_classes(predictions, ground)
    micro_precision, micro_recall, micro_f1 = calculate_micro_metrics(predictions, ground)

    return happy_f1, sad_f1, angry_f1, micro_f1


def calculate_accuracy(predictions, ground):
    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    return np.mean(predictions == ground)


def calculate_micro_metrics(predictions, ground, num_classes=4):
    true_positives, false_positives, false_negatives = sensitivity_and_specificity(predictions, ground, num_classes)

    true_positives = true_positives[1:].sum()
    false_positives = false_positives[1:].sum()
    false_negatives = false_negatives[1:].sum()

    micro_precision = true_positives / (true_positives + false_positives)
    micro_recall = true_positives / (true_positives + false_negatives)

    micro_f1 = (2 * micro_recall * micro_precision) / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0 else 0

    return micro_precision, micro_recall, micro_f1


def calculte_macro_f1_for_classes(predictions, ground, num_classes=4):
    true_positives, false_positives, false_negatives = sensitivity_and_specificity(predictions, ground, num_classes)

    result = []
    for c in range(1, num_classes):
        precision = true_positives[c] / (true_positives[c] + false_positives[c])
        recall = true_positives[c] / (true_positives[c] + false_negatives[c])
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        result.append(f1)

    return result


def sensitivity_and_specificity(predictions, ground, num_classes=4):
    discrete_predictions = to_categorical(predictions.argmax(axis=1), num_classes=num_classes)

    true_positives = np.sum(discrete_predictions * ground, axis=0)
    false_positives = np.sum(np.clip(discrete_predictions - ground, 0, 1), axis=0)
    false_negatives = np.sum(np.clip(ground - discrete_predictions, 0, 1), axis=0)

    return true_positives, false_positives, false_negatives