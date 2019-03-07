import numpy as np
import operator


def others_class_regularizer(predictions, percent):
    all_records = len(predictions)

    found_amplifications = {1: 0, 2: 0, 3: 0}

    for i in range(3):
        counts = {}
        for others_amplification in np.arange(0, 1, step=0.01):
            new_predictions = []
            for prediction in predictions:
                clazz = np.argmax(prediction)
                new_prediction = list(prediction)
                if clazz == i + 1:
                    new_prediction[0] = new_prediction[0] + others_amplification
                new_predictions.append(new_prediction)

            count_for_class = 0
            classes = np.argmax(new_predictions, axis=1)
            for clazz in classes:
                if clazz == i + 1:
                    count_for_class += 1

            counts[others_amplification] = abs(percent - (count_for_class / all_records))

        found_amplifications[i + 1] = sorted(counts.items(), key=operator.itemgetter(1), reverse=False)[0][0]

    print(found_amplifications)

    regularized_predictions = []
    for prediction in predictions:
        clazz = np.argmax(prediction)
        new_prediction = list(prediction)
        if clazz > 0:
            new_prediction[0] = new_prediction[0] + found_amplifications[clazz]
        regularized_predictions.append(new_prediction)

    return np.array(regularized_predictions)
