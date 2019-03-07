import os
import sys
import numpy as np
from structures import Config, load_configuration
from transform import TokensToNumbers
from utils import create_solution, format_metrics
from dataloader import load_data_for_conssed, load_data, load_labels
from model import ConSSEDModel, BaselineModel, BiLSTMModel
from metrics import calculate_f1_metrics

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(config: Config):
    transformer = TokensToNumbers()
    transformer.load(config.word_index_file)

    if config.model == 'ConSSED':
        model = ConSSEDModel(config, transformer.word_index)
        model.load_weights()
        test_sem, test_sen = load_data_for_conssed(config, transformer, 'test')
        predictions, predictions_without_others_class_regularizer = model.predict(test_sem, test_sen)
    else:
        model = (BiLSTMModel if config.model == 'BiLSTM' else BaselineModel)(config, transformer.word_index)
        model.load_weights()
        test = load_data(config, transformer, 'test')
        predictions = model.predict(test)
        predictions_without_others_class_regularizer = None

    create_solution(predictions, config.test_data_path)
    if config.metrics_summary:
        show_metrics_summary(predictions, predictions_without_others_class_regularizer, config)


def show_metrics_summary(predictions, regularized_predictions, config: Config):
    ground_true = load_labels(config, 'test')

    happy_f1, sad_f1, angry_f1, micro_f1 = calculate_f1_metrics(np.array(predictions), ground_true)
    print(f'F1 scores: {format_metrics([happy_f1, sad_f1, angry_f1, micro_f1])}')

    if regularized_predictions is not None:
        happy_f1, sad_f1, angry_f1, micro_f1 = calculate_f1_metrics(np.array(regularized_predictions), ground_true)
        print(f'F1 scores (without Others Class Regularizer): {format_metrics([happy_f1, sad_f1, angry_f1, micro_f1])}')


if __name__ == '__main__':
    config = load_configuration(str(sys.argv[1]))

    main(config)
