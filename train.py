import os
import sys
from structures import Config, load_configuration
from transform import TokensToNumbers
from dataloader import load_data_and_labels, load_data_and_labels_for_conssed
from model import ConSSEDModel, BaselineModel, BiLSTMModel

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(config: Config):
    transformer = TokensToNumbers()
    transformer.load(config.word_index_file)

    if config.model == 'ConSSED':
        train_sem, train_sen, train_labels = load_data_and_labels_for_conssed(config, transformer, 'train')
        validation_sem, validation_sen, validation_labels = load_data_and_labels_for_conssed(config, transformer,
                                                                                             'validation')

        model = ConSSEDModel(config, transformer.word_index)
        model.fit(train_sem, train_sen, train_labels, validation_sem, validation_sen, validation_labels)
    else:
        train_data, train_labels = load_data_and_labels(config, transformer, 'train')
        validation_data, validation_labels = load_data_and_labels(config, transformer, 'validation')

        model = (BiLSTMModel if config.model == 'BiLSTM' else BaselineModel)(config, transformer.word_index)
        model.fit(train_data, train_labels, validation_data, validation_labels)

    return 1 - model.f1_score()


if __name__ == '__main__':
    config = load_configuration(str(sys.argv[1]))

    train(config)
