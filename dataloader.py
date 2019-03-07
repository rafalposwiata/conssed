import numpy as np
from keras.utils import to_categorical
from structures import Config
from transform import TokensToNumbers
from utils import read_data_file, emotion2label
from preprocess import pre_process


def load_data(config: Config, transformer: TokensToNumbers, type):
    lines = read_data_file(config.get_data_path(type))
    tokens = pre_process(lines, config.baseline_config.embedding_config.pre_processing)
    return transformer.transform(config.baseline_config.embedding_config.input_type, tokens, config.max_sequence_length)


def load_data_for_conssed(config: Config, transformer: TokensToNumbers, type):
    lines = read_data_file(config.get_data_path(type))
    data = []
    for part_config in [config.conssed_config.semantic_part, config.conssed_config.sentiment_part]:
        tokens = pre_process(lines, part_config.embedding_config.pre_processing)
        data.append(transformer.transform(part_config.embedding_config.input_type, tokens, config.max_sequence_length))

    sem_data, sen_data = data
    return sem_data, sen_data


def load_labels(config, type):
    lines = read_data_file(config.get_data_path(type))
    labels = [emotion2label[line[4]] for line in lines]
    return to_categorical(np.asarray(labels))


def load_data_and_labels(config, transformer, type):
    data = load_data(config, transformer, type)
    labels = load_labels(config, type)
    return data, labels


def load_data_and_labels_for_conssed(config, transformer, type):
    sem_data, sen_data = load_data_for_conssed(config, transformer, type)
    labels = load_labels(config, type)
    return sem_data, sen_data, labels
