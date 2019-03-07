import io
import numpy as np
from abc import abstractmethod, ABC

from keras import Input
from keras.layers import Embedding

from custom_layers import ELMoLayer


class EmbeddingLayer(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_input(self):
        raise NotImplementedError

    @abstractmethod
    def get_output(self):
        raise NotImplementedError


class StaticEmbeddingLayer(EmbeddingLayer):
    def __init__(self, name, file_path, max_sequence_length, word_index, unk_token):
        super().__init__(name)
        self.file_path = file_path
        self.max_sequence_length = max_sequence_length
        self.word_index = word_index
        self.unk_token = unk_token

    def _get_embedding_matrix(self):
        embedding_index = {}
        dim = 0
        mapping = self.mapping()

        with io.open(self.file_path, encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                dim = len(values) - 1
                embedding_index[word] = np.asarray(values[1:], dtype='float32')

        embedding_matrix = np.zeros((len(self.word_index) + 1, dim))
        for word, i in self.word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is None:
                if mapping is not None and word in mapping:
                    embedding_vector = embedding_index.get(mapping[word])
                else:
                    embedding_vector = embedding_index.get(self.unk_token)

            embedding_matrix[i] = embedding_vector

        return embedding_matrix

    def get_input(self):
        return Input(shape=(self.max_sequence_length,), dtype='int32', name=f'input_{self.name}')

    def get_output(self):
        embedding_matrix = self._get_embedding_matrix()
        return Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                         weights=[embedding_matrix],
                         input_length=self.max_sequence_length,
                         trainable=False)

    @staticmethod
    def mapping():
        return None


class GloveEmbeddingLayer(StaticEmbeddingLayer):
    def __init__(self, file_path, max_sequence_length, word_index):
        super().__init__('glove', file_path, max_sequence_length, word_index, '<unknown>')


class Word2VecEmbeddingLayer(StaticEmbeddingLayer):
    def __init__(self, file_path, max_sequence_length, word_index):
        super().__init__('word2vec', file_path, max_sequence_length, word_index, '<unknown>')

    @staticmethod
    def mapping():
        return {
            ':*': '<kiss>',
            ':)': '<happy>',
            ':d': '<laugh>',
            ':(': '<sad>',
            ':|': '<sad>',
            ':O': '<surprise>',
            ';)': '<wink>',
            ':p': '<tong>',
            ':/': '<annoyed>',
            '<3': '<heart>'
        }


class SSWEEmbeddingLayer(StaticEmbeddingLayer):
    def __init__(self, file_path, max_sequence_length, word_index):
        super().__init__('sswe', file_path, max_sequence_length, word_index, '<unk>')


class Emo2VecEmbeddingLayer(StaticEmbeddingLayer):
    def __init__(self, file_path, max_sequence_length, word_index):
        super().__init__('emo2vec', file_path, max_sequence_length, word_index, 'UNK')

    @staticmethod
    def mapping():
        return {
            '<repeated>': '<repeat>',
            '</allcaps>': '<allcaps>',
            '<elongated>': '<elong>',
            '</hashtag>': '<hashtag>',
            ':)': '<smile>',
            '😂': '<lolface>',
            '😭': '<sadface>',
            ':(': '<sadface>',
            ':d': '<smile>',
            ':p': '<lolface>',
            '😞': '<sadface>',
            '😁': '<smile>',
            '😢': '<sadface>',
            '😡': 'angry',
            ';)': '<smile>',
            '😍': 'love',
            '2': '<number>',
            '😀': '<smile>',
            '😆': '<smile>',
            '😅': '<smile>',
            '😄': '<smile>',
            ':/': '<neutralface>',
            '1': '<number>',
            '😃': '<smile>',
            '💔': 'sad',
            '😊': '<smile>',
            '😘': 'love',
            '😒': '<neutralface>',
            '3': '<number>',
            '😠': 'angry',
            '<3': '<heart>',
            '😤': '<smile>',
            '😩': '☹',
            '😹': '<lolface>',
            '😉': '<smile>',
            '😌': 'relieve',
            '😜': '<lolface>',
            '4': '<number>',
            '😫': 'tired',
            '😺': '<smile>',
            '5': '<number>',
            '👍': 'ok',
            '😋': '<lolface>',
            '😸': '<smile>',
            '❤️': '❤',
            '😻': '<smile>',
            '😑': '<neutralface>'
        }


class DynamicEmbeddingLayer(EmbeddingLayer):
    def __init__(self, name, max_sequence_length, model_dir):
        super().__init__(name)
        self.max_sequence_length = max_sequence_length
        self.model_dir = model_dir

    @abstractmethod
    def get_input(self):
        raise NotImplementedError

    @abstractmethod
    def get_output(self):
        raise NotImplementedError


class ELMoEmbeddingLayer(DynamicEmbeddingLayer):
    def __init__(self, max_sequence_length, model_dir):
        super().__init__('elmo', max_sequence_length, model_dir)

    def get_input(self):
        return Input(shape=(1,), dtype="string", name=f'input_{self.name}')

    def get_output(self):
        return ELMoLayer(self.max_sequence_length)


static_embedding_layers = {
    'word2vec': Word2VecEmbeddingLayer,
    'glove': GloveEmbeddingLayer,
    'sswe': SSWEEmbeddingLayer,
    'emo2vec': Emo2VecEmbeddingLayer
}

dynamic_embedding_layers = {
    'elmo': ELMoEmbeddingLayer
}


def create_static_embedding_layer(name, file_path, max_sequence_length, word_index):
    return static_embedding_layers[name](file_path, max_sequence_length, word_index)


def create_dynamic_embedding_layer(name, max_sequence_length, model_dir):
    return dynamic_embedding_layers[name](max_sequence_length, model_dir)
