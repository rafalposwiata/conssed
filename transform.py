import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences


class TokensToNumbers:
    def __init__(self, limit=None):
        self.word_index = {}
        self.idx = 1
        self.limit = limit

    def fit_on_tokens(self, lists_of_tokens):
        for tokens in lists_of_tokens:
            for token in tokens:
                if self.limit is not None and self.idx == self.limit:
                    break
                if token not in self.word_index:
                    self.word_index[token] = self.idx
                    self.idx += 1

    def transform(self, transform_type, lists_of_tokens, max_sequence_length):
        if transform_type == 'tokens':
            return self.tokens_to_sequences(lists_of_tokens, max_sequence_length)
        elif transform_type == 'string':
            return self.tokens_to_string(lists_of_tokens, max_sequence_length)

    def tokens_to_sequences(self, lists_of_tokens, max_sequence_length):
        result = []
        for tokens in lists_of_tokens:
            numbers = []
            for token in tokens:
                if token in self.word_index:
                    numbers.append(self.word_index[token])
            result.append(numbers)
        return pad_sequences(result, maxlen=max_sequence_length)

    @staticmethod
    def tokens_to_string(lists_of_tokens, max_sequence_length, pad='_PAD_'):
        result = []
        for tokens in lists_of_tokens:
            padded_tokens = []
            for i in range(max_sequence_length):
                try:
                    padded_tokens.append(tokens[i])
                except:
                    padded_tokens.append(pad)
            result.append(' '.join(np.array(padded_tokens)))
        return np.array(result)

    def save(self, word_index_file='word_index.pkl'):
        pickle.dump(self.word_index, open(word_index_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, word_index_file='word_index.pkl'):
        self.word_index = pickle.load(open(word_index_file, 'rb'))
        self.idx = len(word_index_file) + 1


if __name__ == '__main__':
    lists_of_tokens = [['I', 'am', 'free'], ['Hey', '!']]
    transformer = TokensToNumbers()
    transformer.fit_on_tokens(lists_of_tokens)
    word_index = transformer.word_index
    transformer.save()

    transformer = TokensToNumbers()
    transformer.load()
    assert word_index == transformer.word_index
