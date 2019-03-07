import json


class JsonSerializable:
    def __str__(self):
        result = {}
        for attr, value in self.__dict__.items():
            if isinstance(value, RecurrentNetworkConfig):
                obj = {}
                for attr_2, value_2 in value.__dict__.items():
                    if isinstance(value_2, EmbeddingConfig):
                        obj[attr_2] = value_2.__dict__
                    else:
                        obj[attr_2] = value_2
                result[attr] = obj
            else:
                result[attr] = value
        return json.dumps(result, ensure_ascii=False, indent=1)

    def __repr__(self):
        return self.__str__()


class Config:
    def __init__(self, name, model, dense_dim, dropout, recurrent_dropout, learning_rate, others_class_weight,
                 others_class_regularizer_param, batch_size, epochs=10, max_sequence_length=200, num_classes=4,
                 verbose=1, metrics_summary=False, word_index_file=None, model_file_path=None, checkpoint_dir=None,
                 results_file_path=None):
        self.name = name
        self.model = model
        self.baseline_config: BaselineConfig = None
        self.conssed_config: ConSSEDConfig = None

        self.dense_dim = dense_dim
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.others_class_weight = others_class_weight
        self.others_class_regularizer_param = others_class_regularizer_param

        self.batch_size = batch_size
        self.epochs = epochs
        self.max_sequence_length = max_sequence_length
        self.num_classes = num_classes
        self.verbose = verbose
        self.metrics_summary = metrics_summary

        self.word_index_file = word_index_file
        self.model_file_path = model_file_path
        self.checkpoint_dir = checkpoint_dir
        self.results_file_path = results_file_path

        self.train_data_path = None
        self.validation_data_path = None
        self.test_data_path = None

    def set_baseline_config(self, baseline_config):
        self.baseline_config = baseline_config

    def set_conssed_config(self, conssed_config):
        self.conssed_config = conssed_config

    def configure_data_sets(self, train_data_path=None, validation_data_path=None, test_data_path=None):
        self.train_data_path = train_data_path
        self.validation_data_path = validation_data_path
        self.test_data_path = test_data_path

    def get_data_path(self, type):
        if type == 'train':
            return self.train_data_path
        elif type == 'validation':
            return self.validation_data_path
        elif type == 'test':
            return self.test_data_path
        else:
            raise Exception(f'Not valid data type: {type}. Valid data types: train, validation, test.')

    def __str__(self):
        result = {}
        for attr, value in self.__dict__.items():
            if isinstance(value, BaselineConfig) or isinstance(value, ConSSEDConfig):
                result[attr] = value.__str__()
            else:
                result[attr] = value
        return json.dumps(result, ensure_ascii=False, indent=1)

    def __repr__(self):
        return self.__str__()


class BaselineConfig:
    def __init__(self, embedding, embedding_file_path, lstm_dim, dynamic=False, model_dir=None,
                 pre_processing='default', input_type='tokens'):
        self.embedding_config = EmbeddingConfig(embedding, embedding_file_path, dynamic, model_dir,
                                                pre_processing, input_type)
        self.lstm_dim = lstm_dim

    def __str__(self):
        result = {}
        for attr, value in self.__dict__.items():
            if isinstance(value, EmbeddingConfig):
                result[attr] = value.__str__()
            else:
                result[attr] = value
        return result

    def __repr__(self):
        return self.__str__()


class ConSSEDConfig:
    def __init__(self):
        self.semantic_part: RecurrentNetworkConfig = None
        self.sentiment_part: RecurrentNetworkConfig = None

    def configure_part(self, part_type, embedding, embedding_file_path, dynamic, model_dir, lstm_dim,
                       first_bidirectional, second_bidirectional, pre_processing='default', input_type='tokens'):
        recurrent_network_config = RecurrentNetworkConfig(part_type, embedding, embedding_file_path, dynamic, model_dir,
                                                          lstm_dim,
                                                          first_bidirectional, second_bidirectional, pre_processing,
                                                          input_type)

        if part_type == 'semantic':
            self.semantic_part = recurrent_network_config
        elif part_type == 'sentiment':
            self.sentiment_part = recurrent_network_config

    def __str__(self):
        result = {}
        for attr, value in self.__dict__.items():
            if isinstance(value, RecurrentNetworkConfig):
                result[attr] = value.__str__()
            else:
                result[attr] = value
        return result

    def __repr__(self):
        return self.__str__()


class RecurrentNetworkConfig:
    def __init__(self, name, embedding, embedding_file_path, dynamic, model_dir, lstm_dim, first_bidirectional,
                 second_bidirectional, pre_processing, input_type):
        self.embedding_config = EmbeddingConfig(embedding, embedding_file_path, dynamic, model_dir, pre_processing,
                                                input_type)
        self.name = name
        self.lstm_dim = lstm_dim
        self.first_bidirectional = first_bidirectional
        self.second_bidirectional = second_bidirectional

    def __str__(self):
        result = {}
        for attr, value in self.__dict__.items():
            if isinstance(value, EmbeddingConfig):
                result[attr] = value.__str__()
            else:
                result[attr] = value
        return result

    def __repr__(self):
        return self.__str__()


class EmbeddingConfig:
    def __init__(self, embedding, embedding_file_path, dynamic, model_dir, pre_processing, input_type):
        self.embedding = embedding
        self.embedding_file_path = embedding_file_path
        self.dynamic = dynamic
        self.model_dir = model_dir
        self.pre_processing = pre_processing
        self.input_type = input_type

    def __str__(self):
        return self.__dict__

    def __repr__(self):
        return self.__str__()


def load_configuration(file_path) -> Config:
    with open(file_path) as config_file:
        obj = json.load(config_file)
    config = Config(name=obj['name'],
                    model=obj['model'],
                    dense_dim=obj['dense_dim'],
                    dropout=obj['dropout'],
                    recurrent_dropout=obj['recurrent_dropout'],
                    learning_rate=obj['learning_rate'],
                    others_class_weight=obj['others_class_weight'],
                    others_class_regularizer_param=obj['others_class_regularizer_param'],
                    batch_size=obj['batch_size'],
                    epochs=obj['epochs'],
                    max_sequence_length=obj['max_sequence_length'],
                    num_classes=obj['num_classes'],
                    verbose=obj['verbose'],
                    metrics_summary=obj['metrics_summary'],
                    word_index_file=obj['word_index_file'],
                    model_file_path=obj['model_file_path'],
                    checkpoint_dir=obj['checkpoint_dir'],
                    results_file_path=obj['results_file_path'])

    config.train_data_path = obj['train_data_path']
    config.validation_data_path = obj['validation_data_path']
    config.test_data_path = obj['test_data_path']

    if obj['conssed_config'] is not None:
        conssed_config = ConSSEDConfig()
        for conssed_part_obj in [obj['conssed_config']['semantic_part'], obj['conssed_config']['sentiment_part']]:
            conssed_config.configure_part(part_type=conssed_part_obj['name'],
                                          embedding=conssed_part_obj['embedding_config']['embedding'],
                                          embedding_file_path=conssed_part_obj['embedding_config'][
                                              'embedding_file_path'],
                                          dynamic=conssed_part_obj['embedding_config']['dynamic'],
                                          model_dir=conssed_part_obj['embedding_config']['model_dir'],
                                          pre_processing=conssed_part_obj['embedding_config']['pre_processing'],
                                          input_type=conssed_part_obj['embedding_config']['input_type'],
                                          lstm_dim=conssed_part_obj['lstm_dim'],
                                          first_bidirectional=conssed_part_obj['first_bidirectional'],
                                          second_bidirectional=conssed_part_obj['second_bidirectional'])
        config.set_conssed_config(conssed_config)

    if obj['baseline_config'] is not None:
        baseline_config_obj = obj['baseline_config']
        baseline_config = BaselineConfig(embedding=baseline_config_obj['embedding_config']['embedding'],
                                         embedding_file_path=baseline_config_obj['embedding_config'][
                                             'embedding_file_path'],
                                         dynamic=baseline_config_obj['embedding_config']['dynamic'],
                                         model_dir=baseline_config_obj['embedding_config']['model_dir'],
                                         pre_processing=baseline_config_obj['embedding_config']['pre_processing'],
                                         input_type=baseline_config_obj['embedding_config']['input_type'],
                                         lstm_dim=baseline_config_obj['lstm_dim'])
        config.set_baseline_config(baseline_config)

    return config


if __name__ == '__main__':
    config = Config(name="ConSSED_Word2Vec_Emo2Vec", model="ConSSED", dense_dim=150, dropout=0.3, recurrent_dropout=0.3,
                    learning_rate=0.003,
                    others_class_weight=2,
                    others_class_regularizer_param=0.055, batch_size=100, word_index_file='word_index.pkl',
                    model_file_path='model.hdf5')

    conssed_config = ConSSEDConfig()
    conssed_config.configure_part('semantic', 'word2vec', 'word2vec.txt', False, None, 320, True, False)
    conssed_config.configure_part('sentiment', 'emo2vec', 'emo2vec.txt', False, None, 256, True, True)
    config.set_conssed_config(conssed_config)

    baseline_config = BaselineConfig('word2vec', 'word2vec.txt', 234, False, None)
    config.set_baseline_config(baseline_config)

    config.configure_data_sets(test_data_path='test.txt')

    print(config)
