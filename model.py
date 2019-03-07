from abc import abstractmethod

from keras.callbacks import ModelCheckpoint, Callback

from metrics import calculate_metrics
from regularization import others_class_regularizer
from structures import Config, RecurrentNetworkConfig, EmbeddingConfig
from keras.layers import Dense, LSTM, Bidirectional, Concatenate
from keras import optimizers, Model
from embeddings import create_static_embedding_layer, create_dynamic_embedding_layer
from utils import create_unique_id, create_directory, append_to_file


class AbstractF1ScoreCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.best_f1 = 0
        self.best_accuracy = 0
        self.metrics = []

    def on_epoch_end(self, epoch, logs=None):
        micro_f1, accuracy = self.predict()

        if micro_f1 > self.best_f1:
            self.best_f1 = micro_f1
            self.best_accuracy = accuracy

        return

    @abstractmethod
    def predict(self):
        raise NotImplementedError


class ConSSEDF1ScoreCallback(AbstractF1ScoreCallback):
    def predict(self):
        validation_data = self.validation_data
        data = [validation_data[0], validation_data[1]]
        true_labels = validation_data[2]
        predictions = self.model.predict(data, batch_size=self.config.batch_size)

        output = ''

        accuracy, micro_precision, micro_recall, micro_f1 = calculate_metrics(predictions, true_labels)
        output += f'{micro_f1}, {accuracy}, '
        print(f'val_f1: {micro_f1}, acc: {accuracy}')

        regularized_predictions = others_class_regularizer(predictions, self.config.others_class_regularizer_param)
        accuracy, micro_precision, micro_recall, micro_f1 = calculate_metrics(regularized_predictions, true_labels)
        output += f'{micro_f1}, {accuracy}'
        print(f'val_f1 after regularization: {micro_f1}, acc after regularization: {accuracy}')

        self.metrics.append(output)

        return micro_f1, accuracy


class BaseF1ScoreCallback(AbstractF1ScoreCallback):
    def predict(self):
        validation_data = self.validation_data
        data, true_labels = validation_data[0], validation_data[1]
        predictions = self.model.predict(data, batch_size=self.config.batch_size)

        accuracy, micro_precision, micro_recall, micro_f1 = calculate_metrics(predictions, true_labels)
        print(f'val_f1: {micro_f1}, acc: {accuracy}')

        self.metrics.append(f'{micro_f1}, {accuracy}')

        return micro_f1, accuracy


class AbstractModel:
    def __init__(self, config: Config, f1_score_callback: AbstractF1ScoreCallback, word_index=None):
        self.config = config
        self.f1_score_callback = f1_score_callback
        self.word_index = word_index
        self.unique_id = create_unique_id()
        self.model = self._build_model()

    @abstractmethod
    def _build_model(self):
        raise NotImplementedError

    def _predict(self, test_data):
        return list(self.model.predict(test_data, batch_size=self.config.batch_size, verbose=1))

    def _fit(self, train_data, train_labels, validation_data, validation_labels):
        self._save_configuration()
        self.model.fit(train_data,
                       train_labels,
                       validation_data=(validation_data, validation_labels),
                       callbacks=self._get_callbacks(),
                       epochs=self.config.epochs, batch_size=self.config.batch_size,
                       class_weight=self._get_class_weight(), verbose=self.config.verbose)
        self._save_info_about_model()

    def _compile(self, model):
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.rmsprop(lr=self.config.learning_rate),
                      metrics=['acc'])

        print(model.summary())

        return model

    def _create_embedding_layer(self, embedding_config: EmbeddingConfig):
        if embedding_config.dynamic:
            return create_dynamic_embedding_layer(embedding_config.embedding, self.config.max_sequence_length,
                                                  embedding_config.model_dir)
        else:
            return create_static_embedding_layer(embedding_config.embedding, embedding_config.embedding_file_path,
                                                 self.config.max_sequence_length, self.word_index)

    def load_weights(self, file_path=None):
        if file_path is None:
            self.model.load_weights(self.config.model_file_path)
        else:
            self.model.load_weights(file_path)

    def f1_score(self):
        return self.f1_score_callback.best_f1

    def accuracy(self):
        return self.f1_score_callback.best_accuracy

    def _get_class_weight(self):
        return {0: self.config.others_class_weight,
                1: 1,
                2: 1,
                3: 1}

    def _get_callbacks(self):
        return [self._get_model_checkpoint(), self.f1_score_callback]

    def _get_model_checkpoint(self):
        return ModelCheckpoint(
            self._unique_model_directory() + '/model.loss_{val_loss:.4f}-acc_{val_acc:.4f}-epoch_{epoch:02d}.hdf5',
            monitor='val_acc')

    def _unique_model_directory(self):
        return create_directory(self.config.checkpoint_dir, self.unique_id)

    def _save_configuration(self):
        append_to_file(self._unique_model_directory() + '/model.config', f'{self.config}')

    def _save_info_about_model(self):
        append_to_file(self.config.results_file_path, f'{self.unique_id}, {self.config.name}, {self.f1_score()}')
        metrics = '\n'.join(self.f1_score_callback.metrics)
        append_to_file(self._unique_model_directory() + '/meta.txt', metrics)


class BaselineModel(AbstractModel):
    def __init__(self, config: Config, word_index=None):
        super().__init__(config, BaseF1ScoreCallback(config), word_index)

    def _build_model(self):
        neural_net_config = self.config.baseline_config
        embedding_layer = self._create_embedding_layer(neural_net_config.embedding_config)

        x = embedding_layer.get_input()
        emb = embedding_layer.get_output()(x)
        lstm = LSTM(neural_net_config.lstm_dim, dropout=self.config.dropout,
                    recurrent_dropout=self.config.recurrent_dropout)(emb)
        out = Dense(self.config.num_classes, activation='softmax')(lstm)

        model = Model(x, out)
        model = self._compile(model)

        return model

    def predict(self, test):
        return super()._predict(test)

    def fit(self, train_data, train_labels, validation, validation_labels):
        super()._fit(train_data, train_labels, validation, validation_labels)


class BiLSTMModel(AbstractModel):
    def __init__(self, config: Config, word_index=None):
        super().__init__(config, BaseF1ScoreCallback(config), word_index)

    def _build_model(self):
        neural_net_config = self.config.baseline_config
        embedding_layer = self._create_embedding_layer(neural_net_config.embedding_config)

        x = embedding_layer.get_input()
        emb = embedding_layer.get_output()(x)

        bi_lstm_1 = self._create_bi_lstm_layer(neural_net_config.lstm_dim, True)(emb)
        bi_lstm_2 = self._create_bi_lstm_layer(neural_net_config.lstm_dim)(bi_lstm_1)
        hidden = Dense(self.config.dense_dim, activation='sigmoid')(bi_lstm_2)
        out = Dense(self.config.num_classes, activation='softmax')(hidden)

        model = Model(x, out)
        model = self._compile(model)

        return model

    def _create_bi_lstm_layer(self, lstm_dim, return_sequences=False):
        return Bidirectional(LSTM(lstm_dim, return_sequences=return_sequences, dropout=self.config.dropout,
                                  recurrent_dropout=self.config.recurrent_dropout))

    def predict(self, test):
        return super()._predict(test)

    def fit(self, train_data, train_labels, validation, validation_labels):
        super()._fit(train_data, train_labels, validation, validation_labels)


class ConSSEDModel(AbstractModel):
    def __init__(self, config: Config, word_index=None):
        super().__init__(config, ConSSEDF1ScoreCallback(config), word_index)

    def _build_model(self):
        conssed_config = self.config.conssed_config
        x_1, SemRN = self._create_layers(conssed_config.semantic_part)
        x_2, SenRN = self._create_layers(conssed_config.sentiment_part)

        merged = Concatenate(axis=-1)([SemRN, SenRN])
        hidden = Dense(self.config.dense_dim, activation='sigmoid')(merged)
        out = Dense(self.config.num_classes, activation='softmax')(hidden)

        model = Model([x_1, x_2], out)
        model = self._compile(model)

        return model

    def _create_layers(self, neural_net_part: RecurrentNetworkConfig):
        embedding_layer = self._create_embedding_layer(neural_net_part.embedding_config)

        input = embedding_layer.get_input()
        emb = embedding_layer.get_output()(input)
        lstm_1 = self._create_first_lstm_layer(neural_net_part)(emb)
        lstm_2 = self._create_second_lstm_layer(neural_net_part)(lstm_1)

        return input, lstm_2

    def _create_first_lstm_layer(self, neural_net_part: RecurrentNetworkConfig):
        lstm = LSTM(neural_net_part.lstm_dim, return_sequences=True, dropout=self.config.dropout,
                    recurrent_dropout=self.config.recurrent_dropout)
        return Bidirectional(lstm) if neural_net_part.first_bidirectional else lstm

    def _create_second_lstm_layer(self, neural_net_part: RecurrentNetworkConfig):
        lstm = LSTM(neural_net_part.lstm_dim, dropout=self.config.dropout,
                    recurrent_dropout=self.config.recurrent_dropout)

        return Bidirectional(lstm) if neural_net_part.second_bidirectional else lstm

    def predict(self, test_sem, test_sen):
        predictions_without_others_class_regularizer = super()._predict([test_sem, test_sen])
        predictions = others_class_regularizer(predictions_without_others_class_regularizer,
                                               self.config.others_class_regularizer_param)
        return predictions, predictions_without_others_class_regularizer

    def fit(self, train_sem, train_sen, train_labels, validation_sem, validation_sen, validation_labels):
        super()._fit([train_sem, train_sen], train_labels, [validation_sem, validation_sen], validation_labels)
