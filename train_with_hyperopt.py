import sys
from structures import load_configuration
from hyperopt import hp, tpe, fmin, space_eval
from train import train


def train_with_hyperopt(args):
    sem_lstm_dim, sem_first_bidirectional, sem_second_bidirectional, sen_lstm_dim, sen_first_bidirectional,  \
    sen_second_bidirectional, hidden_dim, lstm_dim, batch_size, dropout, recurrent_dropout, learning_rate, others_class_weight\
        = args

    config = load_configuration(str(sys.argv[1]))

    config.dense_dim = hidden_dim
    config.dropout = dropout
    config.recurrent_dropout = recurrent_dropout
    config.learning_rate = learning_rate
    config.others_class_weight = others_class_weight
    config.batch_size = batch_size

    if config.conssed_config is not None:
        sem_rn = config.conssed_config.semantic_part
        sem_rn.lstm_dim = sem_lstm_dim
        sem_rn.first_bidirectional = sem_first_bidirectional
        sem_rn.second_bidirectional = sem_second_bidirectional

        sen_rn = config.conssed_config.sentiment_part
        sen_rn.lstm_dim = sen_lstm_dim
        sen_rn.first_bidirectional = sen_first_bidirectional
        sen_rn.second_bidirectional = sen_second_bidirectional

    if config.baseline_config is not None:
        config.baseline_config.lstm_dim = lstm_dim

    return train(config)


if __name__ == '__main__':
    space = [
        hp.choice('SEM_LSTM_DIM', [200, 230, 256, 280, 300, 320]),
        hp.choice('SEM_FIRST_BIDIRECTIONAL', [False, True]),
        hp.choice('SEM_SECOND_BIDIRECTIONAL', [False, True]),
        hp.choice('SEN_LSTM_DIM', [200, 230, 256, 280, 300, 320]),
        hp.choice('SEN_FIRST_BIDIRECTIONAL', [False, True]),
        hp.choice('SEN_SECOND_BIDIRECTIONAL', [False, True]),
        hp.choice('HIDDEN_DIM', [100, 128, 150]),
        hp.choice('LSTM_DIM', [200, 230, 256, 280, 300, 320]),
        hp.choice('BATCH_SIZE', [32, 64, 80, 100, 128]),
        hp.uniform('DROPOUT', 0.1, 0.5),
        hp.uniform('RECURRENT_DROPOUT', 0.1, 0.5),
        hp.uniform('LEARNING_RATE', 0.001, 0.004),
        hp.uniform('OTHERS_CLASS_WEIGHT', 1.0, 3.0)
    ]

    best = fmin(train_with_hyperopt, space, algo=tpe.suggest, max_evals=10)

    print(space_eval(space, best))
