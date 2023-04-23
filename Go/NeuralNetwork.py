import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Model
from pathlib import Path
import pandas as pd
import datetime
import shutil
import argparse
import yaml
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error
import random
import math

# tf.config.run_functions_eagerly(True)

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared', help="path to input directory")
    parser.add_argument('--output_dir', '-od', type=str, default='data/models', help="path to output directory")
    parser.add_argument('--logs_dir', '-ld', type=str, default='data/logs', help="path to logs directory")
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib', help='path to linear regression prod version')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', help='file with dvc stage params')
    return parser.parse_args()

class NeuNet(Model):
    def __init__(self, neurons=128):
        super(NeuNet, self).__init__()
        self.in_layer = Dense(10, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')
        self.hidden_1 = Dense(neurons, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')
        self.hidden_2 = Dense(neurons, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')
        self.out_layer = Dense(1, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros')

    def call(self, inputs):
        x = self.in_layer(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        outputs = self.out_layer(x)
        return outputs


# @tf.function
def train_net(data: pd.DataFrame, labels: pd.DataFrame, net: NeuNet, optimizer: tf.keras.optimizers, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = net(data)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

# def apply_train():
#     @tf.function
#     def train_net(data: pd.DataFrame, labels: pd.DataFrame, net: NeuNet, optimizer: tf.keras.optimizers, train_loss, train_accuracy):
#         with tf.GradientTape() as tape:
#             predictions = net(data)
#             loss = loss_object(labels, predictions)
#         gradients = tape.gradient(loss, net.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, net.trainable_variables))
#         # train_loss(loss)
#         # train_accuracy(labels, predictions)
#         return loss, predictions
#     return train_net

@tf.function
def val_net(data: pd.DataFrame, labels: pd.DataFrame, net: NeuNet, val_loss, val_accuracy):
    predictions = net(data)
    loss = loss_object(labels, predictions)
    val_loss(loss)
    val_accuracy(labels, predictions)


if __name__ == '__main__':
    random.seed(42)
    args = parser_args()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['neural_network']

    BATCH_SIZE = params.get('batch_size')
    BUFFER_SIZE = params.get('buffer_size')
    NEURONS = params.get('neurons')
    LEARNING_RATE = params.get('learning_rate')
    EPOCHS = 100

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    logs_path = Path(args.logs_dir)
    baseline_model_path = Path(args.baseline_model)

    if logs_path.exists():
        shutil.rmtree(logs_path)
    out_dir.mkdir(exist_ok=True, parents=True)

    X_train_name = in_dir / 'X_train.csv'
    y_train_name = in_dir / 'y_train.csv'
    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_val_name = in_dir / 'X_val.csv'
    y_val_name = in_dir / 'y_val.csv'
    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    bestAcc = 1000
    bestParameters = []

    loss_object = tf.keras.losses.MeanSquaredError()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_mse')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.MeanSquaredError(name='val_mse')

    for _batch_size in BATCH_SIZE:
        for _buffer_size in BUFFER_SIZE:
            for _neurons in NEURONS:
                for _learning_rate in LEARNING_RATE:

                    template1 = 'Batch Size {}, Buffer Size: {}, Neurons: {}, Learning Rate: {}'
                    print(template1.format(_batch_size,
                                            _buffer_size,
                                            _neurons,
                                            _learning_rate))

                    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(_buffer_size).batch(_batch_size)
                    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(_batch_size)

                    net = NeuNet(neurons=_neurons)
                    optimizer = tf.keras.optimizers.SGD(learning_rate=_learning_rate)

                    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                    train_log_dir = logs_path / 'gradient_tape' / current_time / 'train'
                    val_log_dir = logs_path / 'gradient_tape' / current_time / 'val'

                    train_log_dir.mkdir(exist_ok=True, parents=True)
                    val_log_dir.mkdir(exist_ok=True, parents=True)

                    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
                    val_summary_writer = tf.summary.create_file_writer(str(val_log_dir))

                    logdir = logs_path / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    logdir.mkdir(exist_ok=True, parents=True)
                    fit_summary_writer = tf.summary.create_file_writer(str(logdir))

                    tf.summary.trace_on(graph=True, profiler=True)

                    for epoch in range(EPOCHS):
                        for (X_train, y_train) in train_ds:
                            with fit_summary_writer.as_default():
                                train_net(X_train, y_train, net, optimizer, train_loss, train_accuracy)

                                # train = apply_train()
                                # loss, pred = train(X_train, y_train, net, optimizer, train_loss, train_accuracy)
                                # train_loss(loss)
                                # train_accuracy(y_train, pred)

                        with train_summary_writer.as_default():
                            tf.summary.scalar('loss', train_loss.result(), step=epoch)
                            tf.summary.scalar('mse', train_accuracy.result(), step=epoch)

                        for (X_val, y_val) in val_ds:
                            val_net(X_val, y_val, net, val_loss, val_accuracy)

                        with val_summary_writer.as_default():
                            tf.summary.scalar('loss', val_loss.result(), step=epoch)
                            tf.summary.scalar('mse', val_accuracy.result(), step=epoch)

                        template = 'Epoch {}, Loss: {}, Accuracy: {}, val Loss: {}, val MSE: {}'
                        print (template.format(epoch+1,
                                                train_loss.result(),
                                                train_accuracy.result(),
                                                val_loss.result(),
                                                val_accuracy.result()))

                        if epoch == EPOCHS - 1:
                            if val_accuracy.result().numpy() < bestAcc:
                                bestAcc = val_accuracy.result().numpy()
                                bestParameters = [_batch_size, _buffer_size, _neurons, _learning_rate]
                                net.save('./data/models/NeuNet', overwrite=True)

                        # Reset metrics every epoch
                        train_loss.reset_states()
                        val_loss.reset_states()
                        train_accuracy.reset_states()
                        val_accuracy.reset_states()

                    with fit_summary_writer.as_default():
                        tf.summary.trace_export(
                            name="my_func_trace",
                            step=0,
                            profiler_outdir=logdir
                        )
    template1 = 'Best parameters: Batch Size {}, Buffer Size: {}, Neurons: {}, Learning Rate: {}'
    print(template1.format(*[str(x) for x in bestParameters]))
    print("bessAccuracy (RMSE): ", math.sqrt(bestAcc))

    baseline_model = load(baseline_model_path)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_val))
    print("Baseline RMSE: ", math.sqrt(mean_squared_error(y_val, y_pred_baseline)))


# import tensorflow as tf
# from tensorflow.python.keras.layers import Dense
# from tensorflow.python.keras import Model
# import pandas as pd
# from pathlib import Path
# import argparse
# import yaml
# import datetime
# from joblib import load
# from sklearn.metrics import mean_absolute_error as mae
#
#
# def parser_args_for_sac():
#     parser = argparse.ArgumentParser(description='Paths parser')
#     parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
#                         required=False, help='path to input data directory')
#     parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
#                         required=False, help='path to save prepared data')
#     parser.add_argument('--logs_dir', '-logd', type=str, default='data/logs/',
#                         required=False, help='path to save prepared data')
#     parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/ExtraTree_prod.joblib',
#                         required=False, help='path to ExtraTree prod version')
#     parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
#                         help='file with dvc stage params')
#     return parser.parse_args()
#
#
# def get_log_dirs(params):
#     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     train_log_dir = logs_path / params / 'gradient_tape' / current_time / 'train'
#     train_log_dir.mkdir(exist_ok=True, parents=True)
#     test_log_dir = logs_path / params / 'gradient_tape' / current_time / 'test'
#     test_log_dir.mkdir(exist_ok=True, parents=True)
#     fit_log_dir = logs_path / params / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#     fit_log_dir.mkdir(exist_ok=True, parents=True)
#
#     return train_log_dir, test_log_dir, fit_log_dir
#
#
# class MyModel(Model):
#     def __init__(self, n_of_neurons=10):
#         super(MyModel, self).__init__()
#
#         self.d_in = Dense(25, activation='relu')
#         self.d_1 = Dense(n_of_neurons, activation='sigmoid')
#         self.d_2 = Dense(n_of_neurons, activation='relu')
#         #self.d_3 = Dense(n_of_neurons, activation='sigmoid')
#         self.d_out = Dense(1)
#
#     def call(self, x):
#         x = self.d_in(x)
#         x = self.d_1(x)
#         x = self.d_2(x)
#         #x = self.d_3(x)
#         x = self.d_out(x)
#         return x
#
#
# @tf.function
# def train_step(input_vector, labels, model, optimizer):
#     with tf.GradientTape() as tape:
#         predictions = model(input_vector, training=True)
#         loss = loss_object(labels, predictions)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#
#     train_loss(loss)
#     train_accuracy(labels, predictions)
#
#
# @tf.function
# def test_step(input_vector, labels, model):
#     predictions = model(input_vector, training=False)
#     t_loss = loss_object(labels, predictions)
#
#     test_loss(t_loss)
#     test_accuracy(labels, predictions)
#
#
# tf.config.run_functions_eagerly(True)
#
# if __name__ == '__main__':
#     args = parser_args_for_sac()
#
#     with open(args.params, 'r') as f:
#         params_a = yaml.safe_load(f)
#     params = params_a['NeuralNetwork']
#
#     input_dir = Path(args.input_dir)
#     output_dir = Path(args.output_dir)
#     logs_path = Path(args.logs_dir)
#     baseline_model_path = Path(args.baseline_model)
#     output_model_path = output_dir / ('NeuralNetwork')
#
#     output_dir.mkdir(exist_ok=True, parents=True)
#
#     X_train_name = input_dir / 'X_train.csv'
#     y_train_name = input_dir / 'y_train.csv'
#     X_test_name = input_dir / 'X_test.csv'
#     y_test_name = input_dir / 'y_test.csv'
#
#     BATCH_SIZE = params['batch_size']
#     BUFFER_SIZE = params['buffer_size']
#     LEARNING_RATE = params['learning_rate']
#     EPOCHS = params['epochs']
#     NUMBER_OF_NEURONS = params['n_of_neurons']
#
#     top_mae = 100
#     top_params = []
#     best_model_mae = None
#
#     loss_object = tf.keras.losses.MeanSquaredError()
#
#     train_loss = tf.keras.metrics.Mean(name='train_loss')
#     train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
#
#     test_loss = tf.keras.metrics.Mean(name='test_loss')
#     test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_mae')
#
#     for batch in BATCH_SIZE:
#         # print(batch)
#         train_ds = tf.data.Dataset.from_tensor_slices((pd.read_csv(X_train_name), pd.read_csv(y_train_name))).shuffle(
#             BUFFER_SIZE).batch(batch)
#         test_ds = tf.data.Dataset.from_tensor_slices((pd.read_csv(X_test_name), pd.read_csv(y_test_name))).batch(batch)
#         for lr in LEARNING_RATE:
#             for n_neurons in NUMBER_OF_NEURONS:
#                 params = 'ba{}_lr{}_nn{}'.format(batch, lr, n_neurons)
#                 train_log_dir, test_log_dir, logdir = get_log_dirs(params)
#                 train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
#                 test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))
#                 fit_summary_writer = tf.summary.create_file_writer(str(logdir))
#                 print("Params are: BATCH_SIZE={}, LEARNING_RATE={}, NUMBER_OF_NEURONS={}".format(batch, lr, n_neurons))
#                 tf.summary.trace_on(graph=True, profiler=True)
#                 model = MyModel(n_of_neurons=n_neurons)
#                 optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#
#                 for epoch in range(EPOCHS):
#                     for (x_train, y_train) in train_ds:
#                         with fit_summary_writer.as_default():
#                             train_step(x_train, y_train, model, optimizer)
#
#                     with train_summary_writer.as_default():
#                         tf.summary.scalar('loss', train_loss.result(), step=epoch)
#                         tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
#
#                     for (x_test, y_test) in test_ds:
#                         test_step(x_test, y_test, model)
#
#                     with test_summary_writer.as_default():
#                         tf.summary.scalar('loss', test_loss.result(), step=epoch)
#                         tf.summary.scalar('mae', test_accuracy.result(), step=epoch)
#
#                     template = 'Epoch: {}, Train Loss: {}, Train MAE: {}, Test Loss: {}, Test MAE: {}'
#                     print(template.format(epoch + 1,
#                                           train_loss.result(),
#                                           train_accuracy.result(),
#                                           test_loss.result(),
#                                           test_accuracy.result()))
#                     if epoch == EPOCHS - 1:
#                         if top_mae > test_accuracy.result():
#                             best_model_mae = test_accuracy.result()
#                             top_params = [batch, lr, n_neurons]
#                             top_mae = test_accuracy.result()
#                             model.save(output_model_path, overwrite=True)
#
#                     train_loss.reset_states()
#                     train_accuracy.reset_states()
#                     test_loss.reset_states()
#                     test_accuracy.reset_states()
#
#                 with fit_summary_writer.as_default():
#                     tf.summary.trace_export(
#                         name="my_func_trace",
#                         step=0,
#                         profiler_outdir=logdir
#                     )
#
#     print(
#         'Best params: BATCH_SIZE={}, LEARNING_RATE={}, NUMBER_OF_NEURONS={}'.format(
#             top_params[0],
#             top_params[1],
#             top_params[2])
#     )
#
#     baseline_model = load(baseline_model_path)
#     baseline_preds = baseline_model.predict(pd.read_csv(X_test_name))
#     print('Baseline MAE: ', mae(pd.read_csv(y_test_name), baseline_preds))
#     print('NN Model MAE: ', best_model_mae)