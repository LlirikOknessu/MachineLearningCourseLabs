import datetime

import tensorflow as tf
import argparse
import yaml
import pandas as pd
from pathlib import Path
from NeuralNetwork import NeuralNetwork_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Model
from joblib import load
from sklearn.metrics import mean_absolute_error as mae

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Path parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--logs_dir', '-ld', type=str, default='data/logs/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression.joblib',
                        required=False, help='path to linear regression prod version')
    return parser.parse_args()

class NeuralNetwork_model(Model):
    def __init__(self, number_of_neurons):
        super(NeuralNetwork_model, self).__init__()
        self.d_in = Dense(25, activation='relu')
        self.d_1 = Dense(number_of_neurons, activation='sigmoid')
        self.d_2 = Dense(number_of_neurons, activation='relu')
        # self.d_3 = Dense(number_of_neurons, activation='relu')
        # self.d_4 = Dense(number_of_neurons, activation='relu')
        # self.d_5 = Dense(number_of_neurons, activation='relu')
        self.d_out = Dense(1)

    def call(self, x):
        x = self.d_in(x)
        x = self.d_1(x)
        x = self.d_2(x)
        # x = self.d_3(x)
        # x = self.d_4(x)
        # x = self.d_5(x)
        x = self.d_out(x)
        return x

@tf.function
def train_step(input_vector, labels, model, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(input_vector, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

if __name__ == '__main__':
    args = parser_args_for_sac()

    best_number_of_neurons = 512
    best_batch_size = 64
    best_buffer_size = 128
    best_learning_rate = 0.001
    best_epochs = 400

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    logs_path = Path(args.logs_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    output_model_path = output_dir / ('NeuralNetwork_Prod')

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'
    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_full, y_full)).shuffle(best_buffer_size).batch(best_batch_size)

    model = NeuralNetwork_model(number_of_neurons=best_number_of_neurons)

    loss_object = tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=best_learning_rate)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = logs_path / 'best' / 'gradient_tape' / current_time / 'train'
    train_log_dir.mkdir(exist_ok=True, parents=True)
    logdir = logs_path / 'best' / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(exist_ok=True, parents=True)

    fit_summary_writer = tf.summary.create_file_writer(str(logdir))
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
    tf.summary.trace_on(graph=True, profiler=True)

    for current_epoch in range(best_epochs):
        for (x_train_current, y_train_current) in train_ds:
            with fit_summary_writer.as_default():
                train_step(x_train_current, y_train_current, model, optimizer)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=current_epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=current_epoch)
                tf.summary.histogram('Weight_in_r', model.d_in.weights[0], step=current_epoch)
                tf.summary.histogram('Weight_1_r', model.d_1.weights[0], step=current_epoch)
                tf.summary.histogram('Weight_2_r', model.d_2.weights[0], step=current_epoch)
                tf.summary.histogram('Wright_out_r', model.d_out.weights[0], step=current_epoch)

        template = 'Epoch: {}, Train_Loss: {}, Train_MAE: {}'
        print(template.format(current_epoch + 1,
                              train_loss.result(),
                              train_accuracy.result()))
        train_loss.reset_state()
        train_accuracy.reset_state()

    with fit_summary_writer.as_default():
        tf.summary.trace_export(name="Func_Trace",
                                step=0,
                                profiler_outdir=logdir)

    model.summary()
    model.save(output_model_path, overwrite=True)











