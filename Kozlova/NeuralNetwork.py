import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import tensorboard
import datetime
import shutil
import yaml
import random
from tensorflow.keras import Model
from pathlib import Path
from sklearn.model_selection import ParameterGrid
import pandas as pd


class SomeModel(Model):
  def __init__(self, neurons_cnt):
    super(SomeModel, self).__init__()
    self.d_in = Dense(8, activation='sigmoid')
    self.d_1 = Dense(neurons_cnt, activation='sigmoid')
    self.d_2 = Dense(neurons_cnt, activation='sigmoid')
    self.d_3 = Dense(neurons_cnt, activation='sigmoid')
    self.d_4 = Dense(neurons_cnt, activation='sigmoid')
    self.d_out = Dense(1)

  def call(self, x):
    x = self.d_in(x)
    x = self.d_1(x)
    x = self.d_2(x)
    x = self.d_3(x)
    x = self.d_4(x)

    return self.d_out(x)

def train_step(input_vector, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(input_vector, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

def test_step(input_vector, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(input_vector, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


if __name__ == '__main__':
    random.seed(35)
    BUFFER_SIZE = 128
    EPOCHS = 50
    defaultMAE = 100
    bestParam = []

    input_dir = Path('./data/prepared')
    logs_path = Path('./data/logs')
    if logs_path.exists():
        shutil.rmtree(logs_path)
    logs_path.mkdir(parents=True)

    with open('params.yaml', 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    grid = ParameterGrid(params['NeuralNetwork'])

    for paramset in grid:
        X_train = pd.read_csv(X_train_name)
        y_train = pd.read_csv(y_train_name)
        X_test = pd.read_csv(X_test_name)
        y_test = pd.read_csv(y_test_name)

        train_ds = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).shuffle(BUFFER_SIZE).batch(paramset['BATCH_SIZE'])

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(paramset['BATCH_SIZE'])

        loss_object = tf.keras.losses.MeanAbsoluteError()
        optimizer = tf.keras.optimizers.SGD(learning_rate=paramset['LEARNING_RATE'])

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_mae')

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = logs_path / 'gradient_tape' / current_time / 'train'
        train_log_dir.mkdir(exist_ok=True, parents=True)
        test_log_dir = logs_path / 'gradient_tape' / current_time / 'test'
        test_log_dir.mkdir(exist_ok=True, parents=True)
        train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
        test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))

        logdir = logs_path / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir.mkdir(exist_ok=True, parents=True)
        fit_summary_writer = tf.summary.create_file_writer(str(logdir))

        model = SomeModel(neurons_cnt=paramset['neurons_cnt'])

        tf.summary.trace_on(graph=True, profiler=True)

        for epoch in range(EPOCHS):
          # Reset the metrics at the start of the next epoch
          for (x_train, y_train) in train_ds:
            with fit_summary_writer.as_default():
              train_step(x_train, y_train)

          with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

          for (x_test, y_test) in test_ds:
            test_step(x_test, y_test)

          with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('mae', test_accuracy.result(), step=epoch)

          template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, ' \
                     'Test MAE: {}, neurons_cnt: {}, BATCH_SIZE: {}, LEARNING_RATE: {}'
          print(template.format(epoch + 1,
                                train_loss.result(),
                                train_accuracy.result(),
                                test_loss.result(),
                                test_accuracy.result(),
                                paramset['neurons_cnt'],
                                paramset['BATCH_SIZE'],
                                paramset['LEARNING_RATE']))

          if epoch == EPOCHS - 1 and test_accuracy.result() < defaultMAE and test_accuracy.result() != 0:
              defaultMAE = test_accuracy.result()
              bestParam = [paramset['neurons_cnt'], paramset['BATCH_SIZE'], paramset['LEARNING_RATE']]
              model.save('data/models/NeuralNetwork', overwrite=True)

          # Reset metrics every epoch
          train_loss.reset_states()
          test_loss.reset_states()
          train_accuracy.reset_states()
          test_accuracy.reset_states()

    with fit_summary_writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir)

print(bestParam)


