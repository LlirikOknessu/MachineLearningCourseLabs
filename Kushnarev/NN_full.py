import argparse
import datetime
import random
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorboard.plugins import projector
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from NN import SomeModel


BEST_PARAMETERS = {'neurons_cnt': 50, 'batch_size': 32, 'learning_rate': 0.09}
random.seed(5)


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/NN_prod',
                        required=False, help='path to save prepared data')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/NN',
                        required=False, help='path to save prepared data')
    parser.add_argument('--logs_dir', '-logd', type=str, default='data/logs/testLogs',
                        required=False, help='path to save prepared data')
    return parser.parse_args()


class SomeModel(Model):
  def __init__(self, neurons_cnt=128):
    super(SomeModel, self).__init__()
    self.d_in = Dense(8, activation='relu')
    self.d_1 = Dense(neurons_cnt, activation='relu')
    self.d_2 = Dense(neurons_cnt, activation='relu')
    self.d_out = Dense(1)

  def call(self, x):
      x = self.d_in(x)
      x = self.d_1(x)
      x = self.d_2(x)
      return self.d_out(x)


@tf.function
def train_step(model, input_vector, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(input_vector, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_model = Path(args.input_model)
    logs_path = Path(args.logs_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'

    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    BUFFER_SIZE = 128
    EPOCHS = 10

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_full, y_full)).shuffle(BUFFER_SIZE).batch(BEST_PARAMETERS['batch_size'])

    model = SomeModel(neurons_cnt=BEST_PARAMETERS["neurons_cnt"])

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=BEST_PARAMETERS['learning_rate'])

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

    paramName = "_".join(str(x) for x in BEST_PARAMETERS.values())

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = logs_path / paramName / 'gradient_tape' / current_time / 'train'
    train_log_dir.mkdir(exist_ok=True, parents=True)

    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))

    logdir = logs_path / paramName / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(exist_ok=True, parents=True)

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        for (x_train, y_train) in train_ds:
            train_step(model, x_train, y_train)

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            # if epoch == 0 or epoch == EPOCHS-1:
            tf.summary.histogram('Weights_in_r', model.d_in.weights[0], step=epoch)
            # print(model.d_in.weights[0])
            tf.summary.histogram('Weights1_r', model.d_1.weights[0], step=epoch)
            tf.summary.histogram('Weights2_r', model.d_2.weights[0], step=epoch)
            tf.summary.histogram('Weights_out_r', model.d_out.weights[0], step=epoch)

        # if epoch == 0:
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='2d')
            # ax.scatter(model.d_1.weights[0][0], model.d_1.weights[0][1], marker="o")
            # plotBuff = []
            # for i in range(2):
            #     plotBuff.extend(model.d_1.weights[0][i])
            # plt.figure(figsize=(10, 7), dpi=80)
            # sns.histplot(plotBuff)

        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result()))

        train_loss.reset_states()
        train_accuracy.reset_states()

    # with fit_summary_writer.as_default():
    #     tf.summary.trace_export(
    #         name="my_func_trace",
    #         step=0,
    #         profiler_outdir=logdir)


    model.save(output_dir, overwrite=True)