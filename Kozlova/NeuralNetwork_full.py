import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import tensorboard
import datetime
import shutil
import argparse
import yaml
import random
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from tensorflow.keras import Model
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from joblib import dump, load
from NeuralNetwork import SomeModel
import pandas as pd

tf.config.run_functions_eagerly(False)
def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                        required=False, help='path to linear regression prod version')
    return parser.parse_args()


@tf.function
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

if __name__ == '__main__':
    random.seed(35)
    BUFFER_SIZE = 128
    EPOCHS = 50
    defaultMAE = 100
    bestParam = []

    input_dir = Path('./data/prepared')
    logs_path = Path('./data/logs')

    X_train_name = input_dir / 'X_full.csv'
    y_train_name = input_dir / 'y_full.csv'

    grid = ParameterGrid({"neurons_cnt": [64],
                            "BATCH_SIZE": [32],
                            "LEARNING_RATE": [0.005]})

    for paramset in grid:

        X_train = pd.read_csv(X_train_name)
        y_train = pd.read_csv(y_train_name)

        train_ds = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).shuffle(BUFFER_SIZE).batch(paramset['BATCH_SIZE'])

        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=paramset['LEARNING_RATE'])

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = logs_path / 'gradient_tape' / current_time / 'train'
        train_log_dir.mkdir(exist_ok=True, parents=True)
        train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))

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
            tf.summary.histogram('Weights_in', model.d_in.get_weights()[0], step=epoch)
            tf.summary.histogram('Weights_1', model.d_1.get_weights()[0], step=epoch)
            tf.summary.histogram('Weights_2', model.d_2.get_weights()[0], step=epoch)
            tf.summary.histogram('Weights_out', model.d_out.get_weights()[0], step=epoch)

          template = 'Epoch {}, Loss: {}, MAE: {}'
          print(template.format(epoch + 1,
                                train_loss.result(),
                                train_accuracy.result()))

          # Reset metrics every epoch
          train_loss.reset_states()
          train_accuracy.reset_states()

        with fit_summary_writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir=logdir)

    model.save('data/models/NeuralNetwork_prob', overwrite=True)

    args = parser_args_for_sac()
    X_full = pd.read_csv(X_train_name)
    y_full = pd.read_csv(y_train_name)

    predicted_values = np.squeeze(model.predict(X_full))

    baseline_model = load(Path(args.baseline_model))
    y_pred_baseline = np.squeeze(baseline_model.predict(X_full))

    print("Accuracy: ", r2_score(y_full, predicted_values))
    print("Baseline MAE: ", mean_absolute_error(y_full, y_pred_baseline))
    print("Model MAE: ", mean_absolute_error(y_full, predicted_values))



