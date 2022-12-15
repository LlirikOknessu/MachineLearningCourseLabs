import argparse
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from joblib import load
from sklearn.metrics import mean_absolute_error
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

tf.config.run_functions_eagerly(True)



def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--logs_dir', '-logd', type=str, default='data/logs/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/DecisionTree_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
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


@tf.function
def test_step(model, input_vector, labels):
  predictions = model(input_vector, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)


if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['NN']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    logs_path = Path(args.logs_dir)
    baseline_model_path = Path(args.baseline_model)

    output_dir.mkdir(exist_ok=True, parents=True)

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    BUFFER_SIZE = 128
    EPOCHS = 40

    HP_NEURONS_CNT = params['neurons_cnt']
    HP_BATCH_SIZE = params['batch_size']
    HP_LEARNING_RATE = params['learning_rate']

    # HP_NEURONS_CNT = [100]
    # HP_HIDDEN_LAYERS = [3]
    # HP_BATCH_SIZE = [64]
    # HP_LEARNING_RATE = [0.09]

    bestMAE = 100
    bestHP = []

    for neurons_cnt in HP_NEURONS_CNT:
        for batch_size in HP_BATCH_SIZE:
            for learning_rate in HP_LEARNING_RATE:
                print("\nPARAMETERS ARE: neurons_cnt: {}, batch_size: {}, learning_rate: {}\n".format(
                    *[str(x) for x in [neurons_cnt, batch_size, learning_rate]]
                ))
                X_train_copy = pd.read_csv(X_train_name)
                y_train_copy = pd.read_csv(y_train_name)
                X_test_copy = pd.read_csv(X_test_name)
                y_test_copy = pd.read_csv(y_test_name)

                train_ds = tf.data.Dataset.from_tensor_slices(
                    (X_train_copy, y_train_copy)).shuffle(BUFFER_SIZE).batch(batch_size)
                test_ds = tf.data.Dataset.from_tensor_slices((X_test_copy, y_test_copy)).batch(batch_size)

                model = SomeModel(neurons_cnt=neurons_cnt)

                loss_object = tf.keras.losses.MeanSquaredError()
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

                train_loss = tf.keras.metrics.Mean(name='train_loss')
                train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

                test_loss = tf.keras.metrics.Mean(name='test_loss')
                test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_mae')

                paramName = "_".join(str(x) for x in [neurons_cnt, batch_size, learning_rate])

                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                train_log_dir = logs_path / paramName / 'gradient_tape' / current_time / 'train'
                train_log_dir.mkdir(exist_ok=True, parents=True)
                test_log_dir = logs_path / paramName / 'gradient_tape' / current_time / 'test'
                test_log_dir.mkdir(exist_ok=True, parents=True)
                train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
                test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))

                logdir = logs_path / paramName / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                logdir.mkdir(exist_ok=True, parents=True)
                fit_summary_writer = tf.summary.create_file_writer(str(logdir))

                tf.summary.trace_on(graph=True, profiler=True)
                for epoch in range(EPOCHS):
                    # Reset the metrics at the start of the next epoch
                    for (x_train, y_train) in train_ds:
                        with fit_summary_writer.as_default():
                            train_step(model, x_train, y_train)

                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss.result(), step=epoch)
                        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

                    for (x_test, y_test) in test_ds:
                        test_step(model, x_test, y_test)

                    with test_summary_writer.as_default():
                        tf.summary.scalar('loss', test_loss.result(), step=epoch)
                        tf.summary.scalar('mae', test_accuracy.result(), step=epoch)

                    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test MAE: {}'
                    print(template.format(epoch + 1,
                                          train_loss.result(),
                                          train_accuracy.result(),
                                          test_loss.result(),
                                          test_accuracy.result()))

                    if epoch == EPOCHS-1:
                        buff = test_accuracy.result().numpy()
                        if buff < bestMAE:

                            bestMAE = buff
                            bestHP = [neurons_cnt, batch_size, learning_rate]
                            model.save('data/models/NN', overwrite=True)


                    # Reset metrics every epoch
                    train_loss.reset_states()
                    test_loss.reset_states()
                    train_accuracy.reset_states()
                    test_accuracy.reset_states()

                # tf.keras.models.save_model(model, 'data/models/NN', overwrite=True)
                with fit_summary_writer.as_default():
                    tf.summary.trace_export(
                        name="my_func_trace",
                        step=0,
                        profiler_outdir=logdir)

    print("Best params are:\nneurons_cnt: {}, batch_size: {}, learning_rate: {}".format(
        *[str(x) for x in bestHP]
    ))

    baseline_model = load(baseline_model_path)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_test))

    print("Baseline MAE: ", mean_absolute_error(y_test, y_pred_baseline))
    print("Model MAE: ", bestMAE)