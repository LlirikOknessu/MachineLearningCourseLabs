import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorboard
import datetime
import shutil
import argparse
import yaml
from NN_without_pretest import bestParam, SomeModel
from tensorflow.keras import Model
from sklearn.model_selection import ParameterGrid
from pathlib import Path
import pandas as pd

tf.config.run_functions_eagerly(True)

def parser_args_for_sac():
  parser = argparse.ArgumentParser(description='Paths parser')
  parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                      required=False, help='path to input data directory')
  parser.add_argument('--logs_path', '-od', type=str, default='data/logs/',
                      required=False, help='path to save prepared data')
  parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                      help='file with dvc stage params')
  parser.add_argument('--input_model', '-im', type=str, default='data/models/NN_without_pretest',
                      required=False, help='path with saved model')
  return parser.parse_args()

@tf.function
def train_step(input_vector, labels):
            with tf.GradientTape() as tape:
                predictions = model(input_vector, training=True)
                loss = loss_object(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss(loss)
            train_accuracy(labels, predictions)


if __name__ == '__main__':
    buffer_size = 64
    epochs = 300
    defaultMAE = 100

    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    logs_path = Path(args.logs_path)
    if logs_path.exists():
        shutil.rmtree(logs_path)
    logs_path.mkdir(parents=True)

    X_full_name = input_dir / 'X_full_2.csv'
    y_full_name = input_dir / 'y_full.csv'

    for param in grid:
        X_train = pd.read_csv(X_full_name)
        y_train = pd.read_csv(y_full_name)

        train_ds = tf.data.Dataset.from_tensor_slices(
            (X_train, y_train)).shuffle(buffer_size).batch(param['batch_size'])

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(param['batch_size'])

        model = SomeModel(neurons_cnt=param['neurons_cnt'])

        loss_object = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.SGD(learning_rate=param['learning_rate'])

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

        tf.summary.trace_on(graph=True, profiler=True)
        for epoch in range(epochs):
            for (X_train, y_train) in train_ds:
                with fit_summary_writer.as_default():
                    train_step(X_train, y_train)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            for (X_test, y_test) in test_ds:
                test_step(X_test, y_test)

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('mae', test_accuracy.result(), step=epoch)

            template = 'Epoch {}, Train Loss: {}, Train MAE: {}, Test Loss: {}, Test MAE: {}, ' \
                       'Neurons: {}, Batch Size: {}, Learning Rate: {}'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result(),
                                  test_loss.result(),
                                  test_accuracy.result(),
                                  param['neurons_cnt'],
                                  param['batch_size'],
                                  param['learning_rate']))

            if epoch == epochs - 1:
                buff = test_accuracy.result().numpy()
                if buff < defaultMAE:
                    defaultMAE = buff
                    bestParam = [param['neurons_cnt'], param['batch_size'], param['learning_rate']]
                    model.save('data/models/NN_without_pretest', overwrite=True)

            # Reset metrics every epoch
            train_loss.reset_states()
            test_loss.reset_states()
            train_accuracy.reset_states()
            test_accuracy.reset_states()

        with fit_summary_writer.as_default():
            tf.summary.trace_export(
                name="my_func_trace",
                step=0,
                profiler_outdir=logdir
            )
    print("Best parameters is:", bestParam)
    print("Model MAE: ", defaultMAE)
