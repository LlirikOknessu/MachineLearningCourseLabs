import argparse
import datetime
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow import keras

BEST_PARAMETERS = {'neurons_cnt': 50, 'batch_size': 32, 'learning_rate': 0.09}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/NN_prod',
                        required=False, help='path to save prepared data')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/NN',
                        required=False, help='path to save prepared data')
    parser.add_argument('--logs_dir', '-logd', type=str, default='data/logs/',
                        required=False, help='path to save prepared data')
    return parser.parse_args()


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
    EPOCHS = 1

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_full, y_full)).shuffle(BUFFER_SIZE).batch(BEST_PARAMETERS['batch_size'])

    model = keras.models.load_model(input_model)

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

        train_loss.reset_states()
        train_accuracy.reset_states()

    with fit_summary_writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir)

    model.save(output_dir, overwrite=True)