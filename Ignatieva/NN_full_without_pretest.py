import tensorflow as tf
import tensorboard
import datetime
import shutil
import argparse
import yaml
from NN_without_pretest import SomeModel
from pathlib import Path
import pandas as pd

tf.config.run_functions_eagerly(False)

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
    epochs = 400
    defaultMAE = 100

    #bestParams
    neurons_cnt = 64
    batch_size = 32
    learning_rate = 0.005

    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    logs_path = Path(args.logs_path)
    #if logs_path.exists():
        #shutil.rmtree(logs_path)
    #logs_path.mkdir(parents=True)

    X_full_name = input_dir / 'X_full_2.csv'
    y_full_name = input_dir / 'y_full.csv'

    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_full, y_full)).shuffle(buffer_size).batch(batch_size)

    model = SomeModel(neurons_cnt=neurons_cnt)

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = logs_path / 'gradient_tape' / current_time / 'train'
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))

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
            tf.summary.histogram('weights_in', model.d_in.get_weights()[0], step=epoch)
            tf.summary.histogram('weights_1', model.d1.get_weights()[0], step=epoch)
            tf.summary.histogram('weights_out', model.d_out.get_weights()[0], step=epoch)


        template = 'Epoch {}, Train Loss: {}, Train MAE: {}, ' \
                       'Neurons: {}, Batch Size: {}, Learning Rate: {}'
        print(template.format(epoch + 1,
                                  train_loss.result(),
                                  train_accuracy.result(),
                                  neurons_cnt,
                                  batch_size,
                                  learning_rate))

        if epoch == epochs - 1:
            buff = train_accuracy.result().numpy()
            if buff < defaultMAE:
                defaultMAE = buff
                model.save('data/models/NN_without_pretest_prod', overwrite=True)

        # Reset metrics every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

    with fit_summary_writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir
        )

    print("Model MAE: ", defaultMAE)
