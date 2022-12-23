import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Model
import pandas as pd
from pathlib import Path
import argparse
import yaml
import datetime

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--logs_dir', '-logd', type=str, default='data/logs/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

class MyModel(Model):
    def __init__(self, n_of_neurons=64):
        super(MyModel, self).__init__()

        self.d_in = Dense(25, activation='relu')
        self.d_1 = Dense(n_of_neurons, activation = 'sigmoid')
        self.d_2 = Dense(n_of_neurons, activation = 'relu')
        self.d_3 = Dense(n_of_neurons, activation = 'sigmoid')
        self.d_out = Dense(1)

    def call(self, x):
        x = self.d_in(x)
        x = self.d_1(x)
        x = self.d_2(x)
        x = self.d_3(x)
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
    with open(args.params, 'r') as f:
        paramsprod = yaml.safe_load(f)
    params = paramsprod['NeuralNetwork_full']

    BATCH_SIZE = params['batch_size']
    BUFFER_SIZE = params['buffer_size']
    LEARNING_RATE = params['learning_rate']
    EPOCHS = params['epochs']
    NUMBER_OF_NEURONS = params['n_of_neurons']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    logs_path = Path(args.logs_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    output_model_path = output_dir / ('NeuralNetwork_prod')

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'

    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_full, y_full)
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    model = MyModel(n_of_neurons=NUMBER_OF_NEURONS)

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = logs_path / 'best' / 'gradient_tape' / current_time / 'train'
    train_log_dir.mkdir(exist_ok=True, parents=True)
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))

    logdir = logs_path / 'best' / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(exist_ok=True, parents=True)
    fit_summary_writer = tf.summary.create_file_writer(str(logdir))

    tf.summary.trace_on(graph=True, profiler=True)

    for epoch in range(EPOCHS):
        for (x_train, y_train) in train_ds:
            with fit_summary_writer.as_default():
                train_step(x_train, y_train, model, optimizer)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
                tf.summary.histogram('Weights_in_r', model.d_in.weights[0], step=epoch)
                tf.summary.histogram('Weights_1_r', model.d_1.weights[0], step=epoch)
                tf.summary.histogram('Weights_2_r', model.d_2.weights[0], step=epoch)
                tf.summary.histogram('Weights_3_r', model.d_3.weights[0], step=epoch)
                tf.summary.histogram('Weights_out_r', model.d_out.weights[0], step=epoch)

        template = 'Epoch: {}, Train Loss: {}, Train MAE: {}'
        print(template.format(epoch+1,
                              train_loss.result(),
                              train_accuracy.result()))
        train_loss.reset_states()
        train_accuracy.reset_states()
    with fit_summary_writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir
        )
    model.summary()
    model.save(output_model_path, overwrite=True)