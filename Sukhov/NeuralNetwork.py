import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Model
import pandas as pd
from pathlib import Path
import argparse
import yaml
import datetime
from joblib import load
from sklearn.metrics import mean_absolute_error as mae


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--logs_dir', '-logd', type=str, default='data/logs/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/ExtraTree_prod.joblib',
                        required=False, help='path to ExtraTree prod version')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


def get_log_dirs(params):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = logs_path / params / 'gradient_tape' / current_time / 'train'
    train_log_dir.mkdir(exist_ok=True, parents=True)
    test_log_dir = logs_path / params / 'gradient_tape' / current_time / 'test'
    test_log_dir.mkdir(exist_ok=True, parents=True)
    fit_log_dir = logs_path / params / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fit_log_dir.mkdir(exist_ok=True, parents=True)

    return train_log_dir, test_log_dir, fit_log_dir


class MyModel(Model):
    def __init__(self, n_of_neurons=10):
        super(MyModel, self).__init__()

        self.d_in = Dense(25, activation='relu')
        self.d_1 = Dense(n_of_neurons, activation='sigmoid')
        self.d_2 = Dense(n_of_neurons, activation='relu')
        self.d_3 = Dense(n_of_neurons, activation='sigmoid')
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


@tf.function
def test_step(input_vector, labels, model):
    predictions = model(input_vector, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


tf.config.run_functions_eagerly(True)

if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_a = yaml.safe_load(f)
    params = params_a['NeuralNetwork']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    logs_path = Path(args.logs_dir)
    baseline_model_path = Path(args.baseline_model)
    output_model_path = output_dir / ('NeuralNetwork')

    output_dir.mkdir(exist_ok=True, parents=True)

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    BATCH_SIZE = params['batch_size']
    BUFFER_SIZE = params['buffer_size']
    LEARNING_RATE = params['learning_rate']
    EPOCHS = params['epochs']
    NUMBER_OF_NEURONS = params['n_of_neurons']

    top_mae = 100
    top_params = []
    best_model_mae = None

    loss_object = tf.keras.losses.MeanSquaredError()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_mae')

    for batch in BATCH_SIZE:
        # print(batch)
        train_ds = tf.data.Dataset.from_tensor_slices((pd.read_csv(X_train_name), pd.read_csv(y_train_name))).shuffle(
            BUFFER_SIZE).batch(batch)
        test_ds = tf.data.Dataset.from_tensor_slices((pd.read_csv(X_test_name), pd.read_csv(y_test_name))).batch(batch)
        for lr in LEARNING_RATE:
            for n_neurons in NUMBER_OF_NEURONS:
                params = 'ba{}_lr{}_nn{}'.format(batch, lr, n_neurons)
                train_log_dir, test_log_dir, logdir = get_log_dirs(params)
                train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
                test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))
                fit_summary_writer = tf.summary.create_file_writer(str(logdir))
                print("Params are: BATCH_SIZE={}, LEARNING_RATE={}, NUMBER_OF_NEURONS={}".format(batch, lr, n_neurons))
                tf.summary.trace_on(graph=True, profiler=True)
                model = MyModel(n_of_neurons=n_neurons)
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

                for epoch in range(EPOCHS):
                    for (x_train, y_train) in train_ds:
                        with fit_summary_writer.as_default():
                            train_step(x_train, y_train, model, optimizer)

                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss.result(), step=epoch)
                        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

                    for (x_test, y_test) in test_ds:
                        test_step(x_test, y_test, model)

                    with test_summary_writer.as_default():
                        tf.summary.scalar('loss', test_loss.result(), step=epoch)
                        tf.summary.scalar('mae', test_accuracy.result(), step=epoch)

                    template = 'Epoch: {}, Train Loss: {}, Train MAE: {}, Test Loss: {}, Test MAE: {}'
                    print(template.format(epoch + 1,
                                          train_loss.result(),
                                          train_accuracy.result(),
                                          test_loss.result(),
                                          test_accuracy.result()))
                    if epoch == EPOCHS - 1:
                        if top_mae > test_accuracy.result():
                            best_model_mae = test_accuracy.result()
                            top_params = [batch, lr, n_neurons]
                            top_mae = test_accuracy.result()
                            model.save(output_model_path, overwrite=True)

                    train_loss.reset_states()
                    train_accuracy.reset_states()
                    test_loss.reset_states()
                    test_accuracy.reset_states()

                with fit_summary_writer.as_default():
                    tf.summary.trace_export(
                        name="my_func_trace",
                        step=0,
                        profiler_outdir=logdir
                    )

    print(
        'Best params: BATCH_SIZE={}, LEARNING_RATE={}, NUMBER_OF_NEURONS={}'.format(
            top_params[0],
            top_params[1],
            top_params[2])
    )

    baseline_model = load(baseline_model_path)
    baseline_preds = baseline_model.predict(pd.read_csv(X_test_name))
    print('Baseline MAE: ', mae(pd.read_csv(y_test_name), baseline_preds))
    print('NN Model MAE: ', best_model_mae)
