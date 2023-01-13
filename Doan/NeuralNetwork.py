import tensorflow as tf
import datetime
import shutil
import argparse
import yaml
import pandas as pd
from joblib import load
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import Model
from pathlib import Path
from sklearn.metrics import mean_absolute_error as mae

tf.config.run_functions_eagerly(True)

def parser_args_fors_sac():
  parser = argparse.ArgumentParser(description='Path parser')
  parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                      required=False, help='path to input data directory')
  parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                      required=False, help='path to save prepared data')
  parser.add_argument('--logs_dir', '-ld', type=str, default='data/logs/',
                      required=False, help='path to save prepared data/')
  parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                      required=False, help='path to linear regression prod version')
  parser.add_argument('--params', '-p', type=str, default='params.yaml',
                      required=False, help='file with dvc stage params')
  return parser.parse_args()


class NeuralNetwork_model(Model):
    def __init__(self, number_of_neurons):
        super(NeuralNetwork_model, self).__init__()
        self.d_in = Dense(25, activation='relu')
        self.d_1 = Dense(number_of_neurons, activation='sigmoid')
        self.d_2 = Dense(number_of_neurons, activation='relu')
        self.d_out = Dense(1)

    def call(self, x):
        x = self.d_in(x)
        x = self.d_1(x)
        x = self.d_2(x)
        x = self.d_out(x)
        return x

def get_log_dirs(params):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = logs_dir / params / 'gradient_tape' / current_time / 'train'
    train_log_dir.mkdir(exist_ok=True, parents=True)
    test_log_dir = logs_dir / params / 'gradient_tape' / current_time / 'test'
    test_log_dir.mkdir(exist_ok=True, parents=True)
    fit_log_dir = logs_dir / params / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fit_log_dir.mkdir(exist_ok=True, parents=True)
    return train_log_dir, test_log_dir, fit_log_dir

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
    test_loss(loss_object(labels, predictions))
    test_accuracy(labels, predictions)



if __name__ == '__main__':
    args = parser_args_fors_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['NeuralNetwork']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    logs_dir = Path(args.logs_dir)
    baseline_model_path = Path(args.baseline_model)
    output_model_path = output_dir / 'NeuralNetwork'
    if logs_dir.exists():
        shutil.rmtree(logs_dir)
    logs_dir.mkdir(exist_ok=True, parents=True)

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    number_of_neurons = params['n_of_neurons']
    batch_size = params['batch_size']
    buffer_size = params['buffer_size']
    learning_rate = params['learning_rate']
    epochs = params['epochs']

    best_mae = 100
    best_params = []
    best_model_mae = None

    loss_object = tf.keras.losses.MeanAbsoluteError()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_mae')

    for current_batch in batch_size:
        train_ds = tf.data.Dataset.from_tensor_slices(
            (pd.read_csv(X_train_name), pd.read_csv(y_train_name))).shuffle(buffer_size).batch(current_batch)
        test_ds = tf.data.Dataset.from_tensor_slices(
            (pd.read_csv(X_test_name), pd.read_csv(y_test_name))).batch(current_batch)

        for current_learning_rate in learning_rate:
            for current_n_of_neuron in number_of_neurons:
                params = 'batch: {}, learning_rate: {}, n_neuron: {}'.format(
                    current_batch, current_learning_rate, current_n_of_neuron)
                print("Params: Batch_Size={}, Learning_Rate={}, Number_Of_Neuron={}".format(
                    current_batch, current_learning_rate, current_n_of_neuron))
                train_log_dir, test_log_dir, fit_log_dir = get_log_dirs(params)
                train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))
                test_summary_writer = tf.summary.create_file_writer(str(test_log_dir))
                fit_summary_writer = tf.summary.create_file_writer(str(fit_log_dir))
                tf.summary.trace_on(graph=True, profiler=True)
                model = NeuralNetwork_model(number_of_neurons=current_n_of_neuron)
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

                for current_epoch in range(epochs):
                    for (x_train_current, y_train_current) in train_ds:
                        with fit_summary_writer.as_default():
                            train_step(x_train_current, y_train_current, model, optimizer)

                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss.result(), step=current_epoch)
                        tf.summary.scalar('accuracy', train_accuracy.result(), step=current_epoch)

                    for (x_test, y_test) in test_ds:
                        test_step(x_test, y_test, model)

                    with test_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss.result(), step=current_epoch)
                        tf.summary.scalar('mae', test_accuracy.result(), step=current_epoch)

                    template = 'Epoch: {}, Train_Loss: {}, Train_MAE: {}, Test_Loss: {}, Test_MAE: {}'
                    print(template.format(current_epoch + 1,
                                          train_loss.result(),
                                          train_accuracy.result(),
                                          test_loss.result(),
                                          test_accuracy.result()))
                    if current_epoch + 1 == epochs:
                        if best_mae > test_accuracy.result():
                            best_model_mae = test_accuracy.result()
                            best_params = [current_batch, current_learning_rate, current_n_of_neuron]
                            best_mae = test_accuracy.result()
                            model.save(output_model_path, overwrite=True)

                    train_loss.reset_state()
                    train_accuracy.reset_state()
                    test_loss.reset_state()
                    test_accuracy.reset_state()

                with fit_summary_writer.as_default():
                    tf.summary.trace_export(
                        name="Func_Trace",
                        step=0,
                        profiler_outdir=logs_dir)

    print('Best Params: Batch_Size={}, Learning_Rate={}, Number_Of_Neurons={}'.format(best_params[0],
                                                                                      best_params[1],
                                                                                      best_params[2]))

    baseline_model = load(baseline_model_path)
    baseline_preds = baseline_model.predict(pd.read_csv(X_test_name))
    print('Baseline MAE: ', mae(pd.read_csv(y_test_name), baseline_preds))
    print('Neural Network MAE: ', best_model_mae)