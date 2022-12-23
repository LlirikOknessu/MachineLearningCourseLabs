import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Model
from pathlib import Path
import pandas as pd
import datetime
import shutil
import argparse
import numpy as np
from joblib import load
from keras.callbacks import TensorBoard
import random

#Need to change these parameters to best values got in training stage
BATCH_SIZE = 32
BUFFER_SIZE = 32
NEURONS = 128
LEARNING_RATE = 0.001
EPOCHS = 50

tf.config.run_functions_eagerly(True)

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared', help="path to input directory")
    parser.add_argument('--output_dir', '-od', type=str, default='data/models', help="path to output directory")
    parser.add_argument('--logs_dir', '-ld', type=str, default='data/logs_prod', help="path to logs directory")    
    return parser.parse_args()

class NeuNet(Model):
    def __init__(self, neurons=128):
        super(NeuNet, self).__init__()        
        self.in_layer = Dense(10, activation='relu')
        self.hidden_1 = Dense(neurons, activation='relu')
        self.hidden_2 = Dense(neurons, activation='relu')
        self.hidden_3 = Dense(neurons, activation='relu')
        self.hidden_4 = Dense(neurons, activation='relu')
        self.hidden_5 = Dense(neurons, activation='relu')
        self.hidden_6 = Dense(neurons, activation='relu')
        self.hidden_7 = Dense(neurons, activation='relu')
        self.hidden_8 = Dense(neurons, activation='relu')
        self.hidden_9 = Dense(neurons, activation='relu')
        self.hidden_10 = Dense(neurons, activation='relu')
        self.out_layer = Dense(1, activation='relu')

    def call(self, inputs):
        x = self.in_layer(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.hidden_3(x)
        x = self.hidden_4(x)
        x = self.hidden_5(x)
        x = self.hidden_6(x)
        x = self.hidden_7(x)
        x = self.hidden_8(x)
        x = self.hidden_9(x)
        x = self.hidden_10(x)
        outputs = self.out_layer(x)
        return outputs

@tf.function
def train_net(data: pd.DataFrame, labels: pd.DataFrame, net: NeuNet, optimizer: tf.keras.optimizers, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        predictions = net(data)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


if __name__ == '__main__':
    args = parser_args()   
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    logs_path = Path(args.logs_dir)

    if logs_path.exists():
        shutil.rmtree(logs_path)
    out_dir.mkdir(exist_ok=True, parents=True)

    X_full_name = in_dir / 'X_full.csv'
    y_full_name = in_dir / 'y_full.csv'

    X_train = pd.read_csv(X_full_name)
    y_train = pd.read_csv(y_full_name)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    net = NeuNet(neurons=NEURONS)
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_mse')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = logs_path / 'gradient_tape' / current_time / 'train'
    train_log_dir.mkdir(exist_ok=True, parents=True)
    train_summary_writer = tf.summary.create_file_writer(str(train_log_dir))

    logdir = logs_path / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(exist_ok=True, parents=True)
    fit_summary_writer = tf.summary.create_file_writer(str(logdir))

    tf.summary.trace_on(graph=True, profiler=True)
                            
    for epoch in range(EPOCHS):
        for (X_train, y_train) in train_ds:
            with fit_summary_writer.as_default():
                train_net(X_train, y_train, net, optimizer, train_loss, train_accuracy)
                
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            tf.summary.histogram('weights_in_layer', net.in_layer.weights[0], step=epoch)
            tf.summary.histogram('biases_in_layer', net.in_layer.weights[1], step=epoch)

            tf.summary.histogram('weights_hidden_1', net.in_layer.weights[0], step=epoch)
            tf.summary.histogram('biases_hidden_1', net.in_layer.weights[1], step=epoch)
            
            tf.summary.histogram('weights_hidden_2', net.in_layer.weights[0], step=epoch)
            tf.summary.histogram('biases_hidden_2', net.in_layer.weights[1], step=epoch)

            tf.summary.histogram('weights_hidden_3', net.in_layer.weights[0], step=epoch)
            tf.summary.histogram('biases_hidden_3', net.in_layer.weights[1], step=epoch)

            tf.summary.histogram('weights_hidden_4', net.in_layer.weights[0], step=epoch)
            tf.summary.histogram('biases_hidden_4', net.in_layer.weights[1], step=epoch)

            tf.summary.histogram('weights_hidden_5', net.in_layer.weights[0], step=epoch)
            tf.summary.histogram('biases_hidden_5', net.in_layer.weights[1], step=epoch)

            tf.summary.histogram('weights_hidden_6', net.in_layer.weights[0], step=epoch)
            tf.summary.histogram('biases_hidden_6', net.in_layer.weights[1], step=epoch)

            tf.summary.histogram('weights_hidden_7', net.in_layer.weights[0], step=epoch)
            tf.summary.histogram('biases_hidden_7', net.in_layer.weights[1], step=epoch)

            tf.summary.histogram('weights_hidden_8', net.in_layer.weights[0], step=epoch)
            tf.summary.histogram('biases_hidden_8', net.in_layer.weights[1], step=epoch)

            tf.summary.histogram('weights_hidden_9', net.in_layer.weights[0], step=epoch)
            tf.summary.histogram('biases_hidden_9', net.in_layer.weights[1], step=epoch)

            tf.summary.histogram('weights_hidden_10', net.in_layer.weights[0], step=epoch)
            tf.summary.histogram('biases_hidden_10', net.in_layer.weights[1], step=epoch)

            tf.summary.histogram('weights_out_layer', net.in_layer.weights[0], step=epoch)
            tf.summary.histogram('biases_out_layer', net.in_layer.weights[1], step=epoch)


        template = 'Epoch {}, Loss: {}, Accuracy: {}'
        print (template.format(epoch+1,
                                train_loss.result(),
                                train_accuracy.result()))

        if epoch == EPOCHS - 1:
            print("Accuracy: ", train_accuracy.result().numpy())

        # Reset metrics every epoch
        train_loss.reset_states()
        train_accuracy.reset_states()

    with fit_summary_writer.as_default():
        tf.summary.trace_export(
            name="my_func_trace",
            step=0,
            profiler_outdir=logdir
        )

    net.summary()
    net.save('./data/models/NeuNetProd', overwrite=True)




                
    
    





