import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import pandas as pd
import argparse
import numpy as np
from joblib import load
from sklearn.metrics import mean_squared_error
import math
import random

BATCH_SIZE = 64


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared', help="path to input data directory")
    parser.add_argument('--input_model', '-im', type=str, default='data/models', help="path to input model directory")
    parser.add_argument('--logs_dir', '-ld', type=str, default='data/logs', help="path to logs directory")
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib', help='path to linear regression prod version')
    return parser.parse_args()

if __name__ == '__main__':
    random.seed(42)
    args = parser_args()
    in_dir = Path(args.input_dir)
    in_model = args.input_model
    baseline_model_path = Path(args.baseline_model)

    X_test_name = in_dir / 'X_test.csv'
    y_test_name = in_dir / 'y_test.csv'
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

    net = keras.models.load_model('./data/models/NeuNet', compile=True)
    
    test_accuracy = tf.keras.metrics.MeanSquaredError(name='test_mse')

    for (X_test, y_test) in test_ds:
        predictions = net(X_test)
        test_accuracy(y_test, predictions)

    baseline_model = load(baseline_model_path)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_test))

    print("Baseline RMSE: ", math.sqrt(mean_squared_error(y_test, y_pred_baseline)))
    print("Net RMSE: ", math.sqrt(test_accuracy.result().numpy()))