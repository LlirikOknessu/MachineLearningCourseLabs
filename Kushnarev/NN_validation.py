import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load
from sklearn.metrics import mean_absolute_error
from tensorflow import keras


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/NN',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/DecisionTree_prod.joblib',
                        required=False, help='path to linear regression prod version')
    return parser.parse_args()


@tf.function
def test_step(model, input_vector, labels):
  predictions = model(input_vector, training=False)
  test_accuracy(labels, predictions)


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)
    baseline_model_path = Path(args.baseline_model)

    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    loaded_model = keras.models.load_model(input_model)
    test_accuracy = tf.keras.metrics.MeanAbsoluteError(name='test_mae')
    test_step(loaded_model, X_val, y_val)

    baseline_model = load(baseline_model_path)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_val))

    print("Baseline MAE: ", mean_absolute_error(y_val, y_pred_baseline))
    print("Model MAE: ", test_accuracy.result().numpy())