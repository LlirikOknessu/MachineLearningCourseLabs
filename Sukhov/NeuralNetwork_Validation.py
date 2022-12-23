import tensorflow as tf
import pandas as pd
from pathlib import Path
import argparse
from joblib import load
from sklearn.metrics import mean_absolute_error as mae


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/NeuralNetwork',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/ExtraTree_prod.joblib',
                        required=False, help='path to ExtraTree prod version')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)
    baseline_model_path = Path(args.baseline_model)

    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    model = tf.keras.models.load_model(input_model)
    baseline_model = load(baseline_model_path)
    model_preds = model.predict(X_val)
    baseline_preds = baseline_model.predict(X_val)

    print('Baseline MAE: ', mae(y_val, baseline_preds))
    print('NN Model MAE: ', mae(y_val, model_preds))