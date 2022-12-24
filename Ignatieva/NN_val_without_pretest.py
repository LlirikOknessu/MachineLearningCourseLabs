import keras.models
import numpy as np
from joblib import load
import argparse
import yaml
from sklearn.metrics import mean_absolute_error
from pathlib import Path
import pandas as pd


def parser_args_for_sac():
  parser = argparse.ArgumentParser(description='Paths parser')
  parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                      required=False, help='path to input data directory')
  parser.add_argument('--input_model', '-im', type=str, default='data/models/NN_without_pretest',
                      required=False, help='path with saved model')
  parser.add_argument('--baseline_model', '-bm', type=str,
                      default='data/models/LinearRegression_without_pretest_prod.joblib',
                      required=False, help='path to linear regression prod version')
  return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['neural_net']

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)
    baseline_model_path = Path(args.baseline_model)

    X_val_name = input_dir / 'X_val_2.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    model = keras.models.load_model(input_model)

    y_pred = np.squeeze(model.predict(X_val))

    baseline_model = load(baseline_model_path)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_val))

    print("Baseline MAE: ", mean_absolute_error(y_val, y_pred_baseline))
    print("Model MAE: ", mean_absolute_error(y_val, y_pred))

