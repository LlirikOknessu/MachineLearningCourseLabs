import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from joblib import dump, load
from sklearn.metrics import mean_squared_error
import math

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared', help="path to input data directory")
    parser.add_argument('--input_model', '-im', type=str, default='data/models', help="path to input model directory")
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib', help='path to linear regression prod version')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args()
    in_dir = Path(args.input_dir)
    in_model = Path(args.input_model)
    baseline_model_path = Path(args.baseline_model)

    X_test_name = in_dir / 'X_test.csv'
    y_test_name = in_dir / 'y_test.csv'
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    reg = load(in_model)
    predicted_values = np.squeeze(reg.predict(X_test))

    baseline_model = load(baseline_model_path)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_test))

    print(reg.score(X_test, y_test))
    print("Baseline RMSE: ", math.sqrt(mean_squared_error(y_test, y_pred_baseline)))
    print("Model RMSE: ", math.sqrt(mean_squared_error(y_test, predicted_values)))