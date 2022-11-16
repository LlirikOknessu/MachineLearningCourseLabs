import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from joblib import load
import math

TREES_MODELS_MAPPER = {'DecisionTree': tree.DecisionTreeRegressor,
                       'RandomForest': RandomForestRegressor,
                       'ExtraTree': ExtraTreesRegressor}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)
    baseline_model_path = Path(args.baseline_model)

    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    reg = load(input_model)

    predicted_values = np.squeeze(reg.predict(X_test))

    baseline_model = load(baseline_model_path)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_test))

    print(reg.score(X_test, y_test))
    print("Baseline RMSE: ", math.sqrt(mean_squared_error(y_test, y_pred_baseline)))
    print("Model RMSE: ", math.sqrt(mean_squared_error(y_test, predicted_values)))