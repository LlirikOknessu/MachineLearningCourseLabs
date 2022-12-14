import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import random
from sklearn.model_selection import GridSearchCV
import math


TREES_MODELS_MAPPER = {'DecisionTree': tree.DecisionTreeRegressor,
                       'RandomForest': RandomForestRegressor,
                       'ExtraTree': ExtraTreesRegressor}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['decision_tree']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    baseline_model_path = Path(args.baseline_model)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    random.seed(42)
    decision_tree_model = TREES_MODELS_MAPPER.get(args.model_name)()
    decision_tree_regressor = GridSearchCV(decision_tree_model, params[args.model_name], scoring='neg_root_mean_squared_error', n_jobs=-1, cv=10)

    if isinstance(decision_tree_model, RandomForestRegressor) or isinstance(decision_tree_model, ExtraTreesRegressor):
        y_train = np.ravel(y_train.values)
        y_val = np.ravel(y_val.values)
    decision_tree_regressor = decision_tree_regressor.fit(X=X_train, y=y_train)

    baseline_model = load(baseline_model_path)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_val))

    predicted_values = np.squeeze(decision_tree_regressor.predict(X_val))

    print(decision_tree_regressor.score(X_val, y_val))
    print(decision_tree_regressor.best_params_)

    print("Baseline RMSE: ", math.sqrt(mean_squared_error(y_val, y_pred_baseline)))
    print("Model RMSE: ", math.sqrt(mean_squared_error(y_val, predicted_values)))

    dump(decision_tree_regressor, output_model_joblib_path)