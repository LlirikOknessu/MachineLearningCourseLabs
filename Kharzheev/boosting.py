import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
import catboost.core as cb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from joblib import dump, load
import random
from sklearn.model_selection import GridSearchCV

MODELS_MAPPER = {'XGBoostRegressor': xgb.XGBRegressor,
                 'CatBoostRegressor': cb.CatBoostRegressor}

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--model_name', '-mn', type=str, default='CatBoostRegressor', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['boosting']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    baseline_model_path = Path(args.baseline_model)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)
    random.seed(42)
    boosting_model = MODELS_MAPPER.get(args.model_name)()
    boosting_regressor = GridSearchCV(boosting_model, params[args.model_name])
    if isinstance(boosting_model, cb.CatBoostRegressor) or isinstance(boosting_model, xgb.XGBRegressor):
        y_train = np.ravel(y_train.values)
        y_test = np.ravel(y_test.values)
    boosting_regressor = boosting_regressor.fit(X=X_train, y=y_train)
    baseline_model = load(baseline_model_path)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_test))

    predicted_values = np.squeeze(boosting_regressor.predict(X_test))

    print(boosting_regressor.score(X_test, y_test))
    print(boosting_regressor.best_params_)

    print("Baseline MAE: ", mean_absolute_error(y_test, y_pred_baseline))
    print("Model MAE: ", mean_absolute_error(y_test, predicted_values))

    dump(boosting_regressor, output_model_joblib_path)
