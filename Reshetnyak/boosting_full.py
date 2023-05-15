import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
import catboost.core as cb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from joblib import dump, load

MODELS_MAPPER = {'XGBoostRegressor': xgb.XGBRegressor,
                 'CatBoostRegressor': cb.CatBoostRegressor}

# Set the best parameters that you get on training stage for all used models
MODELS_BEST_PARAMETERS = {
    'XGBoostRegressor': {'booster': 'gbtree', 'learning_rate': 0.003, 'max_depth': 10},
    'CatBoostRegressor': {'iterations': 100, 'learning_rate': 0.003, 'depth': 4}}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='CatBoostRegressor', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '_prod.jpg')
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_train_name = input_dir / 'X_full.csv'
    y_train_name = input_dir / 'y_full.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    y_train_cols = y_train.columns

    best_params = MODELS_BEST_PARAMETERS.get(args.model_name)
    reg = MODELS_MAPPER.get(args.model_name)(**best_params)
    if isinstance(reg, cb.CatBoostRegressor) or isinstance(reg, xgb.XGBRegressor):
            y_train = np.ravel(y_train.values)
    reg = reg.fit(X_train, y_train)
    dump(reg, output_model_joblib_path)