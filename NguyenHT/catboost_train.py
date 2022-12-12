import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from catboost import CatBoostRegressor
from joblib import dump, load
from sklearn.metrics import mean_squared_error
import random
from sklearn.model_selection import GridSearchCV
import math

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared', help="path to input directory")
    parser.add_argument('--output_dir', '-od', type=str, default='data/models', help="path to output directory")
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib', help='path to linear regression prod version')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['catboost']

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    baseline_model_path = Path(args.baseline_model)
    
    out_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = out_dir / 'catboost.joblib'

    X_train_name = in_dir / 'X_train.csv'
    y_train_name = in_dir / 'y_train.csv'
    X_val_name = in_dir / 'X_val.csv'
    y_val_name = in_dir / 'y_val.csv'
    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    random.seed(42)
    cb_reg = GridSearchCV(CatBoostRegressor(), param_grid=params, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=10, verbose=0)
    cb_reg = cb_reg.fit(X=X_train, y=y_train)
    print(cb_reg.best_params_)
    predicted_values = np.squeeze(cb_reg.predict(X_val))

    baseline_model = load(baseline_model_path)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_val))

    print("Baseline RMSE: ", math.sqrt(mean_squared_error(y_val, y_pred_baseline)))
    print("Model RMSE: ", math.sqrt(mean_squared_error(y_val, predicted_values)))

    dump(cb_reg, output_model_joblib_path)

    
