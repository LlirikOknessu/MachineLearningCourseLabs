import pandas as pd
import argparse
from pathlib import Path
import numpy as np
import catboost as cb
from joblib import dump, load
from sklearn.metrics import mean_squared_error
import math

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared', help="path to input directory")
    parser.add_argument('--output_dir', '-od', type=str, default='data/models', help="path to output directory")
    return parser.parse_args()

params = {
    'n_estimators' : 10,
    'max_depth': 4,
    'eta': 0.1,
    'subsample': 0.9
}

if __name__ == '__main__':
    args = parser_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = out_dir / 'catboost_prod.joblib'

    X_full_name = in_dir / 'X_full.csv'
    y_full_name = in_dir / 'y_full.csv'
    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    cb_reg = cb.CatBoostRegressor(n_estimators=params.get('n_estimators'), 
                                max_depth=params.get('max_depth'), 
                                eta=params.get('eta'), 
                                subsample=params.get('subsample'))
    cb_reg.fit(X_full, y_full)
    print(cb_reg.score)
    print(cb_reg)
    dump(cb_reg, output_model_joblib_path)