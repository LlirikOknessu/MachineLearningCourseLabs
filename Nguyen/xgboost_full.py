import pandas as pd
import argparse
from pathlib import Path
import numpy as np
import xgboost as xgb
from joblib import dump, load
from sklearn.metrics import mean_squared_error
import math
import random

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared', help="path to input directory")
    parser.add_argument('--output_dir', '-od', type=str, default='data/models', help="path to output directory")
    return parser.parse_args()

params = {
    'n_estimators' : 50,
    'max_depth': 7,
    'eta': 0.3,
    'subsample': 1.0
}

if __name__ == '__main__':
    random.seed(42)
    args = parser_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = out_dir / 'xgboost_prod.joblib'

    X_full_name = in_dir / 'X_full.csv'
    y_full_name = in_dir / 'y_full.csv'
    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    xgb_reg = xgb.XGBRegressor(n_estimators=params.get('n_estimators'), 
                                max_depth=params.get('max_depth'), 
                                eta=params.get('eta'), 
                                subsample=params.get('subsample'))
    xgb_reg.fit(X_full, y_full)
    print(xgb_reg.score)
    print(xgb_reg)
    dump(xgb_reg, output_model_joblib_path)