import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from joblib import load

MODELS_MAPPER = {'Ridge': Ridge,
                 'LinearRegression': LinearRegression}

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)

    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    model = load(input_model)

    preds = model.predict(X_val)

    y_baseline = np.random.uniform(y_val.min(), y_val.max(), len(y_val))
    print('Model name is: ', args.model_name)
    print(model.score(X_val, y_val))
    print('Baseline mean: ', y_baseline.mean())
    print('Baseline MAE: ', mae(y_val, y_baseline))
    print('Model MAE: ', mae(y_val, preds))



