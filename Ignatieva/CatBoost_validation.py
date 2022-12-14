import numpy as np
import pandas as pd
from pathlib import Path
from joblib import load
from sklearn import metrics
import argparse
from sklearn.metrics import mean_absolute_error

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str,
                        default='data/models/LinearRegression_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--model_name', '-mn', type=str, default='CatBoost', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)
    baseline_model_path = Path(args.baseline_model)
    model = args.model_name + '.joblib'

    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    cat = load(input_model / model)

    y_pred = np.squeeze(cat.predict(X_val))

    baseline_model = load(baseline_model_path)
    y_pred_baseline = np.squeeze(baseline_model.predict(X_val))

    print(metrics.r2_score(y_val, y_pred))

    print("Baseline MAE: ", mean_absolute_error(y_val, y_pred_baseline))
    print("Model MAE: ", mean_absolute_error(y_val, y_pred))
