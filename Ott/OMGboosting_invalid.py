import pandas as pd
import argparse
from pathlib import Path
from sklearn.metrics import mean_absolute_error as mae
from joblib import load
import xgboost as xgb

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/ExtraTree.joblib',
                        required=False, help='path to Extra tree version')
    parser.add_argument('--model_name', '-mn', type=str, default='XGBRegressor', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)
    baseline_model_path = Path(args.baseline_model)

    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    model = load(input_model)


    preds = model.predict(X_val)

    baseline_model_derevo = load(baseline_model_path)
    baseline_preds = baseline_model_derevo.predict(X_val)

    print('Model name is: XGBoosting')
    print('Model scoring: ', model.score(X_val, y_val))
    print('Baseline is ExtraTree')
    print('Baseline MAE: ', mae(y_val, baseline_preds))
    print('Model MAE:', mae(y_val, preds))