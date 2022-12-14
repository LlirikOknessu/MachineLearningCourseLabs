import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
from joblib import load


def parser_args():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)

    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    reg = load(input_model)

    predicted_values = np.squeeze(reg.predict(X_val))

    y_mean = y_val.mean()
    y_pred_baseline = [y_mean] * len(y_val)

    print(reg.score(X_val, y_val))
    print("Mean: ", y_mean)
    print("Baseline MAE: ", mean_absolute_error(y_val, y_pred_baseline))
    print("Model MAE: ", mean_absolute_error(y_val, predicted_values))