import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.metrics import mean_absolute_error
from joblib import dump, load
from xgboost import XGBRegressor

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='XGBRegressor', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '_prod.jpg')
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'

    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    XGBBoosting_model = XGBRegressor()
    XGBBoosting_model.fit(X_full, y_full, verbose=False)

    predicted_values = np.squeeze(XGBBoosting_model.predict(X_full))

    print(XGBBoosting_model.score(X_full, y_full))
    print("Model MAE: ", mean_absolute_error(y_full, predicted_values))

    dump(XGBBoosting_model, output_model_joblib_path)