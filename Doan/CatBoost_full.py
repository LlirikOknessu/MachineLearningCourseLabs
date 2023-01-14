import pandas as pd
import argparse
from pathlib import Path
from joblib import dump
from catboost import CatBoostRegressor as Cat

BEST_PARAMS = {'learning_rate': 0.8, 'max_depth': 2, 'n_estimators': 300, 'subsample': 0.15}

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to data')
    parser.add_argument('--model_name', '-mn', type=str, default='CatBoost', required=False,
                        help='params file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_train_name = input_dir / 'X_full.csv'
    y_train_name = input_dir / 'y_full.csv'
    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)

    reg = Cat(**BEST_PARAMS)

    dump(reg, output_model_joblib_path)
