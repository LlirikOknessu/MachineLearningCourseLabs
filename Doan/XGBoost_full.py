import pandas as pd
import argparse
from pathlib import Path
from xgboost import XGBRegressor as XGB
from joblib import dump

BEST = {'gamma': 2, 'learning_rate': 0.5, 'max_depth': 5,  'n_estimators':20, 'min_child_weight':5}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to data')
    parser.add_argument('--model_name', '-mn', type=str, default='XGBoost', required=False,
                        help='params file')
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

    reg = XGB(**BEST)

    dump(reg, output_model_joblib_path)
