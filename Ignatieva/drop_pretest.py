import pandas as pd
import argparse
from pathlib import Path
import yaml


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for data_file in input_dir.glob('*.csv'):
        X_test_name = input_dir / 'X_test.csv'
        X_train_name = input_dir / 'X_train.csv'
        X_val_name = input_dir / 'X_val.csv'
        X_full_name = input_dir / 'X_full.csv'

        X_test = pd.read_csv(X_test_name)
        X_train = pd.read_csv(X_train_name)
        X_val = pd.read_csv(X_val_name)
        X_full = pd.read_csv(X_full_name)

        X_test_2 = X_test.drop("pretest", axis=1)
        X_train_2 = X_train.drop("pretest", axis=1)
        X_val_2 = X_val.drop("pretest", axis=1)
        X_full_2 = X_full.drop("pretest", axis=1)

        X_full_name_2 = output_dir / 'X_full_2.csv'
        X_train_name_2 = output_dir / 'X_train_2.csv'
        X_test_name_2 = output_dir / 'X_test_2.csv'
        X_val_name_2 = output_dir / 'X_val_2.csv'

        X_full_2.to_csv(X_full_name_2, index=False)
        X_train_2.to_csv(X_train_name_2, index=False)
        X_test_2.to_csv(X_test_name_2, index=False)
        X_val_2.to_csv(X_val_name_2, index=False)
