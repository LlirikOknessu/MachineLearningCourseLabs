import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

def parser_args():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    full_data = pd.read_csv('data/raw/car_data.csv')
    for name in full_data['Car_Name']:
        full_data = full_data.replace(name, name.title())
    values, counts = np.unique(full_data['Car_Name'], return_counts=True)
    unique_cars_counts = pd.DataFrame({'car names': values, 'counts': counts})
    for i in unique_cars_counts['car names']:
        idx = pd.Index(unique_cars_counts['car names'])
        full_data = full_data.replace(i, idx.get_loc(i))
    full_data = full_data.replace('Petrol', 0)
    full_data = full_data.replace('Diesel', 1)
    full_data = full_data.replace('CNG', 2)
    full_data = full_data.replace('Dealer', 0)
    full_data = full_data.replace('Individual', 1)
    full_data = full_data.replace('Manual', 0)
    full_data = full_data.replace('Automatic', 1)


    X, y = full_data.drop('Selling_Price', axis=1), full_data['Selling_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=params.get('train_test_ratio'),
                                                        random_state=params.get('random_state'))
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      train_size=params.get('train_val_ratio'),
                                                      random_state=params.get('random_state'))
    X_full_name = output_dir / 'X_full.csv'
    y_full_name = output_dir / 'y_full.csv'
    X_train_name = output_dir / 'X_train.csv'
    y_train_name = output_dir / 'y_train.csv'
    X_test_name = output_dir / 'X_test.csv'
    y_test_name = output_dir / 'y_test.csv'
    X_val_name = output_dir / 'X_val.csv'
    y_val_name = output_dir / 'y_val.csv'

    X.to_csv(X_full_name, index=False)
    y.to_csv(y_full_name, index=False)
    X_train.to_csv(X_train_name, index=False)
    y_train.to_csv(y_train_name, index=False)
    X_test.to_csv(X_test_name, index=False)
    y_test.to_csv(y_test_name, index=False)
    X_val.to_csv(X_val_name, index=False)
    y_val.to_csv(y_val_name, index=False)