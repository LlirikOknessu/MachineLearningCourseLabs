import pandas as pd
from pathlib import Path
import argparse
import yaml
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split as tts

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

def imputing(X_train, X_val, X_test, X, strat='median'):
    my_imputer = SimpleImputer(strategy=strat)
    X_train_imputed = pd.DataFrame(my_imputer.fit_transform(X_train))
    X_val_imputed = pd.DataFrame(my_imputer.transform(X_val))
    X_test_imputed = pd.DataFrame(my_imputer.transform(X_test))
    X_imputed = pd.DataFrame(my_imputer.transform(X))

    X_train_imputed.columns = X_train.columns
    X_test_imputed.columns = X_test.columns
    X_val_imputed.columns = X_val.columns
    X_imputed.columns = X.columns
    return X_train_imputed, X_val_imputed, X_test_imputed, X_imputed

def data_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(axis=0, subset=['Life expectancy '])
    df = df.drop(columns='Country')
    df = df.drop(columns='infant deaths')
    df = df.drop(columns='Measles ')
    df = df.drop(columns='under-five deaths ')
    df = df.drop(columns='Population')
    df = df.drop(columns='Total expenditure')
    # developing - 0, developed - 1
    df['Status'] = df['Status'].apply(lambda x: 0 if x == 'Developing' else 1)
    cols_with_nan = [col for col in df.columns
                     if df[col].isnull().any()]
    # reduced_df = df.drop(cols_with_nan, axis=1)

    df_plus = df.copy()
    for col in cols_with_nan:
        df_plus[col + '_was_missing'] = df_plus[col].isnull()

    return df_plus


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    for data_file in input_dir.glob('*.csv'):
        full_data = pd.read_csv(data_file)
        cleaned_data = data_cleaner(df=full_data)
        X = cleaned_data.drop('Life expectancy ', axis=1)
        y = cleaned_data['Life expectancy ']
        #print(y.shape, X.shape)
        X_train, X_test, y_train, y_test = tts(X, y, train_size=params.get('train_test_ratio'),
                                               random_state=params.get('random_state'))
        X_train, X_val, y_train, y_val = tts(X_train, y_train, train_size=params.get('train_val_ratio'),
                                             random_state=params.get('random_state'))
        X_train, X_val, X_test, X = imputing(X_train, X_val, X_test, X)

        X_train_name = output_dir / 'X_train.csv'
        y_train_name = output_dir / 'y_train.csv'
        X_test_name = output_dir / 'X_test.csv'
        y_test_name = output_dir / 'y_test.csv'
        X_val_name = output_dir / 'X_val.csv'
        y_val_name = output_dir / 'y_val.csv'
        X_full_name = output_dir / 'X_full.csv'
        y_full_name = output_dir / 'y_full.csv'

        X_train.to_csv(X_train_name, index=False)
        y_train.to_csv(y_train_name, index=False)
        X_test.to_csv(X_test_name, index=False)
        y_test.to_csv(y_test_name, index=False)
        X_val.to_csv(X_val_name, index=False)
        y_val.to_csv(y_val_name, index=False)
        X.to_csv(X_full_name, index = False)
        y.to_csv(y_full_name, index = False)

