import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

def to_categorical(df: pd.DataFrame):
    df.sex = pd.Categorical(df.sex)
    df = df.assign(sex_code=df.sex.cat.codes)
    df.region = pd.Categorical(df.region)
    df = df.assign(region_code=df.region.cat.codes)
    df.smoker = pd.Categorical(df.smoker)
    df = df.assign(smoker_code=df.smoker.cat.codes)
    # df.region = pd.Categorical(df.region)
    # df = df.assign(region_code=df.region.cat.codes)
    # df.age = pd.Categorical(df.age)
    # df = df.assign(age_code=df.age.cat.codes)
    # df.bmi = pd.Categorical(df.bmi)
    # df = df.assign(bmi_code=df.bmi.cat.codes)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop("region", axis=1, inplace=True)
    df.drop("age", axis=1, inplace=True)
    df.drop("bmi", axis=1, inplace=True)
    df = to_categorical(df)
    return df


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
        cleaned_data = clean_data(df=full_data)
        X, y = cleaned_data.drop("charges", axis=1), cleaned_data['charges']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=params.get('train_test_ratio'),
                                                            random_state=params.get('random_state'))
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          train_size=params.get('train_val_ratio'),
                                                          random_state=params.get('random_state'))
        X_train_name = output_dir / 'X_train.csv'
        y_train_name = output_dir / 'y_train.csv'
        X_test_name = output_dir / 'X_test.csv'
        y_test_name = output_dir / 'y_test.csv'
        X_val_name = output_dir / 'X_val.csv'
        y_val_name = output_dir / 'y_val.csv'

        X_train.to_csv(X_train_name, index=False)
        y_train.to_csv(y_train_name, index=False)
        X_test.to_csv(X_test_name, index=False)
        y_test.to_csv(y_test_name, index=False)
        X_val.to_csv(X_val_name, index=False)
        y_val.to_csv(y_val_name, index=False)


