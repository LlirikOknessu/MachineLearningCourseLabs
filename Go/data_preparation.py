import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description="Paths parser")
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml',
                        required=False, help='file with dvc stage params')
    return parser.parse_args()

def to_catagorical(df: pd.DataFrame):
    df.age = pd.Categorical(df.age)
    df = df.assign(age=df.age.cat.codes)
    df.bmi = pd.Categorical(df.bmi)
    df = df.assign(bmi=df.bmi.cat.codes)
    df.smoker = pd.Categorical(df.smoker)
    df = df.assign(smoker=df.smoker.cat.codes)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop("sex", axis=1, inplace=True)
    df.drop("region", axis=1, inplace=True)
    df.drop("children", axis=1, inplace=True)
    df = to_catagorical(df)
    return df

if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)  # create a subdirectory

    for data_file in input_dir.glob('*.csv'):  # find .csv file
        full_data = pd.read_csv(data_file)
        cleaned_data = clean_data(df=full_data)
        X = cleaned_data.drop("charges", axis=1)  # clean the Charges in file x
        y = np.log(cleaned_data['charges'])  # save the Charges value in file y
        X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                            train_size=params.get('train_test_ratio'),
                                                            random_state=params.get('random_state'))
        X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,
                                                          train_size=params.get('train_test_ratio'),
                                                          random_state=params.get('random_state'))

        X_train_name = output_dir / 'X_train.csv'
        y_train_name = output_dir / 'y_train.csv'
        X_test_name = output_dir / 'X_test.csv'
        y_test_name = output_dir / 'y_test.csv'
        X_val_name = output_dir / 'X_val.csv'
        y_val_name = output_dir / 'y_val.csv'
        X_full_name = output_dir / 'X_full.csv'
        y_full_name = output_dir / 'y_full.csv'

        X.to_csv(X_full_name, index=False)
        y.to_csv(y_full_name, index=False)
        X_train.to_csv(X_train_name, index=False)
        y_train.to_csv(y_train_name, index=False)
        X_test.to_csv(X_test_name, index=False)
        y_test.to_csv(y_test_name, index=False)
        X_val.to_csv(X_val_name, index=False)
        y_val.to_csv(y_val_name, index=False)



