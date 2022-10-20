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
    df.experience_level = pd.Categorical(df.experience_level)
    df = df.assign(experience_level=df.experience_level.cat.codes)
    df.employment_type = pd.Categorical(df.employment_type)
    df = df.assign(employment_type=df.employment_type.cat.codes)
    df.employee_residence = pd.Categorical(df.employee_residence)
    df = df.assign(employee_residence=df.employee_residence.cat.codes)
    df.company_size = pd.Categorical(df.company_size)
    df = df.assign(company_size=df.company_size.cat.codes)
    df.company_location = pd.Categorical(df.company_location)
    df = df.assign(company_location=df.company_location.cat.codes)
    df.job_title = pd.Categorical(df.job_title)
    df = df.assign(job_title=df.job_title.cat.codes)
    df.work_year = pd.Categorical(df.work_year)
    df = df.assign(work_year=df.work_year.cat.codes)
    df.remote_ratio = pd.Categorical(df.remote_ratio)
    df = df.assign(remote_ratio=df.remote_ratio.cat.codes)
    return df


def titles_reduction(x) -> str:
    if x.find("Data Science") >= 0 or x.find("Data Scientist") >= 0:
        return 'Data Scientist'
    elif x.find("Analyst") >= 0 or x.find("Analytics") >= 0:
        return 'Data Analyst'
    elif x.find("ML") >= 0 or x.find("Machine Learning") >= 0:
        return 'Machine Learning Engineer'
    elif x.find("Data Engineer") >= 0 or x.find("Data Engineering") >= 0:
        return 'Data Engineer'
    else:
        return 'AI related'


def res(x) -> str:
    if x == "US":
        return "US"
    else:
        return "Other"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df['job_title'] = df['job_title'].apply(titles_reduction)
    df['employee_residence'] = df['employee_residence'].apply(res)
    df['company_location'] = df['company_location'].apply(res)
    df['salary_in_usd'] = np.log(df['salary_in_usd'])
    df.drop("Unnamed: 0", axis=1, inplace=True)
    df.drop("salary", axis=1, inplace=True)
    df.drop("salary_currency", axis=1, inplace=True)
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
        X, y = cleaned_data.drop("salary_in_usd", axis=1), cleaned_data['salary_in_usd']
        X_train , X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=params.get('train_test_ratio'),
                                                            random_state=params.get('random_state'))
        X_train , X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          train_size=params.get('train_val_raitio'),
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
