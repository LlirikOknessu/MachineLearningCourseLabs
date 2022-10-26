from ast import arg
from email import parser
from matplotlib import test
import pandas as pd
import datetime as dt
import yaml
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/dataset1', help="path to input directory")
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared', help="path to output directory")
    parser.add_argument('--params', '-p', type=str, default="params.yaml", help="path to parameter file for dvc stage")
    return parser.parse_args()

def create_subs_for_cats(df: pd.DataFrame) -> pd.DataFrame:
    df.fuel_type = pd.Categorical(df.fuel_type)
    df = df.assign(fuel_type_subs=df.fuel_type.cat.codes)
    df.seller_type = pd.Categorical(df.seller_type)
    df = df.assign(seller_type_subs=df.seller_type.cat.codes)
    df.transmission = pd.Categorical(df.transmission)
    df = df.assign(transmission_subs=df.transmission.cat.codes)
    df.owner = pd.Categorical(df.owner)
    df = df.assign(owner_subs=df.owner.cat.codes)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=["car_name"], inplace=True)
    df['cars_age'] = dt.date.today().year - df.year
    df.drop(columns="year", inplace=True)
    df.drop(columns=["fuel_type"], inplace=True)
    df.drop(columns=["seller_type"], inplace=True)
    df.drop(columns=["transmission"], inplace=True)
    df.drop(columns=["owner"], inplace=True)
    return df

def XY_split(df: pd.DataFrame) -> pd.DataFrame:
    y = df.iloc[:, 0]
    x = df.iloc[:, 1:]
    return x, y

def train_val_test_split(x: pd.DataFrame, y: pd.DataFrame, train_ratio: float, val_test_ratio: float, randome_state: int) -> pd.DataFrame:
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, random_state=randome_state)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=val_test_ratio, random_state=randome_state)
    return x_train, y_train, x_val, y_val, x_test, y_test

def export_data(name: str, df: pd.DataFrame, out_dir: Path, index: bool, header: bool):
    df.to_csv(out_dir/name+'.csv', index=index, header=header)

if __name__ == '__main__':
    args = parser_args()
    params = yaml.safe_load(open(args.params))
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir).mkdir(Parents=True, exist_ok=True)
    for data_file in in_dir.glob(*.csv):
        full_df = pd.read_csv(data_file)
        df_with_subs = create_subs_for_cats(full_df)
        cleaned_df = clean_data(df_with_subs)
        x_full, y_full = XY_split(cleaned_df)
        




# Load dvc files
deps = yaml.safe_load(open('dvc.yaml'))['stages']['data_preparation']['deps']
outs = yaml.safe_load(open('dvc.yaml'))['stages']['data_preparation']['outs']
params = yaml.safe_load(open('params.yaml'))['data_preparation']

# Read from dvc files
RAW_DATA_PATH = deps[0]
OUTPUT_PATH = outs[0]
TRAIN_RATIO = params['train_ratio']
VALIDATION_TEST_RATIO = params['validation_test_ratio']
RANDOM_STATE = params['random_state']

# Import & cleaning & reformat raw data
df = pd.read_csv(RAW_DATA_PATH)

df.drop(columns=["car_name"], inplace=True)
df['cars_years_old'] = dt.date.today().year - df.year
df.drop(columns="year", inplace=True)

df.fuel_type = pd.Categorical(df.fuel_type)
df = df.assign(fuel_type_subs=df.fuel_type.cat.codes)
df.seller_type = pd.Categorical(df.seller_type)
df = df.assign(seller_type_subs=df.seller_type.cat.codes)
df.transmission = pd.Categorical(df.transmission)
df = df.assign(transmission_subs=df.transmission.cat.codes)
df.owner = pd.Categorical(df.owner)
df = df.assign(owner_subs=df.owner.cat.codes)

df.drop(columns=["fuel_type"], inplace=True)
df.drop(columns=["seller_type"], inplace=True)
df.drop(columns=["transmission"], inplace=True)
df.drop(columns=["owner"], inplace=True)

# Split dependent and independent variables from dataframe
y = df.iloc[:, 0]
x = df.iloc[:, 1:]

# Train, validation, test split for X,Y
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=VALIDATION_TEST_RATIO,
                                                random_state=RANDOM_STATE)


# Export prepared data
x.to_csv(OUTPUT_PATH + 'x_full.csv', index=False, header=True)
y.to_csv(OUTPUT_PATH + 'y_full.csv', index=False, header=True)

x_train.to_csv(OUTPUT_PATH + 'x_train.csv', index=False, header=True)
y_train.to_csv(OUTPUT_PATH + 'y_train.csv', index=False, header=True)

x_val.to_csv(OUTPUT_PATH + 'x_val.csv', index=False, header=True)
y_val.to_csv(OUTPUT_PATH + 'y_val.csv', index=False, header=True)

x_test.to_csv(OUTPUT_PATH + 'x_test.csv', index=False, header=True)
y_test.to_csv(OUTPUT_PATH + 'y_test.csv', index=False, header=True)

# print(yaml.safe_load(open('dvc.yaml'))['stages']['data_preparation']['params'])
