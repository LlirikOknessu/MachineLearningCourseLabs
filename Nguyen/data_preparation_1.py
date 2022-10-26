from ast import arg
from email import parser
import pandas as pd
import datetime as dt
import yaml
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

#Parsing path for dependencies
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/dataset1', help="path to input directory")
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared', help="path to output directory")
    parser.add_argument('--params', '-p', type=str, default="params.yaml", help="path to parameter file for dvc stage")
    return parser.parse_args()

#Creating substitute for categorical variables
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

#Removing unused columns in data frame
#In this case there is no need to reformating data. Data's already been reformatted when combining different data files
#For other cases, formating data maybe needed.
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=["car_name"], inplace=True)
    df['cars_age'] = dt.date.today().year - df.year
    df.drop(columns="year", inplace=True)
    df.drop(columns=["fuel_type"], inplace=True)
    df.drop(columns=["seller_type"], inplace=True)
    df.drop(columns=["transmission"], inplace=True)
    df.drop(columns=["owner"], inplace=True)
    return df

#Spliting data into 2 data frame for dependent and independent variables
def XY_split(df: pd.DataFrame) -> pd.DataFrame:
    y = df.iloc[:, 0]
    x = df.iloc[:, 1:]
    return x, y

#Spliting data into dataset for train, validation and test
#train_ratio show ratio in full dataset between train set and validation+test sets
#val_test_ratio show ratio between validation set and test set in remain data (full - train)
def train_val_test_split(x: pd.DataFrame, y: pd.DataFrame, train_ratio: float, val_test_ratio: float, random_state: int) -> pd.DataFrame:
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=val_test_ratio, random_state=random_state)
    return x_train, y_train, x_val, y_val, x_test, y_test

#Exporting a data frame outto the output directory
def export_data(name: str, df: pd.DataFrame, out_dir: Path, index: bool, header: bool):
        df.to_csv(out_dir/(name+'.csv'), index=index, header=header)

if __name__ == '__main__':
    args = parser_args()
    params = yaml.safe_load(open(args.params))
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for data_file in in_dir.glob('*.csv'):
        full_df = pd.read_csv(data_file)
        df_with_subs = create_subs_for_cats(full_df)
        cleaned_df = clean_data(df_with_subs)
        x_full, y_full = XY_split(cleaned_df)
        x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x_full, y_full, train_ratio=params.get('train_ratio'),
                                                                            val_test_ratio=params.get('validation_test_ratio'),
                                                                            random_state=params.get('random_state'))
    export_data("X_full", x_full, out_dir=out_dir, index=False, header=True)
    export_data("x_train", x_train, out_dir=out_dir, index=False, header=True)
    export_data("x_val", x_val, out_dir=out_dir, index=False, header=True)
    export_data("x_test", x_test, out_dir=out_dir, index=False, header=True)
    export_data("Y_full", y_full, out_dir=out_dir, index=False, header=True)
    export_data("y_train", y_train, out_dir=out_dir, index=False, header=True)
    export_data("y_val", y_val, out_dir=out_dir, index=False, header=True)
    export_data("y_test", y_test, out_dir=out_dir, index=False, header=True)