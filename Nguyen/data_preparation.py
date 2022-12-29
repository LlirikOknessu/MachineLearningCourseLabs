from matplotlib.pyplot import sca
import pandas as pd
import datetime as dt
import numpy as np
import yaml
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw', help="path to input directory")
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared', help="path to output directory")
    parser.add_argument('--params', '-p', type=str, default="params.yaml", help="path to parameter file for dvc stage")
    return parser.parse_args()

def create_subs_for_cats(df: pd.DataFrame) -> pd.DataFrame:    
    df.fuel = pd.Categorical(df.fuel)
    df = df.assign(fuel_code=df.fuel.cat.codes)
    df.seller_type = pd.Categorical(df.seller_type)
    df = df.assign(seller_type_code=df.seller_type.cat.codes)
    df.transmission = pd.Categorical(df.transmission)
    df = df.assign(transmission_code=df.transmission.cat.codes)
    df.owner = pd.Categorical(df.owner)
    df = df.assign(owner_code=df.owner.cat.codes)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['name'], inplace=True)
    df['age'] = dt.date.today().year - df.year
    df.drop(columns=['torque'], inplace=True)
    df.drop(columns=['year'], inplace=True)
    df.drop(columns=["fuel"], inplace=True)
    df.drop(columns=["seller_type"], inplace=True)
    df.drop(columns=["transmission"], inplace=True)
    df.drop(columns=["owner"], inplace=True)
    df['mileage'] = df['mileage'].str.extract('(\d+)').astype(float)
    df['engine'] = df['engine'].str.extract('(\d+)').astype(float)
    df['max_power'] = df['max_power'].str.extract('(\d+)').astype(float)
    for x in df.index:
        if df.loc[x, "mileage"] < 5:
            df.drop(x, inplace=True)
    df.dropna(axis=0, inplace=True)
    return df

def XY_split(df: pd.DataFrame) -> pd.DataFrame:
    y = df['selling_price'].copy()
    X = df.drop(columns="selling_price").copy()
    return X, y

def train_val_test_split(X: pd.DataFrame, y: pd.DataFrame, train_ratio: float, val_test_ratio: float, random_state: int) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=val_test_ratio, random_state=random_state)
    return X_train, y_train, X_val, y_val, X_test, y_test

def data_scaling (X_full: pd.DataFrame, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                    y_full: pd.DataFrame, y_train: pd.DataFrame, y_val: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_full_scaled = scaler.transform(X_full)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)    
    X_train_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_full_df = pd.DataFrame(X_full_scaled, index=X_full.index, columns=X_full.columns)
    X_val_df = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)
    X_test_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    y_train_df = y_train.apply(np.log)
    y_full_df = y_full.apply(np.log)
    y_val_df = y_val.apply(np.log)
    y_test_df = y_test.apply(np.log)
    return X_full_df, X_train_df, X_val_df, X_test_df, y_full_df, y_train_df, y_val_df, y_test_df

def export_data(name: str, df: pd.DataFrame, out_dir: Path, index: bool, header: bool):
    df.to_csv(out_dir/(name+'.csv'), index=index, header=header)

if __name__ == '__main__':
    args = parser_args()
    with open(args.params, 'r') as f:
        params_full = yaml.safe_load(f)
    params = params_full['data_preparation']
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(in_dir/('Car_details_v3.csv'))
    df_with_subs = create_subs_for_cats(full_df)
    cleaned_df = clean_data(df_with_subs)
    X_full, y_full = XY_split(cleaned_df)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_full, y_full, train_ratio=params.get('train_ratio'),
                                                                            val_test_ratio=params.get('validation_test_ratio'),
                                                                            random_state=params.get('random_state'))                                       
    X_full_df, X_train_df, X_val_df, X_test_df, y_full_df, y_train_df, y_val_df, y_test_df = data_scaling(X_full, X_train, X_val, X_test,
                                                                                                            y_full, y_train, y_val, y_test)
                                                                                                            
    export_data("X_full", X_full_df, out_dir=out_dir, index=False, header=True)
    export_data("X_train", X_train_df, out_dir=out_dir, index=False, header=True)
    export_data("X_val", X_val_df, out_dir=out_dir, index=False, header=True)
    export_data("X_test", X_test_df, out_dir=out_dir, index=False, header=True)
    export_data("y_full", y_full_df, out_dir=out_dir, index=False, header=True)
    export_data("y_train", y_train_df, out_dir=out_dir, index=False, header=True)
    export_data("y_val", y_val_df, out_dir=out_dir, index=False, header=True)
    export_data("y_test", y_test_df, out_dir=out_dir, index=False, header=True)