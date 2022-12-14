import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from joblib import dump
import math
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

LINEAR_MODEL_MAPPER = {'LinearRegression': LinearRegression,
                        'Ridge': Ridge, 
                        'Lasso': Lasso, 
                        'ElasticNet': ElasticNet}

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared', help="path to input directory")
    parser.add_argument('--output_dir', '-od', type=str, default='data/models', help="path to output directory")
    parser.add_argument('--model_name', '-mn', type=str, default='LinearRegression', help="model used for regression")
    parser.add_argument('--params', '-p', type=str, default="params.yaml", help="path to parameter file for dvc stage")
    return parser.parse_args()

def import_data(name: str, in_dir: Path):
    df = pd.read_csv(in_dir/(name+'.csv'))
    return df

def switch(model: str, alpha: float, l1_ratio: float):
    if model == 'LinearRegression':
        reg = LinearRegression()
        return reg
    elif model == 'Ridge':
        reg = Ridge(alpha=alpha)
        return reg
    elif model == 'Lasso':
        reg = Lasso(alpha=alpha)
        return reg
    elif model == 'ElasticNet':
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        return reg

if __name__ == '__main__':
    args = parser_args()
    with open(args.params, 'r') as f:
        params_full = yaml.safe_load(f)
    params = params_full['regression']
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = out_dir / (args.model_name + '_prod.csv')
    output_model_joblib_path = out_dir / (args.model_name + '_prod.joblib')

    X_train = import_data("X_full", in_dir)
    y_train = import_data("y_full", in_dir)
    
    reg = switch(model=args.model_name, alpha=params.get('alpha'), l1_ratio=params.get('l1_ratio'))

    reg.fit(X_train, y_train)

    intercept = reg.intercept_.astype(float)
    coefficients = reg.coef_.astype(float)
    intercept = pd.Series(intercept, name='intercept')
    coefficients = pd.Series(coefficients[0], name='coefficients')
    print("Intercept: ", intercept)
    print("List of coefficients: ", coefficients)
    out_model = pd.DataFrame([coefficients, intercept])
    out_model.to_csv(output_model_path, index = False)

    dump(reg, output_model_joblib_path)   