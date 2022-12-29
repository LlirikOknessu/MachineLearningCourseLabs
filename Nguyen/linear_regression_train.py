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
import random

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

def dumping(data: pd.DataFrame, labels: pd.DataFrame, path_csv: Path, path_joblib: Path, model: str, alpha: float, l1_ratio: float):
    if model == 'LinearRegression':
        reg = LinearRegression()
    elif model == 'Ridge':
        reg = Ridge(alpha=alpha)
    elif model == 'Lasso':
        reg = Lasso(alpha=alpha)
    elif model == 'ElasticNet':
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    reg.fit(data, labels)
    predicted_value = reg.predict(data)
    print(model)    
    print("Model score: ", reg.score(data, labels))
    print("RMSE: ", math.sqrt(mean_squared_error(labels, predicted_value)))
    intercept = reg.intercept_.astype(float)
    coefficients = reg.coef_.astype(float)
    intercept = pd.Series(intercept, name='intercept')
    coefficients = pd.Series(coefficients[0], name='coefficients')
    print("Intercept: ", intercept)
    print("List of coefficients: ", coefficients)
    out_model = pd.DataFrame([coefficients, intercept])
    out_model.to_csv(path_csv, index = False)
    dump(reg, path_joblib)
    

def tuning(model: str, data: pd.DataFrame, labels: pd.DataFrame, alpha_start: float, alpha_end: float, alpha_step: float, l1_ratio_start = float, l1_ratio_end = float, l1_ratio_step = float):
    if model == 'LinearRegression':        
        return
    elif model == 'Ridge':
        search = GridSearchCV(Ridge(), param_grid={'alpha': np.arange(alpha_start, alpha_end, alpha_step)}, scoring='neg_root_mean_squared_error', verbose=0, n_jobs=-1, cv=10)
        search.fit(data,labels)
        print(search.best_params_)
        print("Best score: ",search.best_score_)
        return 
    elif model == 'Lasso':
        search = GridSearchCV(Lasso(), param_grid={'alpha': np.arange(alpha_start, alpha_end, alpha_step)}, scoring='neg_root_mean_squared_error', verbose=0, n_jobs=-1, cv=10)
        search.fit(data,labels)
        print(search.best_params_)
        print("Best score: ",search.best_score_)
        return
    elif model == 'ElasticNet':
        search = GridSearchCV(ElasticNet(), param_grid={'alpha': np.arange(alpha_start, alpha_end, alpha_step), 'l1_ratio': np.arange(l1_ratio_start, l1_ratio_end, l1_ratio_step)}, scoring='neg_root_mean_squared_error', verbose=0, n_jobs=-1, cv=10)
        search.fit(data,labels)
        print(search.best_params_)
        print("Best score: ", search.best_score_)
        return

if __name__ == '__main__':
    random.seed(42)
    args = parser_args()
    with open(args.params, 'r') as f:
        params_full = yaml.safe_load(f)
    params = params_full['regression']
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    output_model_path_csv = out_dir / (args.model_name + '.csv')
    output_model_path_joblib = out_dir / (args.model_name + '.joblib')

    X_train = import_data("X_train", in_dir)
    y_train = import_data("y_train", in_dir)

    y_mean = [y_train.mean()] * len(y_train)
    y_norm_distr = np.random.normal(y_train.mean(), y_train.std(), len(y_train))
    y_unif_distr = np.random.uniform(y_train.min(), y_train.max(), len(y_train))

    print("Baseline 1 (mean) RMSE: ", math.sqrt(mean_squared_error(y_train, y_mean)))
    print("Baseline 2 (norm_distr) RMSE: ", math.sqrt(mean_squared_error(y_train, y_norm_distr)))
    print("Baseline 3 (unif_distr) RMSE: ", math.sqrt(mean_squared_error(y_train, y_unif_distr)))

    if params.get('tuning') == 1:
        tuning(model=args.model_name, data=X_train, labels=y_train, alpha_start=params.get('alpha_start_value'), alpha_end=params.get('alpha_end_value'), alpha_step=params.get('alpha_sweep_step'),
             l1_ratio_start=params.get('l1_ratio_start_value'), l1_ratio_end=params.get('l1_ratio_end_value'), l1_ratio_step=params.get('l1_ratio_sweep_step'))
    elif params.get('tuning') == 0:
        dumping(data=X_train, labels=y_train, path_csv=output_model_path_csv, path_joblib=output_model_path_joblib, model=args.model_name, alpha=params.get('alpha'), l1_ratio=params.get('l1_ratio'))
        