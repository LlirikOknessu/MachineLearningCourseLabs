import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from joblib import load
import math
import matplotlib.pyplot as plt
import random

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared')
    parser.add_argument('--input_model', '-im', type=str, default='data/models')
    parser.add_argument('--model_name', '-mn', type=str, default='LinearRegression')
    parser.add_argument('--params', '-p', type=str, default='params.yaml')
    return parser.parse_args()

def import_data(name: str, in_dir: Path):
    df = pd.read_csv(in_dir/(name+'.csv'))
    return df

if __name__ == '__main__':
    random.seed(42)
    args = parser_args()
    in_dir = Path(args.input_dir)
    in_model = Path(args.input_model)

    X_val = import_data("X_val", in_dir)
    y_val = import_data("y_val", in_dir)
    y_mean = [y_val.mean()] * len(y_val)
    y_norm_distr = np.random.normal(y_val.mean(), y_val.std(), len(y_val))
    y_unif_distr = np.random.uniform(y_val.min(), y_val.max(), len(y_val))
    print("Baseline 1 (mean) RMSE: ", math.sqrt(mean_squared_error(y_val, y_mean)))
    print("Baseline 2 (norm_distr) RMSE: ", math.sqrt(mean_squared_error(y_val, y_norm_distr)))
    print("Baseline 3 (unif_distr) RMSE: ", math.sqrt(mean_squared_error(y_val, y_unif_distr)))

    for model in in_model.glob('*.joblib'):
        reg = load(model)
        print(model)
        print("Model score: ", reg.score(X_val, y_val))
        predicted_value = reg.predict(X_val)    
        print("Model RMSE: ", math.sqrt(mean_squared_error(y_val, predicted_value)))
        plt.scatter(y_val, predicted_value)
        plt.xlim(0, 15)
        plt.ylim(0, 15)
        plt.grid()
        plt.xlabel('Actual value')
        plt.ylabel('Predicted value')
        plt.show()