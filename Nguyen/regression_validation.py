import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from joblib import load
import math
import matplotlib.pyplot as plt

LINEAR_MODELS_MAPPER = {'LinearRegression': LinearRegression,
                        'Ridge': Ridge,
                        'Lasso': Lasso,
                        'ElasticNet': ElasticNet}

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
    args = parser_args()
    in_dir = Path(args.input_dir)
    in_model = Path(args.input_model)

    x_val = import_data("x_val", in_dir)
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
        print(reg.coef_.astype(float))
        print(reg.intercept_.astype(float))
        print("Model score: ", reg.score(x_val, y_val))
        predicted_value = reg.predict(x_val)    
        print("Model RMSE: ", math.sqrt(mean_squared_error(y_val, predicted_value)))
        plt.scatter(y_val, predicted_value)
        plt.xlim(0, 6000000)
        plt.ylim(0, 6000000)
        plt.grid()
        plt.xlabel('Actual value')
        plt.ylabel('Predicted value')
        plt.show()

