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

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/LinearRegression.joblib')
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

    X_test = import_data("X_test", in_dir)
    y_test = import_data("y_test", in_dir)

    reg = load(in_model)

    y_mean = [y_test.mean()] * len(y_test)
    y_norm_distr = np.random.normal(y_test.mean(), y_test.std(), len(y_test))
    y_unif_distr = np.random.uniform(y_test.min(), y_test.max(), len(y_test))

    predicted_value = reg.predict(X_test)
    
    print(args.model_name)
    print("Model score: ", reg.score(X_test, y_test))
    print("Baseline 1 (mean) RMSE: ", math.sqrt(mean_squared_error(y_test, y_mean)))
    print("Baseline 2 (norm_distr) RMSE: ", math.sqrt(mean_squared_error(y_test, y_norm_distr)))
    print("Baseline 3 (unif_distr) RMSE: ", math.sqrt(mean_squared_error(y_test, y_unif_distr)))
    print("Model RMSE: ", math.sqrt(mean_squared_error(y_test, predicted_value)))
    plt.scatter(y_test, predicted_value)
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.grid()
    plt.xlabel('Actual value')
    plt.ylabel('Predicted value')
    plt.show()

