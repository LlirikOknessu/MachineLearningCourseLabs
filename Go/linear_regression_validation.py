import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from joblib import load

MODELS_MAPPER = {'Ridge': Ridge,
                 'LinearRegression': LinearRegression}

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--input_model', '-im', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    input_model = Path(args.input_model)

    X_val_name = input_dir / 'X_val.csv'
    y_val_name = input_dir / 'y_val.csv'

    X_val = pd.read_csv(X_val_name)
    y_val = pd.read_csv(y_val_name)

    model = load(input_model)

    preds = model.predict(X_val)

    y_baseline = np.random.uniform(y_val.min(), y_val.max(), len(y_val))
    print('Model name is: ', args.model_name)
    print(model.score(X_val, y_val))
    print('Baseline mean: ', y_baseline.mean())
    print('Baseline MAE: ', mean_absolute_error(y_val, y_baseline))
    print('Model MAE: ', mean_absolute_error(y_val, preds))



#-----------------------------------------------------------------------------------------------------
# LINEAR_MODELS_MAPPER = {'Ridge': Ridge,
#                         'LinearRegression': LinearRegression}
#
# def parser_args_for_sac():
#     parser = argparse.ArgumentParser(description='Paths parser')
#     parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
#                         required=False, help='path to input data directory')
#     parser.add_argument('--input_model', '-im', type=str, default='data/models/',
#                         required=False, help='path to save prepared data')
#     parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
#                         help='file with dvc stage params')
#     return parser.parse_args()
#
# if __name__ == '__main__':
#     args = parser_args_for_sac()
#     with open(args.params, 'r') as f:
#         params_all = yaml.safe_load(f)
#     params = params_all['linear_regression']
#
#     input_dir = Path(args.input_dir)
#     input_model = Path(args.input_model)
#     model = params.get('model_name') + '.joblib'
#
#     X_val_name = input_dir / 'X_val.csv'
#     y_val_name = input_dir / 'y_val.csv'
#
#     X_val = pd.read_csv(X_val_name)
#     y_val = pd.read_csv(y_val_name)
#
#     reg = load(input_model / model)
#
#     predicted_values = np.squeeze(reg.predict(X_val))
#
#     y_mean = y_val.mean()
#     y_std = y_val.std()
#     y_pred_baseline = np.random.normal(y_mean, y_std, len(y_val))
#
#     print(reg.score(X_val, y_val))
#     print("Mean charges value: ", y_mean)
#     print("Baseline MAE: ", mean_absolute_error(y_val, y_pred_baseline))
#     print("Model MAE: ", mean_absolute_error(y_val, predicted_values))

#---------------------------------------------------------------------------------------------------------
#LINEAR_MODELS_MAPPER = {'Ridge': Ridge,
                        #'LinearRegression': LinearRegression}

# def parser_args_for_sac():
#     parser = argparse.ArgumentParser(description='Paths parser')
#     parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
#                         required=False, help='path to input data directory')
#     parser.add_argument('--input_model', '-im', type=str, default='data/models/',
#                         required=False, help='path to save prepared data')
#     parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
#                         help='file with dvc stage params')
#     return parser.parse_args()
#
# if __name__ == '__main__':
#     args = parser_args_for_sac()
#
#     input_dir = Path(args.input_dir)
#     input_model = Path(args.input_model)
#
#     X_val_name = input_dir / 'X_val.csv'
#     y_val_name = input_dir / 'y_val.csv'
#
#     X_val = pd.read_csv(X_val_name)
#     y_val = pd.read_csv(y_val_name)
#
#     reg = load(input_model)
#
#     predicted_values = np.squeeze(reg.predict(X_val))
#
#     y_mean = y_val.mean()
#     y_pred_baseline = [y_mean] * len(y_val)
#
#     print(reg.score(X_val, y_val))
#     print("Mean charges: ", y_mean)
#     print("Baseline MAE: ", mean_absolute_error(y_val, y_pred_baseline))
#     print("Model MAE: ", mean_absolute_error(y_val, predicted_values))