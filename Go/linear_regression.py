import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from joblib import dump

MODELS_MAPPER = {'Ridge': Ridge,
                 'LinearRegression': LinearRegression}

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '.csv')
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    model = MODELS_MAPPER.get(args.model_name)().fit(X_train, y_train)

    y_baseline = np.random.uniform(y_test.min(), y_test.max(), len(y_test))

    preds = model.predict(X_test)

    print(model.score(X_test, y_test))
    print('Baseline mean: ', y_baseline.mean())
    print('Baseline MAE: ', mean_absolute_error(y_test, y_baseline))
    print('Model MAE: ', mean_absolute_error(y_test, preds))

    intercept = model.intercept_.astype(float)
    coeffs = model.coef_.astype(float)
    intercept = pd.Series(intercept, name = 'intercept')
    coeffs = pd.Series(coeffs[0], name = 'coeffs')
    print('intercept: ', intercept)
    print('list of coeffs: ', coeffs)
    columns = [x for x in range (len(coeffs))]
    out_model = pd.DataFrame([coeffs, intercept])
    out_model.to_csv(output_model_path, index=False)

    dump(model, output_model_joblib_path)



# LINEAR_MODELS_MAPPER = {'Ridge': Ridge,
#                         'LinearRegression': LinearRegression}
#
# def parser_args_for_sac():
#     parser = argparse.ArgumentParser(description='Paths parser')
#     parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
#                         required=False, help='path to input data directory')
#     parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
#                         required=False, help='path to save prepared data')
#     parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
#                         help='file with dvc stage params')
#     #parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
#                         #help='file with dvc stage params')
#     return parser.parse_args()
#
#
# if __name__ == '__main__':
#     args = parser_args_for_sac()
#     #with open(args.params, 'r') as f:
#         #params_all = yaml.safe_load(f)
#     #params = params_all['linear_regression']
#
#     input_dir = Path(args.input_dir)
#     output_dir = Path(args.output_dir)
#     #model_name = params.get('model_name')
#
#     output_dir.mkdir(exist_ok=True, parents=True)
#     output_model_path = output_dir / (model_name + '.csv')
#     output_model_joblib_path = output_dir / (model_name + '.joblib')
#
#     X_train_name = input_dir / 'X_train.csv'
#     y_train_name = input_dir / 'y_train.csv'
#
#     X_train = pd.read_csv(X_train_name)
#     y_train = pd.read_csv(y_train_name)
#
#     reg = LINEAR_MODELS_MAPPER.get(model_name)().fit(X_train, y_train)
#
#     y_mean = y_train.mean()
#     y_std = y_train.std()
#     y_pred_baseline = np.random.normal(y_mean, y_std, len(y_train))
#
#     predicted_values = np.squeeze(reg.predict(X_train))
#
#     print(reg.score(X_train, y_train))
#     print("Mean posttest value: ", y_mean)
#     print("Baseline MAE: ", mean_absolute_error(y_train, y_pred_baseline))
#     print("Model MAE: ", mean_absolute_error(y_train, predicted_values))
#
#     intercept = reg.intercept_.astype(float)
#
#
#     coefficients = reg.coef_.astype(float)
#     intercept = pd.Series(intercept, name='intercept')
#     coefficients = pd.Series(coefficients[0], name='coefficients')
#     print("intercept:", intercept)
#     print("list of coefficients:", coefficients)
#     columns = [x for x in range(len(coefficients))]
#     out_model = pd.DataFrame([coefficients, intercept])
#     out_model.to_csv(output_model_path, index=False)
#
#     dump(reg, output_model_joblib_path)

#--------------------------------------------------------------------------------------------
# def parser_args_for_sac():
#     parser = argparse.ArgumentParser(description='Paths parser')
#     parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
#                         required=False, help='path to input data directory')
#     parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
#                         required=False, help='path to save prepared data')
#     parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
#                         help='file with dvc stage params')
#     return parser.parse_args()
#
# if __name__ == '__main__':
#     args = parser_args_for_sac()
#
#     input_dir = Path(args.input_dir)
#     output_dir = Path(args.output_dir)
#
#
#     output_dir.mkdir(exist_ok=True, parents=True)
#     output_model_path = output_dir / (args.model_name + '.csv')
#     output_model_joblib_path = output_dir / (args.model_name + '.joblib')
#
#     X_train_name = input_dir / 'X_train.csv'
#     y_train_name = input_dir / 'y_train.csv'
#     X_test_name = input_dir / 'X_test.csv'
#     y_test_name = input_dir / 'y_test.csv'
#
#     X_train = pd.read_csv(X_train_name)
#     y_train = pd.read_csv(y_train_name)
#     X_test = pd.read_csv(X_test_name)
#     y_test = pd.read_csv(y_test_name)
#
#     reg = LinearRegression().fit(X_train, y_train)
#
#     y_mean = y_test.mean()
#     y_pred_baseline = [y_mean] * len(y_test)
#
#     predicted_values = np.squeeze(reg.predict(X_test))
#
#     print(reg.score(X_test, y_test))
#     print("Mean charges value: ", y_mean)
#     print("Baseline MAE: ", mean_absolute_error(y_test, y_pred_baseline))
#     print("Model MAE: ", mean_absolute_error(y_test, predicted_values))
#
#     intercept = reg.intercept_.astype(float)
#
#     coefficients = reg.coef_.astype(float)
#     intercept = pd.Series(intercept, name='intercept')
#     coefficients = pd.Series(coefficients[0], name='coefficients')
#     print("intercept:", intercept)
#     print("list of coefficients:", coefficients)
#     columns = [x for x in range(len(coefficients))]
#     out_model = pd.DataFrame([coefficients, intercept])
#     out_model.to_csv(output_model_path, index=False)
#
#     dump(reg, output_model_joblib_path)