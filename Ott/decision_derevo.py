import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error as mae
from joblib import dump, load
from sklearn.model_selection import GridSearchCV

MODELS_MAPPER = {'DecisionTree': tree.DecisionTreeRegressor,
                 'RandomForest': RandomForestRegressor,
                 'ExtraTree': ExtraTreesRegressor}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/LinearRegression_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--model_name', '-mn', type=str, default='LR', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ =='__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['decision_tree']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    baseline_model_path = Path(args.baseline_model)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_joblib_path = output_dir / (args.model_name + '.joblib')

    X_train_name = input_dir / 'X_train.csv'
    y_train_name = input_dir / 'y_train.csv'
    X_test_name = input_dir / 'X_test.csv'
    y_test_name = input_dir / 'y_test.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    X_test = pd.read_csv(X_test_name)
    y_test = pd.read_csv(y_test_name)

    tree_model = MODELS_MAPPER.get(args.model_name)()
    tree_gridsearch = GridSearchCV(tree_model, params[args.model_name], refit=True, n_jobs=-1)
    if isinstance(tree_model, RandomForestRegressor) or isinstance(tree_model, ExtraTreesRegressor):
        y_train = np.ravel(y_train.values)
        y_test = np.ravel(y_test.values)

    tree_gridsearch.fit(X_train, y_train)

    baseline_model_linear = load(baseline_model_path)
    y_baseline_preds = baseline_model_linear.predict(X_test)

    preds = tree_gridsearch.best_estimator_.predict(X_test)

    print('Model name is: ', args.model_name)
    print("GridSearch scorring: ", tree_gridsearch.score(X_test, y_test))
    print("Best params: {}".format(tree_gridsearch.best_params_))

    print('Baseline MAE: ', mae(y_test, y_baseline_preds))
    print('Model MAE: ', mae(y_test, preds))

    dump(tree_gridsearch.best_estimator_, output_model_joblib_path)