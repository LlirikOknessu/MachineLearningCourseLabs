import pandas as pd
import argparse
from pathlib import Path
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from joblib import dump
import matplotlib.pyplot as plt

MODELS_MAPPER = {'DecisionTree': tree.DecisionTreeRegressor,
                       'RandomForest': RandomForestRegressor,
                       'ExtraTree': ExtraTreesRegressor}


MODELS_BEST_PARAMETERS = {
    'DecisionTree': {'max_depth': 74, 'min_samples_leaf': 3, 'min_samples_split': 3, 'splitter': 'random'},
    'RandomForest': {'max_depth': 66, 'min_samples_leaf': 1, 'min_samples_split': 7, 'n_estimators': 40},
    'ExtraTree': {'max_depth': 26, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 20}}


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
    output_model_path = output_dir / (args.model_name + '_prod.jpg')
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_train_name = input_dir / 'X_full.csv'
    y_train_name = input_dir / 'y_full.csv'

    X_train = pd.read_csv(X_train_name)
    y_train = pd.read_csv(y_train_name)
    columns = y_train.columns

    best_params = MODELS_BEST_PARAMETERS.get(args.model_name)
    model = MODELS_MAPPER.get(args.model_name)(**best_params)
    if isinstance(model, RandomForestRegressor) or isinstance(model, ExtraTreesRegressor):
        y_train = np.ravel(y_train.values)
    model.fit(X_train, y_train)

    if isinstance(model, tree.DecisionTreeRegressor):
        fig = plt.figure(figsize=(60, 25))
        _ = tree.plot_tree(model,
                           feature_names=X_train.columns,
                           class_names=columns,
                           filled=True)
        fig.savefig(output_model_path)

    dump(model, output_model_joblib_path)