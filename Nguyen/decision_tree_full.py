import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import math
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

TREES_MODELS_MAPPER = {'DecisionTree': tree.DecisionTreeRegressor,
                       'RandomForest': RandomForestRegressor,
                       'ExtraTree': ExtraTreesRegressor}

# Set the best parameters that you get on training stage for all used models
TREES_MODELS_BEST_PARAMETERS = {
    'DecisionTree': {'max_depth': 7, 'min_samples_leaf': 4, 'min_samples_split': 2, 'splitter': 'best'},
    'RandomForest': {'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 15},
    'ExtraTree': {'max_depth': 7, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 15}}


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
    y_train_cols = y_train.columns

    best_params = TREES_MODELS_BEST_PARAMETERS.get(args.model_name)
    reg = TREES_MODELS_MAPPER.get(args.model_name)(**best_params)
    if isinstance(reg, RandomForestRegressor) or isinstance(reg, ExtraTreesRegressor):
            y_train = np.ravel(y_train.values)
    reg = reg.fit(X_train, y_train)

    if isinstance(reg, tree.DecisionTreeRegressor):
        fig = plt.figure(figsize=(150,50), dpi=260)
        _ = tree.plot_tree(reg,
                           feature_names=X_train.columns,
                           class_names=y_train_cols,
                           filled=True)
        fig.savefig(output_model_path)
        #Decision tree structure can also be visualized with *.dot file in case of having difficulties with *.jpeg picture
        export_graphviz(reg, out_file=args.output_dir+'tree.dot', feature_names=X_train.columns)

    dump(reg, output_model_joblib_path)