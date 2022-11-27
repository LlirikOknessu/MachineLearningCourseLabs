import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.metrics import mean_absolute_error
from joblib import dump, load
from catboost import Pool, CatBoostRegressor


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/DecisionTree_prod.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--model_name', '-mn', type=str, default='CatBoosting', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['CatBoosting']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (args.model_name + '_prod.jpg')
    output_model_joblib_path = output_dir / (args.model_name + '_prod.joblib')

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'

    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    full_pool = Pool(X_full, y_full)

    model = CatBoostRegressor(iterations=params.get('iterations'),
                              depth=params.get('depth'),
                              learning_rate=params.get('learning_rate'),
                              loss_function=params.get('loss_function'))
    model.fit(full_pool)

    predicted_values = model.predict(full_pool)

    print(model.score(full_pool))

    print("Model MAE: ", mean_absolute_error(y_full, predicted_values))

    dump(model, output_model_joblib_path)