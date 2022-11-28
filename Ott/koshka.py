import pandas as pd
import argparse
from pathlib import Path
from joblib import dump, load
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error as mae

MAPPER = {'CatBoostRegressor': CatBoostRegressor}

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--baseline_model', '-bm', type=str, default='data/models/ExtraTree.joblib',
                        required=False, help='path to linear regression prod version')
    parser.add_argument('--model_name', '-mn', type=str, default='CatBoostRegressor', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args_for_sac()

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

    model = MAPPER.get(args.model_name)()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    baseline_model_derevo = load(baseline_model_path)
    baseline_preds = baseline_model_derevo.predict(X_test)

    print('Model name is: ', args.model_name)
    print('Baseline is ExtraTree')
    print('Baseline MAE: ', mae(y_test, baseline_preds))
    print('Model MAE:', mae(y_test, preds))

    dump(model, output_model_joblib_path)
