import pandas as pd
import argparse
from pathlib import Path
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from joblib import dump

LINEAR_MODELS_MAPPER = {'Ridge': Ridge,
                        'LinearRegression': LinearRegression}


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/prepared/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/models/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    parser.add_argument('--model_name', '-mn', type=str, default='LinearRegression', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['linear_regression']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_name = params.get('model_name')

    output_dir.mkdir(exist_ok=True, parents=True)
    output_model_path = output_dir / (model_name + '_prod.csv')
    output_model_joblib_path = output_dir / (model_name + '_prod.joblib')

    X_full_name = input_dir / 'X_full.csv'
    y_full_name = input_dir / 'y_full.csv'

    X_full = pd.read_csv(X_full_name)
    y_full = pd.read_csv(y_full_name)

    reg = LINEAR_MODELS_MAPPER.get(model_name)().fit(X_full, y_full)

    print(reg.score(X_full, y_full))
    intercept = reg.intercept_.astype(float)
    coefficients = reg.coef_.astype(float)
    intercept = pd.Series(intercept, name='intercept')
    coefficients = pd.Series(coefficients[0], name='coefficients')
    print("intercept:", intercept)
    print("list of coefficients:", coefficients)
    columns = [x for x in range(len(coefficients))]
    out_model = pd.DataFrame([coefficients, intercept])
    out_model.to_csv(output_model_path, index=False)
    dump(reg, output_model_joblib_path)