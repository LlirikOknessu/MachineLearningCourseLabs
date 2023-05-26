import pandas as pd
from pathlib import Path
import argparse
import yaml
from sklearn.model_selection import train_test_split


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop("avgAnnCount", axis=1, inplace=True)
    df.drop("avgDeathsPerYear", axis=1, inplace=True)
    df.drop("popEst2015", axis=1, inplace=True)
    df.drop("studyPerCap", axis=1, inplace=True)
    df.drop("binnedInc", axis=1, inplace=True)
    df.drop("MedianAge", axis=1, inplace=True)
    df.drop("MedianAgeMale", axis=1, inplace=True)
    df.drop("MedianAgeFemale", axis=1, inplace=True)
    df.drop("Geography", axis=1, inplace=True)
    df.drop("AvgHouseholdSize", axis=1, inplace=True)
    df.drop("PercentMarried", axis=1, inplace=True)
    df.drop("PctNoHS18_24", axis=1, inplace=True)
    df.drop("PctHS18_24", axis=1, inplace=True)
    df.drop("PctSomeCol18_24", axis=1, inplace=True)
    df.drop("PctBachDeg18_24", axis=1, inplace=True)
    df.drop("PctHS25_Over", axis=1, inplace=True)
    df.drop("PctBachDeg25_Over", axis=1, inplace=True)
    df.drop("PctPrivateCoverageAlone", axis=1, inplace=True)
    df.drop("PctUnemployed16_Over", axis=1, inplace=True)
    df.drop("PctEmpPrivCoverage", axis=1, inplace=True)
    df.drop("PctPublicCoverage", axis=1, inplace=True)
    df.drop("PctWhite", axis=1, inplace=True)
    df.drop("PctBlack", axis=1, inplace=True)
    df.drop("PctAsian", axis=1, inplace=True)
    df.drop("PctOtherRace", axis=1, inplace=True)
    df.drop("PctMarriedHouseholds", axis=1, inplace=True)
    df.drop("BirthRate", axis=1, inplace=True)
    df = df.dropna()
    return df

if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_prep']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    for data_file in input_dir.glob('*.csv'):
        full_data = pd.read_csv(data_file)
        cleaned_data = clean_data(df=full_data)
        X, y = cleaned_data.drop("TARGET_deathRate", axis=1), cleaned_data['TARGET_deathRate']
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=params.get('train_test_ratio'),
                                                            random_state=params.get('random_state'))
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          train_size=params.get('train_val_ratio'),
                                                          random_state=params.get('random_state'))
        X_full_name = output_dir / 'X_full.csv'
        y_full_name = output_dir / 'y_full.csv'
        X_train_name = output_dir / 'X_train.csv'
        y_train_name = output_dir / 'y_train.csv'
        X_test_name = output_dir / 'X_test.csv'
        y_test_name = output_dir / 'y_test.csv'
        X_val_name = output_dir / 'X_val.csv'
        y_val_name = output_dir / 'y_val.csv'

        X.to_csv(X_full_name, index=False)
        y.to_csv(y_full_name, index=False)
        X_train.to_csv(X_train_name, index=False)
        y_train.to_csv(y_train_name, index=False)
        X_test.to_csv(X_test_name, index=False)
        y_test.to_csv(y_test_name, index=False)
        X_val.to_csv(X_val_name, index=False)
        y_val.to_csv(y_val_name, index=False)