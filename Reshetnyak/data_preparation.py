import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
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

def reduction_bin(x):
    x = x.replace("(", "")
    x = x.replace(")", "")
    x = x.replace("]", "")
    x = x.replace("[", "")
    x = x.replace(",", "")
    return (sum(map(float, x.split())) / 2)

# Упрощение и преобразование датасета:
def clean_data(df: pd.DataFrame):
    df['binnedInc'] = df['binnedInc'].apply(reduction_bin)
    df['TARGET_deathRate'] = np.log(df['TARGET_deathRate'])
    # Удаление слабокоррелируемых параметров
    df.drop("avgDeathsPerYear", axis=1, inplace=True)
    df.drop("studyPerCap", axis=1, inplace=True)
    df.drop("MedianAge", axis=1, inplace=True)
    df.drop("MedianAgeMale", axis=1, inplace=True)
    df.drop("MedianAgeFemale", axis=1, inplace=True)
    df.drop("Geography", axis=1, inplace=True)
    df.drop("AvgHouseholdSize", axis=1, inplace=True)
    df.drop("PctNoHS18_24", axis=1, inplace=True)
    df.drop("BirthRate", axis=1, inplace=True)
    df = to_categorical(df)
    return df


# Вспомогательные функции для clean_data:
def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df.avgAnnCount = pd.Categorical(df.avgAnnCount)
    df = df.assign(avgAnnCount=df.avgAnnCount.cat.codes)
    df.incidenceRate = pd.Categorical(df.incidenceRate)
    df = df.assign(incidenceRate=df.incidenceRate.cat.codes)
    df.medIncome = pd.Categorical(df.medIncome)
    df = df.assign(medIncome=df.medIncome.cat.codes)
    df.popEst2015 = pd.Categorical(df.popEst2015)
    df = df.assign(popEst2015=df.popEst2015.cat.codes)
    df.povertyPercent = pd.Categorical(df.povertyPercent)
    df = df.assign(povertyPercent=df.povertyPercent.cat.codes)
    df.PercentMarried = pd.Categorical(df.PercentMarried)
    df = df.assign(PercentMarried=df.PercentMarried.cat.codes)
    df.PctHS18_24 = pd.Categorical(df.PctHS18_24)
    df = df.assign(PctHS18_24=df.PctHS18_24.cat.codes)
    df.PctSomeCol18_24 = pd.Categorical(df.PctSomeCol18_24)
    df = df.assign(PctSomeCol18_24=df.PctSomeCol18_24.cat.codes)
    df.PctBachDeg18_24 = pd.Categorical(df.PctBachDeg18_24)
    df = df.assign(PctBachDeg18_24=df.PctBachDeg18_24.cat.codes)
    df.PctHS25_Over = pd.Categorical(df.PctHS25_Over)
    df = df.assign(PctHS25_Over=df.PctHS25_Over.cat.codes)
    df.PctBachDeg25_Over = pd.Categorical(df.PctBachDeg25_Over)
    df = df.assign(PctBachDeg25_Over=df.PctBachDeg25_Over.cat.codes)
    df.PctEmployed16_Over = pd.Categorical(df.PctEmployed16_Over)
    df = df.assign(PctEmployed16_Over=df.PctEmployed16_Over.cat.codes)
    df.PctUnemployed16_Over = pd.Categorical(df.PctUnemployed16_Over)
    df = df.assign(PctUnemployed16_Over=df.PctUnemployed16_Over.cat.codes)
    df.PctPrivateCoverage = pd.Categorical(df.PctPrivateCoverage)
    df = df.assign(PctPrivateCoverage=df.PctPrivateCoverage.cat.codes)
    df.PctPrivateCoverageAlone = pd.Categorical(df.PctPrivateCoverageAlone)
    df = df.assign(PctPrivateCoverageAlone=df.PctPrivateCoverageAlone.cat.codes)
    df.PctEmpPrivCoverage = pd.Categorical(df.PctEmpPrivCoverage)
    df = df.assign(PctEmpPrivCoverage=df.PctEmpPrivCoverage.cat.codes)
    df.PctPublicCoverage = pd.Categorical(df.PctPublicCoverage)
    df = df.assign(PctPublicCoverage=df.PctPublicCoverage.cat.codes)
    df.PctPublicCoverageAlone = pd.Categorical(df.PctPublicCoverageAlone)
    df = df.assign(PctPublicCoverageAlone=df.PctPublicCoverageAlone.cat.codes)
    df.PctWhite = pd.Categorical(df.PctWhite)
    df = df.assign(PctWhite=df.PctWhite.cat.codes)
    df.PctBlack = pd.Categorical(df.PctBlack)
    df = df.assign(PctBlack=df.PctBlack.cat.codes)
    df.PctAsian = pd.Categorical(df.PctAsian)
    df = df.assign(PctAsian=df.PctAsian.cat.codes)
    df.PctOtherRace = pd.Categorical(df.PctOtherRace)
    df = df.assign(PctOtherRace=df.PctOtherRace.cat.codes)
    df.PctMarriedHouseholds = pd.Categorical(df.PctMarriedHouseholds)
    df = df.assign(PctMarriedHouseholds=df.PctMarriedHouseholds.cat.codes)
    return df

if __name__ == '__main__':
    args = parser_args_for_sac()
    print(args)
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']
    print(params)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    for data_file in input_dir.glob('*.csv'):
        full_data = pd.read_csv(data_file)
        cleaned_data = clean_data(df=full_data)
        X, y = cleaned_data.drop("TARGET_deathRate", axis=1), cleaned_data['TARGET_deathRate']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=params.get('train_test_ratio'),
                                                            random_state=params.get('random_state'))
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=params.get('train_val_ratio'),
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
