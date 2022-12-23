import pandas as pd
import argparse
from pathlib import Path
from sklearn import preprocessing
import yaml
import numpy as np
from sklearn.preprocessing import MinMaxScaler
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

def add_region(df: pd.DataFrame):
    Africa = (
        'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cameroon', 'Cape Verde',
        'Central African Republic',
        'Chad', 'Comoros', 'Congo', 'Democratic Republic of the Congo', 'Djibouti', 'Egypt', 'Equatorial Guinea',
        'Eritrea',
        'Ethiopia', 'Gabon', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia',
        'Libya', 'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia', 'Niger',
        'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles', 'Sierra Leone', 'Somalia',
        'South Africa',
        'South Sudan', 'Sudan', 'Swaziland', 'United Republic of Tanzania', 'Togo', 'Tunisia', 'Uganda', 'Zambia',
        'Zimbabwe', 'Cabo Verde', 'Côte d\'Ivoire')
    Asia = (
        'Afghanistan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei Darussalam', 'Myanmar', 'Cambodia', 'China',
        'Timor-Leste',
        'India', 'Indonesia', 'Iran (Islamic Republic of)', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan',
        'Democratic People\'s Republic of Korea', 'Republic of Korea',
        'Kuwait', 'Kyrgyzstan', 'Lao People\'s Democratic Republic', 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia',
        'Nepal', 'Oman', 'Pakistan',
        'Philippines', 'Qatar', 'Russian Federation', 'Saudi Arabia', 'Singapore', 'Sri Lanka', 'Syrian Arab Republic',
        'Tajikistan',
        'Thailand', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Viet Nam', 'Yemen')
    Europe = (
        'Albania', 'Andorra', 'Armenia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium', 'Bosnia and Herzegovina',
        'Bulgaria',
        'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Georgia', 'Germany', 'Greece',
        'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Macedonia',
        'Malta',
        'Republic of Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania',
        'San Marino', 'Serbia',
        'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Ukraine',
        'United Kingdom of Great Britain and Northern Ireland', 'Vatican City',
        'The former Yugoslav republic of Macedonia')
    North_America = ('Antigua and Barbuda', 'Bahamas', 'Barbados', 'Belize', 'Canada', 'Costa Rica', 'Cuba', 'Dominica',
                     'Dominican Republic', 'El Salvador', 'Grenada', 'Guatemala', 'Haiti', 'Honduras', 'Jamaica',
                     'Mexico',
                     'Nicaragua', 'Panama', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines',
                     'Trinidad and Tobago', 'United States of America')
    South_America = (
        'Argentina', 'Bolivia (Plurinational State of)', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay',
        'Peru', 'Suriname', 'Uruguay',
        'Venezuela (Bolivarian Republic of)')
    Australia_Oceania = (
        'Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia (Federated States of)', 'Nauru', 'New Zealand',
        'Palau', 'Papua New Guinea',
        'Samoa', 'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu')
    country = df['Country'].unique()
    for i in range(len(Africa)):
        df.loc[df['Country'] == Africa[i], 'Region'] = 'Africa'
    for i in range(len(Asia)):
        df.loc[df['Country'] == Asia[i], 'Region'] = 'Asia'
    for i in range(len(Europe)):
        df.loc[df['Country'] == Europe[i], 'Region'] = 'Europe'
    for i in range(len(North_America)):
        df.loc[df['Country'] == North_America[i], 'Region'] = 'North_America'
    for i in range(len(South_America)):
        df.loc[df['Country'] == South_America[i], 'Region'] = 'South_America'
    for i in range(len(Australia_Oceania)):
        df.loc[df['Country'] == Australia_Oceania[i], 'Region'] = 'Australia_Oceania'
    return df
def to_categorical(df: pd.DataFrame):
    df.Status = pd.Categorical(df.Status)
    df = df.assign(Status=df.Status.cat.codes)
    df.Region = pd.Categorical(df.Region)
    df = df.assign(Region=df.Region.cat.codes)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.map(lambda x: x.replace("-", "_").replace(" ", "_"))
    df.drop("infant_deaths", axis=1, inplace=True)
    df.drop("Alcohol", axis=1, inplace=True)
    df.drop("percentage_expenditure", axis=1, inplace=True)
    df.drop("_HIV/AIDS", axis=1, inplace=True)
    df.drop("Hepatitis_B", axis=1, inplace=True)
    df.drop("Measles_", axis=1, inplace=True)
    df.drop("_BMI_", axis=1, inplace=True)
    df.drop("under_five_deaths_", axis=1, inplace=True)
    df.drop("Total_expenditure", axis=1, inplace=True)
    df.drop("GDP", axis=1, inplace=True)
    df.drop("Population", axis=1, inplace=True)
    df.drop("_thinness__1_19_years", axis=1, inplace=True)
    df.drop("_thinness_5_9_years", axis=1, inplace=True)
    df.drop("Country", axis=1, inplace=True)
    df = df[df['Schooling'].notna()]
    df = df[df['Polio'].notna()]
    df = df[df['Diphtheria_'].notna()]
    df = df[df['Income_composition_of_resources'].notna()]
    df = df[df['Year'].notna()]
    df = df[df['Status'].notna()]
    df = df[df['Adult_Mortality'].notna()]
    df = df[df['Region'].notna()]
    df = to_categorical(df)

    scaler = MinMaxScaler()
    year = df['Year'].values.reshape(-1, 1)
    year_scaled = scaler.fit_transform(year)
    df['Year'] = year_scaled
    #adultMortality = df['Adult_Mortality'].values.reshape(-1, 1)
    #adultMortality_scaled = scaler.fit_transform(adultMortality)
    #df['Adult_Mortality'] = adultMortality_scaled
    life = df['Life_expectancy_'].values.reshape(-1, 1)
    life_scaled = scaler.fit_transform(life)
    df['Life_expectancy_'] = life_scaled

    return df


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    for data_file in input_dir.glob('*.csv'):
        full_data = pd.read_csv(data_file)
        add_region = add_region(df=full_data)
        cleaned_data = clean_data(df=full_data)
        X, y = cleaned_data.drop("Life_expectancy_", axis=1), cleaned_data['Life_expectancy_']
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