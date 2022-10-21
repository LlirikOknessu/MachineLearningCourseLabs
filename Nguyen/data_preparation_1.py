import pandas as pd
import datetime as dt
import yaml
from sklearn.model_selection import train_test_split

# Load dvc files
deps = yaml.safe_load(open('dvc.yaml'))['stages']['data_preparation']['deps']
outs = yaml.safe_load(open('dvc.yaml'))['stages']['data_preparation']['outs']
params = yaml.safe_load(open('params.yaml'))['data_preparation']

# Read from dvc files
RAW_DATA_PATH = deps[0]
OUTPUT_PATH = outs[0]
TRAIN_RATIO = params['train_ratio']
VALIDATION_TEST_RATIO = params['validation_test_ratio']
RANDOM_STATE = params['random_state']

# Import & cleaning & reformat raw data
df = pd.read_csv(RAW_DATA_PATH)

df.drop(columns=["car_name"], inplace=True)
df['cars_years_old'] = dt.date.today().year - df.year
df.drop(columns="year", inplace=True)

df.fuel_type = pd.Categorical(df.fuel_type)
df = df.assign(fuel_type_sub=df.fuel_type.cat.codes)
df.seller_type = pd.Categorical(df.seller_type)
df = df.assign(seller_type_sub=df.seller_type.cat.codes)
df.transmission = pd.Categorical(df.transmission)
df = df.assign(transmission_sub=df.transmission.cat.codes)
df.owner = pd.Categorical(df.owner)
df = df.assign(owner_sub=df.owner.cat.codes)

df.drop(columns=["fuel_type"], inplace=True)
df.drop(columns=["seller_type"], inplace=True)
df.drop(columns=["transmission"], inplace=True)
df.drop(columns=["owner"], inplace=True)

# Split dependence and independence variables from dataframe
y = df.iloc[:, 0]
x = df.iloc[:, 1:]

# Train, validation, test split for X,Y
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=VALIDATION_TEST_RATIO,
                                                random_state=RANDOM_STATE)


# Export prepared data
x.to_csv(r''+OUTPUT_PATH + 'x_full.csv', index=False, header=True)
y.to_csv(r''+OUTPUT_PATH + 'y_full.csv', index=False, header=True)

x_train.to_csv(r''+OUTPUT_PATH + 'x_train.csv', index=False, header=True)
y_train.to_csv(r''+OUTPUT_PATH + 'y_train.csv', index=False, header=True)

x_val.to_csv(r''+OUTPUT_PATH + 'x_val.csv', index=False, header=True)
y_val.to_csv(r''+OUTPUT_PATH + 'y_val.csv', index=False, header=True)

x_test.to_csv(r''+OUTPUT_PATH + 'x_test.csv', index=False, header=True)
y_test.to_csv(r''+OUTPUT_PATH + 'y_test.csv', index=False, header=True)

# print(yaml.safe_load(open('dvc.yaml'))['stages']['data_preparation']['params'])
