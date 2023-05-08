import argparse
from pathlib import Path
import pandas as pd
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

def to_categorical(df: pd.DataFrame):
    df.genre_Cars = pd.Categorical(df.genre_Cars)
    df = df.assign(genre_Cars=df.genre_Cars.cat.codes)
    df.genre_Vampire = pd.Categorical(df.genre_Vampire)
    df = df.assign(genre_Vampire=df.genre_Vampire.cat.codes)
    df.genre_Fantasy = pd.Categorical(df.genre_Fantasy)
    df = df.assign(genre_Fantasy=df.genre_Fantasy.cat.codes)
    df.genre_Seinen = pd.Categorical(df.genre_Seinen)
    df = df.assign(genre_Seinen=df.genre_Seinen.cat.codes)
    df.genre_ShoujoAi = pd.Categorical(df.genre_ShoujoAi)
    df = df.assign(genre_ShoujoAi=df.genre_ShoujoAi.cat.codes)
    df.genre_Magic = pd.Categorical(df.genre_Magic)
    df = df.assign(genre_Magic=df.genre_Magic.cat.codes)
    df.genre_Samurai = pd.Categorical(df.genre_Samurai)
    df = df.assign(genre_Samurai=df.genre_Samurai.cat.codes)
    df.genre_Music = pd.Categorical(df.genre_Music)
    df = df.assign(genre_Music=df.genre_Music.cat.codes)
    df.genre_Supernatural = pd.Categorical(df.genre_Supernatural)
    df = df.assign(genre_Supernatural=df.genre_Supernatural.cat.codes)
    df.genre_Ecchi = pd.Categorical(df.genre_Ecchi)
    df = df.assign(genre_Ecchi=df.genre_Ecchi.cat.codes)
    df.genre_School = pd.Categorical(df.genre_School)
    df = df.assign(genre_School=df.genre_School.cat.codes)
    df.genre_Thriller = pd.Categorical(df.genre_Thriller)
    df = df.assign(genre_Thriller=df.genre_Thriller.cat.codes)
    df.genre_Shoujo = pd.Categorical(df.genre_Shoujo)
    df = df.assign(genre_Shoujo=df.genre_Shoujo.cat.codes)
    df.genre_Game = pd.Categorical(df.genre_Game)
    df = df.assign(genre_Game=df.genre_Game.cat.codes)
    df.genre_Police = pd.Categorical(df.genre_Police)
    df = df.assign(genre_Police=df.genre_Police.cat.codes)
    df.genre_Yuri = pd.Categorical(df.genre_Yuri)
    df = df.assign(genre_Yuri=df.genre_Yuri.cat.codes)
    df.genre_Horror = pd.Categorical(df.genre_Horror)
    df = df.assign(genre_Horror=df.genre_Horror.cat.codes)
    df.genre_Josei = pd.Categorical(df.genre_Josei)
    df = df.assign(genre_Josei=df.genre_Josei.cat.codes)
    df.genre_Kids = pd.Categorical(df.genre_Kids)
    df = df.assign(genre_Kids=df.genre_Kids.cat.codes)
    df.genre_Dementia = pd.Categorical(df.genre_Dementia)
    df = df.assign(genre_Dementia=df.genre_Dementia.cat.codes)
    df.genre_Historical = pd.Categorical(df.genre_Historical)
    df = df.assign(genre_Historical=df.genre_Historical.cat.codes)
    df.genre_Comedy = pd.Categorical(df.genre_Comedy)
    df = df.assign(genre_Comedy=df.genre_Comedy.cat.codes)
    df.genre_Mystery = pd.Categorical(df.genre_Mystery)
    df = df.assign(genre_Mystery=df.genre_Mystery.cat.codes)
    df.genre_Demons = pd.Categorical(df.genre_Demons)
    df = df.assign(genre_Demons=df.genre_Demons.cat.codes)
    df.genre_Yaoi = pd.Categorical(df.genre_Yaoi)
    df = df.assign(genre_Yaoi=df.genre_Yaoi.cat.codes)
    df.genre_Space = pd.Categorical(df.genre_Space)
    df = df.assign(genre_Space=df.genre_Space.cat.codes)
    df.genre_Parody = pd.Categorical(df.genre_Parody)
    df = df.assign(genre_Parody=df.genre_Parody.cat.codes)
    df.genre_Hentai = pd.Categorical(df.genre_Hentai)
    df = df.assign(genre_Hentai=df.genre_Hentai.cat.codes)
    df.genre_Shounen = pd.Categorical(df.genre_Shounen)
    df = df.assign(genre_Shounen=df.genre_Shounen.cat.codes)
    df.genre_Mecha = pd.Categorical(df.genre_Mecha)
    df = df.assign(genre_Mecha=df.genre_Mecha.cat.codes)
    df.genre_MartialArts = pd.Categorical(df.genre_MartialArts)
    df = df.assign(genre_MartialArts=df.genre_MartialArts.cat.codes)
    df.genre_Action = pd.Categorical(df.genre_Action)
    df = df.assign(genre_Action=df.genre_Action.cat.codes)
    df.genre_Military = pd.Categorical(df.genre_Military)
    df = df.assign(genre_Military=df.genre_Military.cat.codes)
    df.genre_Drama = pd.Categorical(df.genre_Drama)
    df = df.assign(genre_Drama=df.genre_Drama.cat.codes)
    df.genre_Psychological = pd.Categorical(df.genre_Psychological)
    df = df.assign(genre_Psychological=df.genre_Psychological.cat.codes)
    df.genre_Romance = pd.Categorical(df.genre_Romance)
    df = df.assign(genre_Romance=df.genre_Romance.cat.codes)
    df.genre_ShounenAi = pd.Categorical(df.genre_ShounenAi)
    df = df.assign(genre_ShounenAi=df.genre_ShounenAi.cat.codes)
    df.genre_Sports = pd.Categorical(df.genre_Sports)
    df = df.assign(genre_Sports=df.genre_Sports.cat.codes)
    df.genre_SliceofLife = pd.Categorical(df.genre_SliceofLife)
    df = df.assign(genre_SliceofLife=df.genre_SliceofLife.cat.codes)
    df.genre_SciFi = pd.Categorical(df.genre_SciFi)
    df = df.assign(genre_SciFi=df.genre_SciFi.cat.codes)
    df.genre_Adventure = pd.Categorical(df.genre_Adventure)
    df = df.assign(genre_Adventure=df.genre_Adventure.cat.codes)
    df.genre_Harem = pd.Categorical(df.genre_Harem)
    df = df.assign(genre_Harem=df.genre_Harem.cat.codes)
    df.genre_SuperPower = pd.Categorical(df.genre_SuperPower)
    df = df.assign(genre_SuperPower=df.genre_SuperPower.cat.codes)
    df.type = pd.Categorical(df.type)
    df = df.assign(type=df.type.cat.codes)
    return df

def add_genres(df: pd.DataFrame):
    genrestring = ''
    for i in df.genre:
        genrestring += i.replace(' ', '') + ','
    genrestring=genrestring.replace('-','')
    genres = list(set(genrestring.split(',')))[1:]
    for anime_genre in genres:
        encoding = [anime_genre in i for i in df.genre]
        df["genre_" + anime_genre] = encoding
        print(anime_genre)
    df.drop(['genre'], axis=1, inplace=True)
    print(df)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print(df.all())
    df.drop("name", axis=1, inplace=True)
    df.drop("anime_id", axis=1, inplace=True)
    df.dropna(inplace=True)

    df = add_genres(df)
    df = to_categorical(df)

    return df

if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    for data_file in input_dir.glob('anime.csv'):
        full_data = pd.read_csv(data_file)
        full_data = full_data[full_data.episodes != "Unknown"]
        cleaned_data = clean_data(df=full_data)

        X, y = cleaned_data.drop("rating", axis=1), cleaned_data['rating']
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