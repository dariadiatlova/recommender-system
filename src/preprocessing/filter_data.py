import pandas as pd
import numpy as np
from data import MAIN_FOLDER
from src.common.util import DatasetColumnName


def filter_data():
    """ Remove movies from the dataset that were rated by less than 1% of users """
    df_rating = pd.read_csv(MAIN_FOLDER.parent / 'rating.csv')
    df_tag = pd.read_csv(MAIN_FOLDER.parent / 'tag.csv')

    n_users = int(len(df_rating[DatasetColumnName.USER_ID.value].unique()) * 0.01)
    print(f'Initial dataset size: {df_rating.shape[0]} ratings')

    grouped = df_rating.groupby([DatasetColumnName.MOVIE_ID.value])
    n_movies = grouped.size()
    index_names = n_movies[n_movies > n_users].index

    filtered_df_rating = df_rating[df_rating[DatasetColumnName.MOVIE_ID.value].isin(index_names)]
    filtered_df_tag = df_tag[df_tag[DatasetColumnName.MOVIE_ID.value].isin(index_names)]

    print(f'Filtered dataset size: {filtered_df_rating.shape[0]} ratings')
    print(f'Reduced dataset size on {np.round((df_rating.shape[0] - filtered_df_rating.shape[0]) / df_rating.shape[0], 2) * 100}%')

    filtered_df_rating.to_csv(MAIN_FOLDER.parent / 'filtered_rating.csv', index=False)
    filtered_df_tag.to_csv(MAIN_FOLDER.parent / 'filtered_tag.csv', index=False)


if __name__ == "__main__":
    filter_data()
