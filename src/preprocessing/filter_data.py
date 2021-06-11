import pandas as pd
import numpy as np
from data import MAIN_FOLDER
from src.common.util import DatasetColumnName


def filter_data():
    """ Remove movies from the dataset that were rated by less than 1% of users """
    df = pd.read_csv(MAIN_FOLDER.parent / 'rating.csv')
    n_users = int(len(df[DatasetColumnName.USER_ID.value].unique()) * 0.01)
    print(f'Initial dataset size: {df.shape[0]} ratings')
    grouped = df.groupby([DatasetColumnName.MOVIE_ID.value])
    n_movies = grouped.size()
    index_names = n_movies[n_movies > n_users].index
    filtered_df = df[df[DatasetColumnName.MOVIE_ID.value].isin(index_names)]
    print(f'Filtered dataset size: {filtered_df.shape[0]} ratings')
    print(f'Reduced dataset size on {np.round((df.shape[0] - filtered_df.shape[0]) / df.shape[0], 2) * 100}%')
    filtered_df.to_csv(MAIN_FOLDER.parent / 'filtered_rating.csv', index=False)


if __name__ == "__main__":
    filter_data()
