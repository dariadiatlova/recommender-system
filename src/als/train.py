import argparse
import implicit
import pandas as pd
import numpy as np
import scipy.sparse as sparse

from data import MAIN_FOLDER
from src.common.util import DatasetColumnName, EvaluationParams, plot
from src.common.custom_precision import compute_precision
from typing import Tuple


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-tp', '--train_dataset_path',
                        help='Path to training dataset. Default is path to train_rating.csv file in data folder." ',
                        type=int,
                        default=MAIN_FOLDER.parent / 'train_rating.csv')

    parser.add_argument('-vp', '--val_dataset_path',
                        help='Path to val dataset. Default is path to val_rating.csv file in data folder.',
                        type=int,
                        default=MAIN_FOLDER.parent / 'val_rating.csv')

    parser.add_argument('-f', '--factors',
                        help='The number of latent vectors. Default is 20. ',
                        type=int,
                        default=20)

    parser.add_argument('-r', '--regularization',
                        help='The regularization parameter. Default is 0.1. ',
                        type=float,
                        default=0.1)

    parser.add_argument('-i', '--iterations',
                        help='The number of iterations. Default is 15. ',
                        type=int,
                        default=30)

    parser.add_argument('-l', '--train_loss',
                        help='If enables training loss will be logged. Default is True.',
                        action="store_false")

    parser.add_argument('-a', '--alpha',
                        help='Coefficient that controls the rate of confidence increase. Default is 40. ',
                        type=int,
                        default=40)

    parser.add_argument('-e', '--eval_mode',
                        help='If true, model will be evaluated on val dataset and results will be printed. Otherwise'
                              ' will return trained model and loss. Default is True. ',
                        action="store_false")


class ALS:

    def __init__(self):
        self.sparse_item_user = None
        self.sparse_user_item = None
        self.predictions_encoded = None
        self.irrelevant_movies = None
        self.relevant_users = None
        self.predictions = None

    @staticmethod
    def read_dataset(data_path: str) -> Tuple[pd.Series, pd.Series, pd.Series]:
        df = pd.read_csv(data_path)
        movies = df[DatasetColumnName.MOVIE_ID.value]
        users = df[DatasetColumnName.USER_ID.value]
        # rating = df[DatasetColumnName.RATING.value]
        rating = df.iloc[:, 2]
        return movies, users, rating

    def _get_data_encoded(self, data_path: str) -> None:
        df = pd.read_csv(data_path)
        users, movies = df[DatasetColumnName.USER_ID.value], df[DatasetColumnName.MOVIE_ID.value]
        # dict with keys: original movie / user id and values: relevant movie/user id from range 0...len(unique()))
        self.movie_filtered_ids = dict(zip(movies.unique(), range(len(movies.unique()))))
        self.user_filtered_ids = dict(zip(users.unique(), range(len(users.unique()))))

    def _get_sparse_matrix(self, data_path: str) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
        movies, users, rating = self.read_dataset(data_path)
        n_movies, n_users = len(movies.unique()), len(users.unique())

        rating.where(rating < EvaluationParams.MIN_RATING.value, 0, inplace=True)
        rating.where(rating >= EvaluationParams.MIN_RATING.value, 1, inplace=True)

        movie_encoded_idx = [self.movie_filtered_ids[movie] for movie in movies]
        user_encoded_idx = [self.user_filtered_ids[user] for user in users]

        self.sparse_item_user = sparse.csr_matrix((rating, (movie_encoded_idx, user_encoded_idx)),
                                                  shape=(n_movies, n_users))
        self.sparse_user_item = sparse.csr_matrix((rating, (user_encoded_idx, movie_encoded_idx)),
                                                  shape=(n_users, n_movies))
        return self.sparse_item_user, self.sparse_user_item

    def train(self, latent_dim: int, regularization: float, iterations: int, alpha: float, train_loss: bool,
              train_dataset_path: str):

        self._get_data_encoded(train_dataset_path)

        self.sparse_item_user, self.sparse_user_item = self._get_sparse_matrix(train_dataset_path)

        model = implicit.als.AlternatingLeastSquares(factors=latent_dim,
                                                     regularization=regularization,
                                                     iterations=iterations,
                                                     calculate_training_loss=train_loss,
                                                     random_state=EvaluationParams.SEED.value)

        # alpha * r_ui
        matrix = (self.sparse_item_user * alpha).tocsr().astype(float)
        # c_ui = 1 + alpha * r_ui
        matrix.data += 1.
        loss = model.fit(matrix)
        return loss, model

    def filter_users_for_validation(self, validation_dataset_path: str) -> None:
        movies, users, rating = self.read_dataset(validation_dataset_path)

        unique_movies = movies.unique()
        unique_users = users.unique()

        unique_movie_keys = self.movie_filtered_ids.keys()
        unique_user_keys = self.user_filtered_ids.keys()

        self.irrelevant_movies = [movie for movie in unique_movie_keys if self.movie_filtered_ids[movie] not in unique_movies]
        self.relevant_users = [self.user_filtered_ids[user] for user in unique_users if user in unique_user_keys]

    def get_encoded_predictions(self, model, val_dataset_path) -> None:
        self.filter_users_for_validation(val_dataset_path)
        self.predictions = []
        for user in self.relevant_users:
            top_k_movies = self.predict(model, user)
            self.predictions.append(top_k_movies)

    def predict(self, model, user: int) -> np.ndarray:
        top_k_movies = model.recommend(user,
                                       self.sparse_user_item.tocsr().astype(float),
                                       N=EvaluationParams.K.value,
                                       filter_already_liked_items=False,
                                       filter_items=self.irrelevant_movies)
        return top_k_movies

    def get_metric(self, validation_dataset_path: str):
        precision = compute_precision(self.predictions, validation_dataset_path, self.movie_filtered_ids,
                                      self.user_filtered_ids, self.relevant_users)
        return precision


def main():
    als = ALS()
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    args = parser.parse_args()
    print(f'\n Factors: {args.factors}\n Iterations: {args.iterations}\n Alpha: {args.alpha}\n '
          f'Regularization: {args.regularization}')

    loss, model = als.train(args.factors,
                            args.regularization,
                            args.iterations,
                            args.alpha,
                            args.train_loss,
                            args.train_dataset_path)

    plot(loss, args.iterations)

    if args.eval_mode:
        als.get_encoded_predictions(model, args.val_dataset_path)
        precision = als.get_metric(args.val_dataset_path)
        print(f'Precision@{EvaluationParams.K.value}\n{precision}')


if __name__ == "__main__":
    main()
