import argparse
import implicit
import pandas as pd
import scipy.sparse as sparse

from data import MAIN_FOLDER
from src.common.util import DatasetColumnName, EvaluationParams
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
                        default=10)

    parser.add_argument('-l', '--train_loss',
                        help='If enables training loss will be logged. Default is True.',
                        action="store_false")

    parser.add_argument('-a', '--alpha',
                        help='Coefficient that controls the rate of confidence increase. Default is 40. ',
                        type=int,
                        default=40)

    parser.add_argument('-e', '--eval_mode',
                        help='If true, model will be evaluated on val dataset and results will be printed. Otherwise'
                              ' will return trained model. Default is True. ',
                        action="store_false")


def __get_data_encoded(data_path: str) -> Tuple[dict, dict]:
    df = pd.read_csv(data_path)
    users, movies = df[DatasetColumnName.USER_ID.value], df[DatasetColumnName.MOVIE_ID.value]

    # dict with keys: original movie / user id and values: relevant movie/user id from range 0...len(unique()))
    movie_filtered_ids = dict(zip(movies.unique(), range(len(movies.unique()))))
    user_filtered_ids = dict(zip(users.unique(), range(len(users.unique()))))

    return movie_filtered_ids, user_filtered_ids


def __get_sparse_matrix(movie_filtered_ids: dict, user_filtered_ids: dict, data_path: str):
    df = pd.read_csv(data_path)
    movies = df[DatasetColumnName.MOVIE_ID.value]
    users = df[DatasetColumnName.USER_ID.value]
    rating = df[DatasetColumnName.RATING.value]

    n_movies, n_users = len(movies.unique()), len(users.unique())

    rating.where(rating < EvaluationParams.MIN_RATING.value, 0, inplace=True)
    rating.where(rating >= EvaluationParams.MIN_RATING.value, 1, inplace=True)

    movie_encoded_idx = [movie_filtered_ids[movie] for movie in movies]
    user_encoded_idx = [user_filtered_ids[user] for user in users]

    sparse_item_user = sparse.csr_matrix((rating, (movie_encoded_idx, user_encoded_idx)), shape=(n_movies, n_users))
    sparse_user_item = sparse.csr_matrix((rating, (user_encoded_idx, movie_encoded_idx)), shape=(n_users, n_movies))

    return sparse_item_user, sparse_user_item


def train(latent_dim: int, regularization: float, iterations: int, alpha: float, train_loss: bool,
          train_dataset_path: str, val_dataset_path: str, eval_mode: bool):

    movie_filtered_ids, user_filtered_ids = __get_data_encoded(train_dataset_path)

    sparse_item_user, sparse_user_item = __get_sparse_matrix(movie_filtered_ids, user_filtered_ids, train_dataset_path)

    model = implicit.als.AlternatingLeastSquares(factors=latent_dim,
                                                 regularization=regularization,
                                                 iterations=iterations,
                                                 calculate_training_loss=train_loss)

    model.fit((sparse_item_user * alpha).tocsr().astype(float))

    if eval_mode:
        average_precision, precision_k = compute_precision(model, sparse_user_item, movie_filtered_ids,
                                                           user_filtered_ids, val_dataset_path)
        print(f'MAP: {average_precision}')
        return average_precision, precision_k
    return model


def main():
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    args = parser.parse_args()
    print(f'\n Factors: {args.factors}\n Iterations: {args.iterations}\n Alpha: {args.alpha}\n '
          f'Regularization: {args.regularization}')
    train(args.factors, args.regularization, args.iterations, args.alpha, args.train_loss,
          args.train_dataset_path, args.val_dataset_path, args.eval_mode)


if __name__ == "__main__":
    main()
