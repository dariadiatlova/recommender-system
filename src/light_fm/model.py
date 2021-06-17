import argparse
import numpy as np
import pandas as pd

from data import MAIN_FOLDER
from lightfm import LightFM
from typing import List, Tuple

from src.common.custom_precision import compute_precision
from src.common.util import EvaluationParams, DatasetColumnName
from src.light_fm.build_dataset import make_dataset


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-ls', '--latent_size',
                        help='The dimensionality of the feature latent embeddings. Default is 10. ',
                        type=int,
                        default=20)

    parser.add_argument('-lr', '--learning_rate',
                        help='The dimensionality of the feature latent embeddings. Default is 0.01.',
                        type=float,
                        default=0.03)

    parser.add_argument('-ia', '--item_alpha',
                        help='L2 penalty on item features. Default is 1.',
                        type=float,
                        default=0)

    parser.add_argument('-e', '--epochs',
                        help='Number of epochs for training. Default is 1.',
                        type=float,
                        default=10)


class ModelLightFM:
    def __init__(self, val_rating_path):
        self.train_rating_path = MAIN_FOLDER.parent / 'train_rating.csv'
        self.val_rating_path = val_rating_path
        self.tag_csv_path = MAIN_FOLDER.parent / 'filtered_tag.csv'
        self.predictions = None
        self.unique_movies = None
        self.users_to_predict = None
        self.mapping_user_ids = None
        self.mapping_item_ids = None

    def fit(self, latent_size: int, learning_rate: float, item_alpha: float, epochs: int) -> LightFM:

        model = LightFM(no_components=latent_size,
                        learning_schedule="adagrad",
                        loss="logistic",
                        learning_rate=learning_rate,
                        item_alpha=item_alpha,
                        random_state=EvaluationParams.SEED.value)

        interactions, item_features, dataset = make_dataset(self.train_rating_path, self.tag_csv_path)
        self.mapping_user_ids, _, self.mapping_item_ids, _ = dataset.mapping()
        print('Dataset is built! Start fitting the model...')
        model.fit(interactions, item_features=item_features, epochs=epochs, verbose=True)
        return model

    def predict(self, latent_size: int, learning_rate: float, item_alpha: float, epochs: int) -> Tuple[List, dict, dict]:
        model = self.fit(latent_size, learning_rate, item_alpha, epochs)

        self.unique_movies = self.mapping_item_ids.values()
        unique_user_keys = self.mapping_user_ids.keys()

        df = pd.read_csv(self.val_rating_path)
        val_users = df[DatasetColumnName.USER_ID.value].unique()

        self.users_to_predict = [self.mapping_user_ids[user] for user in val_users if user in unique_user_keys]
        print(f'Model is fitted, start making predictions!')
        self.predictions = []

        for i, user in enumerate(self.users_to_predict):
            print(f'\rPredicted top@{EvaluationParams.K.value} movies for {i}/{len(self.users_to_predict)} users...', end='')
            input_user_id = np.array([user for _ in range(len(self.unique_movies))], dtype=np.int32)
            scores = model.predict(input_user_id, np.array(list(self.unique_movies), dtype=np.int32))
            movie_scores = dict(zip(self.unique_movies, scores))
            sorted_movie_scores = {k: v for k, v in sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)}
            self.predictions.append(list(sorted_movie_scores.keys())[:EvaluationParams.K.value])
        print(f'Predictions are saved! Let me compute precision@{EvaluationParams.K.value}.')
        return self.predictions, self.mapping_user_ids, self.mapping_item_ids

    def get_metric(self) -> float:
        precision = compute_precision(self.predictions, self.val_rating_path, self.mapping_item_ids,
                                      self.mapping_user_ids, self.users_to_predict, nested_pred=False)
        return precision


def main():
    lightfm = ModelLightFM(MAIN_FOLDER.parent / 'val_rating.csv')
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    args = parser.parse_args()
    print(f'\n Latent_size: {args.latent_size}\n Learning rate: {args.learning_rate}\n Item alpha: {args.item_alpha}\n '
          f'Epochs: {args.epochs}')

    lightfm.predict(args.latent_size, args.learning_rate, args.item_alpha, args.epochs)
    precision = lightfm.get_metric()
    print(f'Precision@{EvaluationParams.K.value}:{precision}')


if __name__ == "__main__":
    main()
