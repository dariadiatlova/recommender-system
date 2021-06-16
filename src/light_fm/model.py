import argparse
import numpy as np
import pandas as pd
from data import MAIN_FOLDER
from lightfm import LightFM

from src.common.custom_precision import compute_precision
from src.common.util import EvaluationParams, DatasetColumnName
from src.light_fm.build_dataset import make_dataset


def configure_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('-ls', '--latent_size',
                        help='The dimensionality of the feature latent embeddings. Default is 10. ',
                        type=int,
                        default=10)

    parser.add_argument('-lr', '--learning_rate',
                        help='The dimensionality of the feature latent embeddings. Default is 0.01.',
                        type=float,
                        default=0.01)

    parser.add_argument('-ia', '--item_alpha',
                        help='L2 penalty on item features. Default is 1.',
                        type=float,
                        default=0)

    parser.add_argument('-e', '--epochs',
                        help='Number of epochs for training. Default is 1.',
                        type=float,
                        default=1)


class ModelLightFM:
    def __init__(self):
        self.train_rating_path = MAIN_FOLDER.parent / 'train_rating.csv'
        self.val_rating_path = MAIN_FOLDER.parent / 'val_rating.csv'
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

    def predict(self, latent_size: int, learning_rate: float, item_alpha: float, epochs: int) -> None:
        model = self.fit(latent_size, learning_rate, item_alpha, epochs)

        self.unique_movies = list(self.mapping_item_ids.values())
        unique_users = list(self.mapping_user_ids.keys())
        df = pd.read_csv(self.val_rating_path)
        val_users = df[DatasetColumnName.USER_ID.value]
        self.users_to_predict = [self.mapping_user_ids[user] for user in val_users if user in unique_users]
        self.users_to_predict = np.unique(self.users_to_predict)
        print(f'Model is fitted, start making predictions!')
        predictions = []

        for i, user in enumerate(self.users_to_predict):
            print(f'\rPredicted top@{EvaluationParams.K.value} movies for {i}/{len(self.users_to_predict)} users...', end='')
            scores = model.predict(np.array([user for _ in range(len(self.unique_movies))], dtype=np.int32),
                                   self.unique_movies)
            movie_scores = dict(zip(self.unique_movies, scores))
            sorted_movie_scores = {k: v for k, v in sorted(movie_scores.items(), key=lambda item: item[1])}
            predictions.append(sorted_movie_scores.keys()[:EvaluationParams.K.value])
        print(f'Predictions are saved! Let me compute precision@{EvaluationParams.K.value}.')

    def get_metric(self) -> float:
        precision = compute_precision(self.predictions, self.val_rating_path, self.mapping_item_ids,
                                      self.mapping_item_ids, self.users_to_predict)
        return precision


def main():
    lightfm = ModelLightFM()
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
