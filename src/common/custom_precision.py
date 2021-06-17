import numpy as np
import pandas as pd
from src.common.util import DatasetColumnName, EvaluationParams


def compute_precision(predictions, validation_dataset_path: str, movie_filtered_ids, user_filtered_ids, users,
                      nested_pred=True) -> float:
    inverse_encoding_us = dict(zip(user_filtered_ids.values(), user_filtered_ids.keys()))
    inverse_encoding_mv = dict(zip(movie_filtered_ids.values(), movie_filtered_ids.keys()))

    df = pd.read_csv(validation_dataset_path)
    precision = []

    for i, user in enumerate(users):
        y_pred = predictions[i]
        movies_of_user = df[df[DatasetColumnName.USER_ID.value] == inverse_encoding_us[user]]

        relevant_movies = movies_of_user[DatasetColumnName.RATING.value] >= EvaluationParams.MIN_RATING.value
        relevant_movies = movies_of_user[relevant_movies][DatasetColumnName.MOVIE_ID.value].tolist()

        irrelevant_movies = movies_of_user[DatasetColumnName.RATING.value] < EvaluationParams.MIN_RATING.value
        irrelevant_movies = movies_of_user[irrelevant_movies][DatasetColumnName.MOVIE_ID.value].tolist()

        true_positive, false_positive = 0, 0

        for movie in y_pred:
            if nested_pred:
                movie = movie[0]
            movie_id = inverse_encoding_mv[movie]
            if movie_id in relevant_movies:
                true_positive += 1
            elif movie_id in irrelevant_movies:
                false_positive += 1

        if true_positive > 0 or false_positive > 0:
            precision.append(true_positive / (true_positive + false_positive))

    print(f'Evaluated on {len(precision)} users')

    return np.mean(precision)
