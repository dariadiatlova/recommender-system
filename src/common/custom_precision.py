import pandas as pd
import numpy as np

from src.common.util import DatasetColumnName, EvaluationParams


def compute_precision(model, sparse_user_item, movie_filtered_ids, user_filtered_ids, val_dataset_path):

    df = pd.read_csv(val_dataset_path)
    users = df[DatasetColumnName.USER_ID.value].unique()
    inverse_encoded_movies = dict(zip(list(movie_filtered_ids.values()), list(movie_filtered_ids.keys())))
    prediction_statistic, unknown = [], 0

    for user in users:
        true_positive, false_positive = 0, 0
        try:
            user_id = user_filtered_ids[user]

            predictions = model.recommend(user_id, sparse_user_item,
                                          N=EvaluationParams.K.value,
                                          filter_already_liked_items=False)

            movies_of_user = df[df[DatasetColumnName.USER_ID.value] == user]
            relevant_movies = movies_of_user[DatasetColumnName.RATING.value] >= EvaluationParams.MIN_RATING.value
            relevant_movies = movies_of_user[relevant_movies][DatasetColumnName.MOVIE_ID.value].tolist()

            if relevant_movies:
                for prediction in predictions:
                    if inverse_encoded_movies[prediction[0]] in relevant_movies:
                        if prediction[1] >= 1:
                            true_positive += 1
                        else:
                            false_positive += 1
                if true_positive > 0 or false_positive > 0:
                    prediction_statistic.append(true_positive / (true_positive + false_positive))

        except KeyError:
            unknown += 1
            # ignore cases when meet users we have not seen on test dataset
            pass

    print(f'Evaluated on {len(users) - unknown} users')
    return np.round(np.mean(prediction_statistic), 3)
