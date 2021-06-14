import random
from enum import Enum, unique
import pandas as pd
import numpy as np
from data import MAIN_FOLDER


@unique
class DatasetColumnName(Enum):
    USER_ID = 'userId'
    MOVIE_ID = 'movieId'
    TIMESTAMP = 'timestamp'
    RATING = 'rating'


@unique
class TrainTestSize(Enum):
    MIN_TEST_SIZE = 0.25
    MAX_TEST_SIZE = 0.35
    TARGET_TEST_SIZE = 0.3
    SAMPLE_SIZE = 5000


@unique
class EvaluationParams(Enum):
    MIN_RATING = 4
    K = 25


def get_user_timestamp_threshold():
    """ 1. Picks a sample_size of unique ids of users.
        2. Sorts timestamps of chosen users in ascending order.
        3. Takes the the percentile timestamp.
        4. Returns the median of computed timestamp percentiles.
    """
    df = pd.read_csv(MAIN_FOLDER.parent / 'train_rating.csv')
    user_id = np.unique(df[DatasetColumnName.USER_ID.value].to_numpy())
    user_id_sample = random.sample(list(user_id), TrainTestSize.SAMPLE_SIZE.value)
    thirty_percentile_timestamp = []

    for user in user_id_sample:
        user_timestamp = df.loc[df[DatasetColumnName.USER_ID.value] == user, DatasetColumnName.TIMESTAMP.value]
        sorted_timestamp = user_timestamp.sort_values(ascending=False).to_numpy()
        current_percentile = sorted_timestamp[int(len(sorted_timestamp) * TrainTestSize.TARGET_TEST_SIZE.value)]
        thirty_percentile_timestamp.append(current_percentile)

    sorted_percentile_timestamp = pd.Series(thirty_percentile_timestamp).sort_values(ascending=False)
    optimal_split_value = int(TrainTestSize.SAMPLE_SIZE.value * 0.7)
    return sorted_percentile_timestamp.to_numpy()[optimal_split_value]
