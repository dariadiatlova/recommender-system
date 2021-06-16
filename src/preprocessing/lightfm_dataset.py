import pandas as pd
import numpy as np

from lightfm.data import Dataset
from scipy import sparse
from src.common.util import DatasetColumnName, EvaluationParams
from typing import Tuple


def mask_movies(matrix: sparse.coo_matrix) -> sparse.coo_matrix:
    """ Converts all ratings bellow min_rating into -1 (disliked by user),
        all ratings above min_rating into 1 (liked by user)
    """
    matrix = matrix.tocsr()
    irrelevant_movies_mask = np.array(matrix[matrix.nonzero()] < EvaluationParams.MIN_RATING.value)[0]
    matrix[matrix.nonzero()[0][irrelevant_movies_mask], matrix.nonzero()[1][irrelevant_movies_mask]] = -1
    matrix[matrix > 0] = 1
    return matrix.tocoo()


def make_dataset(rating_dataset_path: str, tag_dataset_path: str) -> Tuple[sparse.coo_matrix, sparse.coo_matrix]:

    df_rating = pd.read_csv(rating_dataset_path)
    df_tag = pd.read_csv(tag_dataset_path)

    dataset = Dataset()
    dataset.fit(df_rating[DatasetColumnName.USER_ID.value].unique(),
                df_rating[DatasetColumnName.MOVIE_ID.value].unique(),
                item_features=df_tag[DatasetColumnName.TAG_ID.value].unique())

    _, weights = dataset.build_interactions(
        [tuple(row) for row in df_rating.drop([DatasetColumnName.TIMESTAMP.value], axis=1).values]
    )
    user_item_interaction = mask_movies(weights)
    item_features = dataset.build_item_features([(row[0], {row[1]: row[2]}) for row in df_tag.values])

    return user_item_interaction, item_features
