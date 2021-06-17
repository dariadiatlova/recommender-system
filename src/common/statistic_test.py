import pandas as pd
from typing import List
from scipy.stats import chi2

from src.common.util import DatasetColumnName, EvaluationParams


def count_contingency_table(predictions_als: List, predictions_light_fm: List, test_dataset_path: str,
                            relevant_users: List, user_mapping: dict, item_mapping: dict, item_encoding) -> List:

    """ McNemar test of homogeneity..

        Statistic is reporting on the different correct or incorrect predictions between the two models.
        Null hypothesis of the test is that the two algorithms disagree to the same amount.
        If the null hypothesis is rejected, it suggests that there is evidence to suggest that
        algorithms disagree in different ways, that the disagreements are skewed.

    """

    df = pd.read_csv(test_dataset_path)

    # [correctly classified by both ALS and LightFM] [only by ALS] [only by LightFM] [classified wrong by both]
    yes_yes, yes_no, no_yes, no_no = 0, 0, 0, 0

    inverse_als_mapping = dict(zip(list(item_encoding.values()), list(item_encoding.keys())))
    inverse_light_fm_mapping = dict(zip(list(item_mapping.values()), list(item_mapping.keys())))

    for i, user in enumerate(relevant_users):
        als_pred = predictions_als[i]
        light_fm_pred = predictions_light_fm[i]

        movies_of_user = df[df[DatasetColumnName.USER_ID.value] == user_mapping[user]]
        relevant_movies = movies_of_user[DatasetColumnName.RATING.value] >= EvaluationParams.MIN_RATING.value
        relevant_movies = movies_of_user[relevant_movies][DatasetColumnName.MOVIE_ID.value].tolist()

        for movie_als, movie_lightfm in (als_pred, light_fm_pred):

            if item_mapping[inverse_als_mapping[movie_als]] in light_fm_pred:
                # movie is recommended by both ALS and LightFm
                if inverse_als_mapping[movie_als] in relevant_movies:
                    yes_yes += 1
                else:
                    no_no += 1

            # movie is recommended only by ALS
            else:
                if inverse_als_mapping[movie_als] in relevant_movies:
                    yes_no += 1
                else:
                    no_yes += 1

            # movie is recommended only by LightFM
            if item_encoding[inverse_light_fm_mapping[movie_lightfm]] not in als_pred:
                if inverse_light_fm_mapping[movie_lightfm] in relevant_movies:
                    no_yes += 1
                else:
                    yes_no += 1

    print(f'Yes/No count: {yes_no} /n No/Yes count: {no_yes} /n Yes/Yes count: {yes_yes} /n No/No count: {no_no}')

    if yes_no > 0 or no_yes > 0:
        statistic = (yes_no - no_yes)**2 / (yes_no + no_yes)
        p_value = chi2(statistic, 1)
        if p_value <= 0.05:
            print('The results are statistically significant.')
    else:
        print('Can not calculate statistic. No values in yes/no, no/yes categories.')
