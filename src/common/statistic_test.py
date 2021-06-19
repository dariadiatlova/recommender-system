import pandas as pd

from typing import List
from data import MAIN_FOLDER
from src.common.util import DatasetColumnName, EvaluationParams
from src.als.test import test_als_model
from src.light_fm.test import test_light_fm_model
from statsmodels.stats.contingency_tables import mcnemar


def count_contingency_table(predictions_als: List, predictions_light_fm: List, relevant_users: List,
                            user_mapping: dict, item_mapping: dict, item_encoding: dict) -> List[List[int]]:
    """ Gather predictions of 2 models on test dataset, compare recommended items for users with values
        in test datasets and creates contingency_table with correct and incorrect recommendations distribution:

        Algorithm     | ALS - yes  | ALS - no  |
                      |––––––––––––––––––––––––|
        LightFm - yes | yes / yes  |  yes / no |
        ––––––––––––––|            |           |
        Light_FM – no | no / yes   |  no / no  |

        Yes: item has rating equal or above 4 in test dataset and was recommended or has rating below 4 and
             was not recommended.

        No: item has rating bellow 4 in test dataset and was recommended or has rating equal or above 4 and
             was not recommended.

        Compute not all available items in tests dataset but only those that were recommended by either ALS or LightFM
        algorithm, and which rating has presence in test dataset.
    """
    df = pd.read_csv(MAIN_FOLDER.parent / 'test_rating.csv')

    # [correctly classified by both ALS and LightFM] [only by ALS] [only by LightFM] [classified wrong by both]
    yes_yes, yes_no, no_yes, no_no = 0, 0, 0, 0

    inverse_als_mapping = dict(zip(list(item_encoding.values()), list(item_encoding.keys())))
    inverse_light_fm_mapping = dict(zip(list(item_mapping.values()), list(item_mapping.keys())))
    inverse_light_fm_user_mapping = dict(zip(list(user_mapping.values()), list(user_mapping.keys())))

    for i, user in enumerate(relevant_users):
        als_pred = predictions_als[i]
        light_fm_pred = predictions_light_fm[i]

        movies_of_user = df[df[DatasetColumnName.USER_ID.value] == inverse_light_fm_user_mapping[user]]
        relevant_movies = movies_of_user[DatasetColumnName.RATING.value] >= EvaluationParams.MIN_RATING.value
        relevant_movies = movies_of_user[relevant_movies][DatasetColumnName.MOVIE_ID.value].tolist()

        for movie_als, movie_lightfm in zip(als_pred, light_fm_pred):

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
    return [[yes_yes, yes_no], [no_yes, no_no]]


def main():
    """ McNemar test of homogeneity.

        Statistic is reporting on the different correct or incorrect predictions between the two models.
        Null hypothesis of the test is that the two algorithms disagree to the same amount.
        If the null hypothesis is rejected, it suggests that there is evidence to suggest that
        algorithms disagree in different ways, that the disagreements are skewed.

    """
    predictions_als, relevant_users_als, item_encoding, user_encoding, precision_als = test_als_model()
    predictions_light_fm, relevant_users_light_fm, user_mapping, item_mapping, precision_lightfm = test_light_fm_model()

    print(f'Precision@{EvaluationParams.K.value} with ALS algorithm:{precision_als}.')
    print(f'Precision@{EvaluationParams.K.value} with LightFN algorithm:{precision_lightfm}.')

    predictions = []
    for user_predictions in predictions_als:
        predictions.append([movie[0] for movie in user_predictions])

    table = count_contingency_table(predictions,
                                             predictions_light_fm,
                                             relevant_users_light_fm,
                                             user_mapping,
                                             item_mapping,
                                             item_encoding)
    mc_nemar_test_result = mcnemar(table, exact=True)
    if mc_nemar_test_result.pvalue <= 0.05:
        print('The results are statistically significant.')
    else:
        print('Can not calculate statistic. No values in yes/no, no/yes categories.')


if __name__ == "__main__":
    main()
