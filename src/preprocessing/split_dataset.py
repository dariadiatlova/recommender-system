import pandas as pd
import numpy as np
import logging.config


from data import MAIN_FOLDER
from src.common.util import DatasetColumnName, TrainTestSize, get_user_timestamp_threshold

logger = logging.getLogger(__name__)


def __get_time_wise_split():
    for _ in range(5):
        threshold_timestamp = get_user_timestamp_threshold()
        df = pd.read_csv(MAIN_FOLDER.parent / 'filtered_rating.csv')
        train_df = df[df[DatasetColumnName.TIMESTAMP.value] > threshold_timestamp]
        test_df = df[df[DatasetColumnName.TIMESTAMP.value] <= threshold_timestamp]
        test_size = np.round(test_df.shape[0] / df.shape[0], 2)
        print(f'Found a split with test size: ~{test_size}')
        if TrainTestSize.MIN_TEST_SIZE.value < test_size < TrainTestSize.MAX_TEST_SIZE.value:
            print(f'Seems good enough, will save!')
            return train_df, test_df


def main():
    try:
        train_df, test_df = __get_time_wise_split()
        train_data, test_data = train_df.iloc[:, :3], test_df.iloc[:, :3]
        train_data.to_csv(MAIN_FOLDER.parent / 'train_rating.csv', index=False)
        test_data.to_csv(MAIN_FOLDER.parent / 'test_rating.csv', index=False)
    except TypeError as e:
        logger.error('Could not find the split to satisfy chosen parameters :( Try again or change the parameters!')
        raise e


if __name__ == '__main__':
    main()
