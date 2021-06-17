import argparse

from src.als.model import configure_arguments, ALS


def test_als_model():
    """
    Return:
        - list: predictions on test dataset with default parameters set in configure_arguments
        - list: relevant users ids encoded
        - dict: items and its encoding into original movie_ids
        - dict: users and its encoding into original user_ids
    """
    als = ALS()
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    args = parser.parse_args()
    loss, model = als.train(args.factors,
                            args.regularization,
                            args.iterations,
                            args.alpha,
                            args.train_loss,
                            args.train_dataset_path)

    predictions, relevant_users, item_encoding = als.test(model)
    return predictions, relevant_users, item_encoding
