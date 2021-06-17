import argparse

from data import MAIN_FOLDER
from src.light_fm.model import ModelLightFM, configure_arguments


def test_light_fm_model():
    lightfm = ModelLightFM(MAIN_FOLDER.parent / 'test_rating.csv')
    parser = argparse.ArgumentParser()
    configure_arguments(parser)
    args = parser.parse_args()
    predictions, user_mapping_ids, item_mapping_ids = lightfm.predict(args.latent_size,
                                                                      args.learning_rate,
                                                                      args.item_alpha,
                                                                      args.epochs)
    return predictions, item_mapping_ids
