import pickle
from typing import Tuple


def model_loader(model_path: str, encoder_path: str) -> Tuple:
    """

    :rtype: model
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(encoder_path, "rb") as ec:
        enc = pickle.load(ec)

    return model, enc
