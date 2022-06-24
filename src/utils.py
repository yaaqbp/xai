import random
import numpy as np
import pandas as pd


def seed_everything(seed: int) -> int:
    random.seed(seed)
    np.random.seed(seed)

    return seed


def read_data(data_path):
    data = pd.read_csv(data_path)

    return data.drop('Y', axis=1), data['Y']