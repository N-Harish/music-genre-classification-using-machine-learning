import pandas as pd
import numpy as np
from typing import List


def load_data(file_path: str, col_to_drop: List[str], target_col: str):
    data = pd.read_csv(file_path)
    targets = data[target_col]
    targets = np.array(targets)
    targets = targets.reshape(-1, 1)
    inputs = inputs = data.drop(col_to_drop, axis=1)
    return inputs, targets
