import numpy as np
import os


def check_filtered_metadata(path):
    if not os.path.isfile(path):
        return False

    arr = np.load(path, allow_pickle=True)
    if not arr.size:
        return False

    return True
