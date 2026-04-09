import pandas as pd
import numpy as np


def load_pickle(path: str):
    return pd.read_pickle(path)


def save_pickle(obj, path: str):
    import joblib
    joblib.dump(obj, path)


def class_balance(y):
    from collections import Counter
    return Counter(y)
