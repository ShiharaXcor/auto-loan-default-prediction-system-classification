import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Age_Days' in df.columns:
        df['Age_Years'] = (df['Age_Days'] / 365).astype(int)
    return df


def top_k_rare_flags(df: pd.DataFrame, col: str, k: int = 5) -> pd.DataFrame:
    df = df.copy()
    topk = df[col].value_counts().index[-k:]
    for t in topk:
        df[f'{col}_rare_{str(t)}'] = (df[col] == t).astype(int)
    return df
