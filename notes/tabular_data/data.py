import missingno
import pandas as pd


def visualize_features(df: pd.DataFrame, start_idx: int = 0, end_idx: int = 0):

    if end_idx:
        msno.matrix(df=df.iloc[:, start_idx:end_idx], figsize=(20, 14), color=(0.42, 0.1, 0.05))
    else:
        msno.matrix(df=df.iloc[:, start_idx], figsize=(20, 14), color=(0.42, 0.1, 0.05))
