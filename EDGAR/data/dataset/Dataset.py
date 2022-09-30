import pandas as pd


class Dataset:
    def __init__(self, name: str, dataframe: pd.DataFrame, target: pd.Series):
        self.name = name
        self.data = dataframe
        self.target = target
