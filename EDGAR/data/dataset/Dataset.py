import pandas as pd
from EDGAR.data.transformators import Transformator


class Dataset:
    def __init__(self, name: str, dataframe: pd.DataFrame, target: pd.Series):
        self.name = name
        self.data = dataframe
        self.target = target

    def transform(self, transformator: Transformator):
        X_new, y_new = transformator.transform(self)
        return Dataset(name=self.name, dataframe=X_new, target=y_new)
