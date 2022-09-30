import pandas as pd
from EDGAR.data.dataset import Dataset


class DatasetFromCSV(Dataset):
    def __init__(self, path: str, target: str, name: str = 'dataset', *args, **kwargs):
        X = pd.read_csv(path, *args, **kwargs)
        y = X[target]
        y = pd.Series(y, name='target')
        X = X.drop([target], axis=1)
        super(Dataset, self).__init__(name=name, dataframe=X, target=y)

