from abc import abstractmethod
from EDGAR.data.dataset import Dataset


class Transformator:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def transform(self, X, y):
        pass
