from abc import ABC, abstractmethod
from EDGAR.data.Dataset import Dataset


class BaseTransformer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, dataset: Dataset):
        pass

    @abstractmethod
    def transform(self, dataset: Dataset):
        pass

    @abstractmethod
    def set_params(self, **params):
        pass

    @abstractmethod
    def get_params(self):
        pass
