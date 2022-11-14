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

    @property
    @abstractmethod
    def was_fitted(self) -> bool:
        pass

    def __str__(self):
        return f"BaseTransformer {self.__class__.__name__}"

    def __repr__(self):
        return f"<BaseTransformer {self.__class__.__name__}>"
