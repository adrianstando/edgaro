from abc import ABC, abstractmethod
from typing import Dict, Any

from EDGAR.data.Dataset import Dataset


class BaseTransformer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, dataset: Dataset) -> None:
        pass

    @abstractmethod
    def transform(self, dataset: Dataset) -> Dataset:
        pass

    @abstractmethod
    def set_params(self, **params) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def was_fitted(self) -> bool:
        pass

    def __str__(self) -> str:
        return f"BaseTransformer {self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"<BaseTransformer {self.__class__.__name__}>"
