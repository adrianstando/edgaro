from typing import List, Dict, Union
from abc import ABC, abstractmethod
from EDGAR.data.dataset import Dataset


class Transformer(ABC):
    def __init__(self, name_sufix: str = '_transformed'):
        self.name_sufix = name_sufix

    @abstractmethod
    def fit(self, dataset: Dataset):
        pass

    @abstractmethod
    def transform(self, dataset: Dataset) -> Dataset:
        pass

    @abstractmethod
    def set_params(self, **params):
        pass

    @abstractmethod
    def get_params(self) -> Union[Dict, List]:
        pass

    def set_name_sufix(self, name_sufix: str):
        self.name_sufix = name_sufix
