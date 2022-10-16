from abc import ABC, abstractmethod
from imblearn.base import BaseSampler
from imblearn.under_sampling import RandomUnderSampler as RUS
from imblearn.over_sampling import RandomOverSampler as ROS
from EDGAR.base.BaseTransformer import BaseTransformer
from EDGAR.data.Dataset import Dataset


class Transformer(BaseTransformer, ABC):
    def __init__(self, name_sufix: str = '_transformed'):
        super().__init__()
        self.name_sufix = name_sufix

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

    def set_name_sufix(self, name_sufix: str):
        self.name_sufix = name_sufix


class TransformerFromIMBLEARN(Transformer):
    """
    for example:

    from imblearn.under_sampling import RandomUnderSampler
    dataset = DatasetFromOpenML(task_id=3)
    transformator = TransformerFromIMBLEARN(RandomUnderSampler(sampling_strategy=n_minority/n_majority, random_state=42))
    transformator.fit(dataset)
    transformator.transform(dataset)
    """

    def __init__(self, transformer: BaseSampler, name_sufix: str = '_transformed'):
        self.__transformer = transformer
        super().__init__(name_sufix=name_sufix)

    def fit(self, dataset: Dataset):
        return self.__transformer.fit(dataset.data, dataset.target)

    def transform(self, dataset: Dataset) -> Dataset:
        X, y = self.__transformer.fit_resample(dataset.data, dataset.target)
        name = dataset.name + self.name_sufix
        return Dataset(name=name, dataframe=X, target=y)

    def get_imblearn_transformer(self):
        return self.__transformer

    def set_params(self, **params):
        return self.__transformer.set_params(**params)

    def get_params(self):
        return self.__transformer.get_params()


class RandomUnderSampler(TransformerFromIMBLEARN):
    def __init__(self, imbalance_ratio: float = 1, name_sufix: str = '_transformed', random_state: int = None, *args, **kwargs):
        transformer = RUS(sampling_strategy=1/imbalance_ratio, random_state=random_state, *args, **kwargs)
        super().__init__(transformer=transformer, name_sufix=name_sufix)


class RandomOverSampler(TransformerFromIMBLEARN):
    def __init__(self, imbalance_ratio: float = 1, name_sufix: str = '_transformed', random_state: int = None, *args, **kwargs):
        transformer = ROS(sampling_strategy=1/imbalance_ratio, random_state=random_state, *args, **kwargs)
        super().__init__(transformer=transformer, name_sufix=name_sufix)
