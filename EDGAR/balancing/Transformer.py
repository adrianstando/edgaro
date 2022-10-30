import warnings
from abc import ABC, abstractmethod
from typing import Dict, Protocol, Any
from imblearn.under_sampling import RandomUnderSampler as RUS
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE as SM_C, SMOTENC as SM_NC, SMOTEN as SM_N
from sklearn.utils.validation import check_is_fitted
from EDGAR.base.BaseTransformer import BaseTransformer
from EDGAR.data.Dataset import Dataset


class Transformer(BaseTransformer, ABC):
    def __init__(self, name_sufix: str = '_transformed') -> None:
        super().__init__()
        self.name_sufix = name_sufix

    def fit(self, dataset: Dataset) -> None:
        d = dataset.train if dataset.was_split else dataset
        self._fit(d)

    @abstractmethod
    def _fit(self, dataset: Dataset) -> None:
        pass

    def transform(self, dataset: Dataset) -> Dataset:
        d = dataset.train if dataset.was_split else dataset
        return self._transform(d)

    @abstractmethod
    def _transform(self, dataset: Dataset) -> Dataset:
        pass

    @abstractmethod
    def set_params(self, **params) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        pass

    def set_dataset_suffixes(self, name_sufix: str) -> None:
        self.name_sufix = name_sufix

    def __str__(self) -> str:
        return f"Transformer {self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"<Transformer {self.__class__.__name__}>"


class ImblearnProtocol(Protocol):
    def fit(self, X, y) -> Any:
        ...

    def fit_resample(self, X, y) -> Any:
        ...

    def get_params(self) -> Dict:
        ...

    def set_params(self, **params) -> Any:
        ...


class TransformerFromIMBLEARN(Transformer):
    """
    for example:

    from imblearn.under_sampling import RandomUnderSampler
    dataset = DatasetFromOpenML(task_id=3)
    transformator = TransformerFromIMBLEARN(RandomUnderSampler(sampling_strategy=n_minority/n_majority, random_state=42))
    transformator.fit(dataset)
    transformator.transform(dataset)
    """

    def __init__(self, transformer: ImblearnProtocol, name_sufix: str = '_transformed') -> None:
        self.__transformer = transformer
        super().__init__(name_sufix=name_sufix)

    def _change_transformer(self, transformer: ImblearnProtocol) -> None:
        if check_is_fitted(transformer):
            warnings.warn('Transformer was not changed! The existing one was already fitted.')
        else:
            self.__transformer = transformer

    def _fit(self, dataset: Dataset) -> None:
        self.__transformer.fit(dataset.data, dataset.target)

    def _transform(self, dataset: Dataset) -> Dataset:
        X, y = self.__transformer.fit_resample(dataset.data, dataset.target)
        name = dataset.name + self.name_sufix
        return Dataset(name=name, dataframe=X, target=y)

    def get_imblearn_transformer(self) -> ImblearnProtocol:
        return self.__transformer

    def set_params(self, **params) -> None:
        self.__transformer.set_params(**params)

    def get_params(self) -> Dict:
        return self.__transformer.get_params()


class RandomUnderSampler(TransformerFromIMBLEARN):
    def __init__(self, imbalance_ratio: float = 1, name_sufix: str = '_transformed',
                 random_state: int = None, *args, **kwargs):
        transformer = RUS(sampling_strategy=1 / imbalance_ratio, random_state=random_state, *args, **kwargs)
        super().__init__(transformer=transformer, name_sufix=name_sufix)

    def set_params(self, **params):
        if 'IR' in params.keys():
            IR = params.pop('IR')
            params['sampling_strategy'] = 1 / IR
        return super().set_params(**params)


class RandomOverSampler(TransformerFromIMBLEARN):
    def __init__(self, imbalance_ratio: float = 1, name_sufix: str = '_transformed',
                 random_state: int = None, *args, **kwargs):
        transformer = ROS(sampling_strategy=1 / imbalance_ratio, random_state=random_state, *args, **kwargs)
        super().__init__(transformer=transformer, name_sufix=name_sufix)

    def set_params(self, **params):
        if 'IR' in params.keys():
            IR = params.pop('IR')
            params['sampling_strategy'] = 1 / IR
        return super().set_params(**params)


class SMOTE(TransformerFromIMBLEARN):
    def __init__(self, imbalance_ratio: float = 1, name_sufix: str = '_transformed',
                 random_state: int = None, *args, **kwargs):
        self.__sampling_strategy = 1 / imbalance_ratio
        self.__random_state = random_state
        self.__args = args
        self.__kwargs = kwargs
        transformer = SM_C(sampling_strategy=1 / imbalance_ratio, random_state=random_state, *args, **kwargs)
        super().__init__(transformer=transformer, name_sufix=name_sufix)

    def _fit(self, dataset: Dataset):
        columns_categorical = list(dataset.data.select_dtypes(include=['category', 'object', 'int']))

        if len(columns_categorical) > 0:
            columns = dataset.data.columns
            if len(columns_categorical) == len(columns):
                super()._change_transformer(
                    SM_N(sampling_strategy=1 / self.__sampling_strategy, random_state=self.__random_state,
                         *self.__args, **self.__kwargs))
            elif len(columns_categorical) < len(columns):
                SM_NC(columns_categorical, sampling_strategy=1 / self.__sampling_strategy,
                      random_state=self.__random_state, *self.__args, **self.__kwargs)

        super()._fit(dataset)

    def set_params(self, **params):
        if 'IR' in params.keys():
            IR = params.pop('IR')
            params['sampling_strategy'] = 1 / IR
        return super().set_params(**params)
