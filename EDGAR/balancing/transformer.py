import warnings
import pandas as pd

from abc import ABC, abstractmethod
from typing import Dict, Protocol, Any, Tuple, Optional, List
from copy import deepcopy
from imblearn.under_sampling import RandomUnderSampler as RUS
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE as SM_C, SMOTENC as SM_NC, SMOTEN as SM_N
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from EDGAR.data.dataset import Dataset
from EDGAR.base.base_transformer import BaseTransformer
from EDGAR.base.utils import print_unbuffered


class Transformer(BaseTransformer, ABC):
    def __init__(self, name_sufix: str = '_transformed', verbose: bool = False) -> None:
        super().__init__()
        self.name_sufix = name_sufix
        self.__was_fitted = False
        self.verbose = verbose

    def fit(self, dataset: Dataset) -> None:
        d = dataset.train if dataset.was_split else dataset
        self._fit(d)
        self.__was_fitted = True

        if self.verbose:
            print_unbuffered(f'Transformer {self.__repr__()} was fitted with {dataset.name}')

    @abstractmethod
    def _fit(self, dataset: Dataset) -> None:
        pass

    @property
    def was_fitted(self) -> bool:
        return self.__was_fitted

    def transform(self, dataset: Dataset) -> Dataset:
        ds = deepcopy(dataset)
        if ds.was_split:
            d = ds.train
            new_train = self._transform(d)
            out = Dataset(name=ds.name + self.name_sufix, dataframe=None, target=None)
            out.custom_train_test_split(
                train=new_train,
                test=ds.test
            )
        else:
            out = self._transform(ds)

        if self.verbose:
            print_unbuffered(f'Transformer {self.__repr__()} transformed with {dataset.name}')

        return out

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

    def fit_resample(self, X, y) -> Tuple[pd.DataFrame, pd.Series]:
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

    def __init__(self, transformer: ImblearnProtocol, name_sufix: str = '_transformed', verbose: bool = False) -> None:
        self.__transformer = transformer
        super().__init__(name_sufix=name_sufix, verbose=verbose)

    def _change_transformer(self, transformer: ImblearnProtocol) -> None:
        try:
            check_is_fitted(transformer)
            warnings.warn('Transformer was not changed! The existing one was already fitted.')
        except NotFittedError:
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
                 random_state: int = None, *args, **kwargs) -> None:
        transformer = RUS(sampling_strategy=1 / imbalance_ratio, random_state=random_state, *args, **kwargs)
        super().__init__(transformer=transformer, name_sufix=name_sufix)

    def set_params(self, **params) -> None:
        if 'IR' in params.keys():
            IR = params.pop('IR')
            params['sampling_strategy'] = 1 / IR
        return super().set_params(**params)


class RandomOverSampler(TransformerFromIMBLEARN):
    def __init__(self, imbalance_ratio: float = 1, name_sufix: str = '_transformed',
                 random_state: int = None, *args, **kwargs) -> None:
        transformer = ROS(sampling_strategy=1 / imbalance_ratio, random_state=random_state, *args, **kwargs)
        super().__init__(transformer=transformer, name_sufix=name_sufix)

    def set_params(self, **params) -> None:
        if 'IR' in params.keys():
            IR = params.pop('IR')
            params['sampling_strategy'] = 1 / IR
        return super().set_params(**params)


class SMOTE(TransformerFromIMBLEARN):
    def __init__(self, imbalance_ratio: float = 1, name_sufix: str = '_transformed',
                 random_state: int = None, columns_categorical: Optional[List[str]] = None, *args, **kwargs) -> None:
        self.__sampling_strategy = 1 / imbalance_ratio
        self.__random_state = random_state
        self.__args = args
        self.__kwargs = kwargs
        self.__given_columns_categorical = columns_categorical
        transformer = SM_C(sampling_strategy=1 / imbalance_ratio, random_state=random_state, *args, **kwargs)
        super().__init__(transformer=transformer, name_sufix=name_sufix)

    def _fit(self, dataset: Dataset) -> None:
        if dataset.target is None:
            raise Exception('Target data is not provided!')
        if dataset.data is None:
            raise Exception('Data in dataset is not provided!')

        if self.__given_columns_categorical is not None:
            columns_categorical = self.__given_columns_categorical
        else:
            columns_categorical = list(dataset.data.select_dtypes(include=['category', 'object', 'int']))

        if len(columns_categorical) > 0:
            columns = dataset.data.columns
            if len(columns_categorical) == len(columns):
                super()._change_transformer(
                    SM_N(sampling_strategy=1 / self.__sampling_strategy, random_state=self.__random_state,
                         *self.__args, **self.__kwargs))
            else:
                all_columns = list(dataset.data.columns)
                categorical_indexes = [all_columns.index(c) for c in columns_categorical]
                super()._change_transformer(
                    SM_NC(categorical_indexes, sampling_strategy=1 / self.__sampling_strategy,
                          random_state=self.__random_state, *self.__args, **self.__kwargs))

        super()._fit(dataset)

    def set_params(self, **params) -> None:
        if 'IR' in params.keys():
            IR = params.pop('IR')
            params['sampling_strategy'] = 1 / IR
        return super().set_params(**params)
