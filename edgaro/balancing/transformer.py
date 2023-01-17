import warnings
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, Protocol, Any, Tuple, Optional, List, Union
from copy import deepcopy
from imblearn.under_sampling import RandomUnderSampler as RUS
from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE as SM_C, SMOTENC as SM_NC, SMOTEN as SM_N
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from edgaro.data.dataset import Dataset
from edgaro.data.dataset_array import DatasetArray
from edgaro.base.base_transformer import BaseTransformer
from edgaro.base.utils import print_unbuffered


class Transformer(BaseTransformer, ABC):
    """
    The abstract class to define balancing transformations for a single Dataset.

    Parameters
    ----------
    name_sufix : str, default='_transformed'
        Sufix to be set to a transformed Dataset.
    verbose : bool, default=False
        Print messages during calculations.

    Attributes
    ----------
    name_sufix : str
        Sufix to be set to a transformed Dataset.
    verbose : bool
        Print messages during calculations.
    """

    def __init__(self, name_sufix: str = '_transformed', verbose: bool = False) -> None:
        super().__init__()
        self.name_sufix = name_sufix
        self.__was_fitted = False
        self.verbose = verbose

    def fit(self, dataset: Dataset) -> None:
        """
        Fit the transformer.

        Parameters
        ----------
        dataset : Dataset
            The object to fit Transformer on.
        """
        if dataset.target is None:
            raise Exception('Target data is not provided!')
        if dataset.data is None:
            raise Exception('Data in dataset is not provided!')

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
        """
        The information whether the Transformer was fitted.

        Returns
        -------
        bool
        """
        return self.__was_fitted

    def transform(self, dataset: Dataset) -> Union[Dataset, DatasetArray]:
        """
        Transform the object.

        Parameters
        ----------
        dataset : Dataset
            The object to be transformed.

        Returns
        -------
        Dataset, DatasetArray
            The transformed object.
        """
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

        if dataset.majority_class_label is None:
            names, counts = np.unique(dataset.target, return_counts=True)
            out.majority_class_label = names[1 if counts[0] < counts[1] else 0]
        else:
            out.majority_class_label = dataset.majority_class_label

        return out

    @abstractmethod
    def _transform(self, dataset: Dataset) -> Dataset:
        pass

    @abstractmethod
    def set_params(self, **params) -> None:
        """
        Set params for Transformer.

        Parameters
        ----------
        params : dict
            The parameters to be set.
        """
        pass

    @abstractmethod
    def get_params(self) -> Union[Dict, List]:
        """
        Get parameters of Transformer.

        Returns
        -------
        Dict, list
            The parameters.
        """
        pass

    def set_dataset_suffixes(self, name_sufix: Union[str, list]) -> None:
        """
        Set sufix to be set to transformed Dataset.

        Parameters
        ----------
        name_sufix : str, list
            Sufix to be set to a transformed Dataset.
        """
        self.name_sufix = name_sufix

    def __str__(self) -> str:
        return f"Transformer {self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"<Transformer {self.__class__.__name__}>"


class ImblearnProtocol(Protocol):
    """
    A Protocol to define the expected structure of a Transformer from `imblearn` library.
    """
    def fit(self, X, y) -> Any:
        ...

    def fit_resample(self, X, y) -> Tuple[pd.DataFrame, pd.Series]:
        ...

    def get_params(self) -> Dict:
        ...

    def set_params(self, **params) -> Any:
        ...


class TransformerFromIMBLEARN(Transformer):
    """ Create balancing Transformer from Transformer implemented in `imblearn` library.

    Parameters
    ----------
    transformer : ImblearnProtocol
        Transformer from imblearn` library.
    name_sufix : str, default='_transformed'
        Sufix to be set to a transformed Dataset.
    verbose : bool, default=False
        Print messages during calculations.

    Examples
    ----------

    >>> from test.resources.objects import *
    >>> from imblearn.under_sampling import RandomUnderSampler
    >>> dataset = Dataset(name_1, df_1, target_1)
    >>> transformator = TransformerFromIMBLEARN(RandomUnderSampler(sampling_strategy=1, random_state=42))
    >>> transformator.fit(dataset)
    >>> transformator.transform(dataset)
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

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the object.

        Parameters
        ----------
        dataset : Dataset
            The object to be transformed.

        Returns
        -------
        Dataset
            The transformed object.
        """
        return super().transform(dataset)

    def _transform(self, dataset: Dataset) -> Dataset:
        X, y = self.__transformer.fit_resample(dataset.data, dataset.target)
        name = dataset.name + self.name_sufix
        return Dataset(name=name, dataframe=X, target=y)

    def get_imblearn_transformer(self) -> ImblearnProtocol:
        """
        Get the base transformer object from `imblearn` library.

        Returns
        -------
        ImblearnProtocol
        """
        return self.__transformer

    def set_params(self, **params) -> None:
        """
        Set params for Transformer.

        The function allows using `imbalance_ratio` and `IR` parameters, which are transformed to `sampling_strategy`
        parameter in `imblearn` Transformer.

        Parameters
        ----------
        params : dict
            The parameters to be set.
        """
        if 'imbalance_ratio' in params.keys():
            x = params.pop('imbalance_ratio')
            params['sampling_strategy'] = 1/x
        elif 'IR' in params.keys():
            IR = params.pop('IR')
            params['sampling_strategy'] = 1 / IR
        self.__transformer.set_params(**params)

    def get_params(self) -> Dict:
        """
        Get parameters of Transformer.

        Returns
        -------
        Dict, list
            The parameters.
        """
        return self.__transformer.get_params()

    def set_dataset_suffixes(self, name_sufix: str) -> None:
        """
        Set sufix to be set to transformed Dataset.

        Parameters
        ----------
        name_sufix : str
            Sufix to be set to a transformed Dataset.
        """
        super().set_dataset_suffixes(name_sufix)


class RandomUnderSampler(TransformerFromIMBLEARN):
    """ Create Random Under Sampling transformer.

    Parameters
    ----------
    imbalance_ratio : float, default=1
        Imbalance Ratio after transformations.
    name_sufix : str, default='_transformed'
        Sufix to be set to a transformed Dataset.
    verbose : bool, default=False
        Print messages during calculations.
    random_state : int, optional
        Random state seed.
    *args : tuple, optional
        Additional parameter for Random Under Sampling transformer from `imblearn`.
    **kwargs : dict, optional
        Additional parameter for Random Under Sampling transformer from `imblearn`.
    """

    def __init__(self, imbalance_ratio: float = 1, name_sufix: str = '_transformed',
                 random_state: int = None, verbose: bool = False, *args, **kwargs) -> None:
        transformer = RUS(sampling_strategy=1 / imbalance_ratio, random_state=random_state, *args, **kwargs)
        super().__init__(transformer=transformer, name_sufix=name_sufix, verbose=verbose)

    def set_params(self, **params) -> None:
        return super().set_params(**params)


class RandomOverSampler(TransformerFromIMBLEARN):
    """ Create Random Over Sampling transformer.

    Parameters
    ----------
    imbalance_ratio : float, default=1
        Imbalance Ratio after transformations.
    name_sufix : str, default='_transformed'
        Sufix to be set to a transformed Dataset.
    verbose : bool, default=False
        Print messages during calculations.
    random_state : int, optional
        Random state seed.
    *args : tuple, optional
        Additional parameter for Random Over Sampling transformer from `imblearn`.
    **kwargs : dict, optional
        Additional parameter for Random Over Sampling transformer from `imblearn`.
    """

    def __init__(self, imbalance_ratio: float = 1, name_sufix: str = '_transformed',
                 random_state: int = None, verbose: bool = False, *args, **kwargs) -> None:
        transformer = ROS(sampling_strategy=1 / imbalance_ratio, random_state=random_state, *args, **kwargs)
        super().__init__(transformer=transformer, name_sufix=name_sufix, verbose=verbose)

    def set_params(self, **params) -> None:
        return super().set_params(**params)


class SMOTE(TransformerFromIMBLEARN):
    """ Create SMOTE/SMOTENC/SMOTEN transformer.

    The method works also with categorical variables. The method guesses that columns of types 'category', 'object',
    'int' are always categorical. Keep that in mind before using this Transformer!

    Parameters
    ----------
    imbalance_ratio : float, default=1
        Imbalance Ratio after transformations.
    name_sufix : str, default='_transformed'
        Sufix to be set to a transformed Dataset.
    verbose : bool, default=False
        Print messages during calculations.
    random_state : int, optional
        Random state seed.
    *args : tuple, optional
        Additional parameter for SMOTE/SMOTENC/SMOTEN transformer from `imblearn`.
    **kwargs : dict, optional
        Additional parameter for SMOTE/SMOTENC/SMOTEN transformer from `imblearn`.
    """

    def __init__(self, imbalance_ratio: float = 1, name_sufix: str = '_transformed', random_state: int = None,
                 columns_categorical: Optional[List[str]] = None, verbose: bool = False, *args, **kwargs) -> None:
        self.__sampling_strategy = 1 / imbalance_ratio
        self.__random_state = random_state
        self.__args = args
        self.__kwargs = kwargs
        self.__given_columns_categorical = columns_categorical
        transformer = SM_C(sampling_strategy=1 / imbalance_ratio, random_state=random_state, *args, **kwargs)
        super().__init__(transformer=transformer, name_sufix=name_sufix, verbose=verbose)

    def _fit(self, dataset: Dataset) -> None:
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
        return super().set_params(**params)
