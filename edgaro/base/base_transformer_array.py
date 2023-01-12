from __future__ import annotations

import numpy as np

from typing import List, Dict, Optional, Any, Union
from copy import deepcopy

from edgaro.data.dataset import Dataset
from edgaro.data.dataset_array import DatasetArray
from edgaro.base.base_transformer import BaseTransformer
from edgaro.base.utils import print_unbuffered


class BaseTransformerArray:
    """
    Create a class to apply BaseTransformer transformation with more than one set of parameters and/or
    to each of the Dataset objects in DatasetArray.

    Parameters
    ----------
    base_transformer : BaseTransformer
        The object defining the transformation procedure.
    parameters : list[list, Dict[str, Any]]], optional
        The list of parameters for base_transformer. If the object is used for a DatasetArray object,
        the parameter list should be nested. For details, see Examples section.
    transformer_sufix : str
        A suffix to be added to name in a transformed object.

    Attributes
    ----------
    transformer_sufix : str
        A suffix to be added to name in a transformed object.

    Examples
    ----------
    Example 1

    >>> from test.resources.objects import *
    >>> from edgaro.data.dataset import Dataset
    >>> from edgaro.data.dataset_array import DatasetArray
    >>> from edgaro.balancing.transformer import RandomUnderSampler
    >>> from edgaro.base.base_transformer_array import BaseTransformerArray
    >>> df = Dataset(name_1, df_1, target_1)
    >>> params = [{'sampling_strategy': 0.98}, {'sampling_strategy': 1}]
    >>> transformer = RandomUnderSampler()
    >>> array = BaseTransformerArray(transformer, parameters=params)
    >>> array.fit(df)
    >>> array.transform(df)

    Example 2

    >>> from test.resources.objects import *
    >>> from edgaro.data.dataset import Dataset
    >>> from edgaro.data.dataset_array import DatasetArray
    >>> from edgaro.balancing.transformer import RandomUnderSampler
    >>> from edgaro.base.base_transformer_array import BaseTransformerArray
    >>> df = DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)])
    >>> params = [ [{'sampling_strategy': 0.98}, {'sampling_strategy': 1}] for _ in range(len(df)) ]
    >>> transformer = RandomUnderSampler()
    >>> array = BaseTransformerArray(transformer, parameters=params)
    >>> array.fit(df)
    >>> array.transform(df)
    """

    def __init__(self, base_transformer: BaseTransformer, parameters: Optional[List[Union[List, Dict[str, Any]]]] = None,
                 transformer_sufix: str = '_transformed_array') -> None:
        self.__base_transformer = base_transformer
        self.__transformers = []
        self.__input_shape = None
        self.__parameters = parameters
        self.__was_fitted = False
        self.transformer_sufix = transformer_sufix

    def __create_new_transformer(self, param: Dict[str, Any]) -> BaseTransformer:
        t = deepcopy(self.__base_transformer)
        t.set_params(**param)
        return t

    def fit(self, dataset: Union[Dataset, DatasetArray]) -> None:
        """
        Fit the transformer.

        Parameters
        ----------
        dataset : Dataset, DatasetArray
            The object to fit BaseTransformerArray on.
        """
        # Single dataset case
        if isinstance(dataset, Dataset):
            self.__input_shape = 1
            if self.__parameters is None:
                self.__transformers = [deepcopy(self.__base_transformer)]
                self.__transformers[0].fit(dataset)
            else:
                self.__transformers = [self.__create_new_transformer(params) for params in self.__parameters]
                for i in range(len(self.__transformers)):
                    self.__transformers[i].fit(dataset)
        # DatasetArray case
        else:
            self.__input_shape = len(dataset)
            if self.__parameters is None:
                self.__transformers = [
                    deepcopy(self.__base_transformer)
                    if isinstance(dataset[i], Dataset)
                    else BaseTransformerArray(deepcopy(self.__base_transformer))
                    for i in range(len(dataset))
                ]
                for i in range(len(self.__transformers)):
                    self.__transformers[i].fit(dataset[i])
            else:
                if not len(dataset) == len(self.__parameters):
                    raise Exception('Not enough parameters were provided!')

                self.__transformers = [deepcopy(BaseTransformerArray(
                    base_transformer=self.__base_transformer,
                    parameters=BaseTransformerArray.__transform_dict_to_lst(self.__parameters[i]) if not isinstance(self.__parameters[i], list) else self.__parameters[i]))
                    for i in range(len(dataset))]

                for i in range(len(dataset)):
                    self.__transformers[i].fit(dataset[i])
        self.__was_fitted = True

    def transform(self, dataset: Union[Dataset, DatasetArray]) -> DatasetArray:
        """
        Transform the object.

        Parameters
        ----------
        dataset : Dataset, DatasetArray
            The object to be transformed.

        Returns
        -------
        DatasetArray
            The transformed object.
        """
        # Single dataset case
        if isinstance(dataset, Dataset):
            if not self.__input_shape == 1:
                raise Exception('DatasetArray was fitted, but single Dataset was provided!')

            tab = []
            for transformator in self.__transformers:
                try:
                    t = transformator.transform(dataset)
                    tab.append(t)
                except (Exception,) as exc:
                    print_unbuffered(f"An Exception occurred while transforming {dataset.name} with "
                                     f"{transformator.__str__()}")
                    print_unbuffered(exc)

            try:
                return DatasetArray(
                    tab,
                    name=dataset.name + self.transformer_sufix
                )
            except (Exception,):
                for i in range(len(tab)):
                    tab[i].name += ('_' + str(i))
                return DatasetArray(
                    tab,
                    name=dataset.name + self.transformer_sufix
                )
        # DatasetArray case
        else:
            if not self.__input_shape == len(dataset):
                raise Exception('Dataset was fitted, but DatasetArray was provided!')

            tab = []
            for i in range(len(dataset)):
                try:
                    t = self.__transformers[i].transform(dataset[i])
                    tab.append(t)
                except (Exception,) as exc:
                    print_unbuffered(f"An Exception occurred while transforming {dataset[i].name} with "
                                     f"{self.__transformers[i].__str__()}")
                    print_unbuffered(exc)

            return DatasetArray(
                tab,
                name=dataset.name + self.transformer_sufix
            )

    def set_params(self, **params) -> None:
        """
        Set params for BaseTransformerArray.

        The parameters should be in a form of a list or a nested list - it depends on whether the object is used
        on a Dataset or a DatasetArray object. For details see Examples section.

        Parameters
        ----------
        params : dict
            The parameters to be set.

        Examples
        ----------
        Example 1

        >>> from test.resources.objects import *
        >>> from edgaro.balancing.transformer import RandomUnderSampler
        >>> df = Dataset(name_1, df_1, target_1)
        >>> transformer = RandomUnderSampler()
        >>> array = BaseTransformerArray(transformer)
        >>> params = {'sampling_strategy': [1, 0.98]}
        >>> array.set_params(**params)
        >>> array.fit(df)
        >>> array.transform(df)

        Example 2

        >>> from test.resources.objects import *
        >>> from edgaro.balancing.transformer import RandomUnderSampler
        >>> df = DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)])
        >>> transformer = RandomUnderSampler()
        >>> array = BaseTransformerArray(transformer)
        >>> params = {'sampling_strategy': [[1, 0.98] for _ in range(len(df))]}
        >>> array.set_params(**params)
        >>> array.fit(df)
        >>> array.transform(df)
        """
        lengths = [len(val) for key, val in params.items()]
        if len(lengths) == 0:
            raise Exception('Parameters were not provided!')

        if not np.alltrue(np.array(lengths) == lengths[0]):
            raise Exception('Parameters do not have the same length!')

        self.__parameters = BaseTransformerArray.__transform_dict_to_lst(params)

        if len(self.__transformers) != 0:
            for i in range(len(self.__transformers)):
                self.__transformers[i].set_params(**self.__parameters[i])

    @staticmethod
    def __transform_dict_to_lst(dct):
        if np.alltrue([not isinstance(val, list) for key, val in dct.items()]):
            return [dct]
        lengths = [len(val) for key, val in dct.items()]
        tmp = []
        for i in range(lengths[0]):
            tmp_dict = {}
            for key in dct:
                tmp_dict[key] = dct[key][i]
            tmp.append(tmp_dict)
        return tmp

    def get_params(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get parameters of BaseTransformerArray.

        Returns
        -------
        list[list, Dict[str, Any]]], optional
            The parameters.
        """
        return self.__parameters

    @property
    def was_fitted(self) -> bool:
        """
        The information whether the BaseTransformerArray was fitted.

        Returns
        -------
        bool
        """
        if self.__was_fitted:
            return True
        elif len(self.__transformers) == 0:
            return False
        else:
            return np.alltrue([
                transformer.was_fitted for transformer in self.__transformers
            ])

    @property
    def parameters(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get parameters of BaseTransformerArray.

        Returns
        -------
        list[list, Dict[str, Any]]], optional
            The parameters.
        """
        return self.__parameters

    @parameters.setter
    def parameters(self, val: Optional[List[Dict[str, Any]]]) -> None:
        if not self.was_fitted:
            self.__parameters = val
        else:
            raise Exception('Parameters were not set since Transformer has already been fitted!')

    @property
    def transformers(self) -> List[Union[BaseTransformer, BaseTransformerArray, List]]:
        """
        BaseTransformer and BaseTransformerArray objects inside this object.

        Returns
        -------
        list[BaseTransformer, BaseTransformerArray, list]
        """
        return self.__transformers

    @transformers.setter
    def transformers(self, val: List[Union[BaseTransformer, BaseTransformerArray, List]]) -> None:
        if not self.was_fitted:
            self.__transformers = val
        else:
            raise Exception('Transformers were not set since Transformer has already been fitted!')

    @property
    def base_transformer(self) -> BaseTransformer:
        """
        BaseTransformer of this object.

        Returns
        -------
        BaseTransformer
        """
        return self.__base_transformer

    @base_transformer.setter
    def base_transformer(self, val: BaseTransformer):
        if not self.was_fitted:
            self.__transformers = val
        else:
            raise Exception('Base transformers were not set since Transformer has already been fitted!')

    def __len__(self) -> int:
        return len(self.__transformers)

    def __getitem__(self, key: Union[int, List[int]]) -> Optional[
                    Union[BaseTransformer, BaseTransformerArray, List]]:
        if isinstance(key, list):
            out = [self.__getitem__(k) for k in key]
            out = [o for o in out if o is not None]
            if len(out) == 0:
                return None
            else:
                return out
        elif isinstance(key, int):
            if key <= len(self.__transformers):
                return self.__transformers[key]
        return None

    def __str__(self) -> str:
        return f"BaseTransformerArray{(' ' + self.__class__.__name__) if self.__class__.__name__ != 'BaseTransformerArray' else ''} with {len(self.transformers)} transformers"

    def __repr__(self) -> str:
        return f"<BaseTransformerArray{(' ' + self.__class__.__name__) if self.__class__.__name__ != 'BaseTransformerArray' else ''} with {len(self.transformers)} transformers>"
