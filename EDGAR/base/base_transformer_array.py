from __future__ import annotations

import numpy as np

from typing import List, Dict, Optional, Any, Union
from copy import deepcopy

from EDGAR.data.dataset import Dataset
from EDGAR.data.dataset_array import DatasetArray
from EDGAR.base.base_transformer import BaseTransformer


class BaseTransformerArray:
    def __init__(self, base_transformer: BaseTransformer, parameters: Optional[List[Dict[str, Any]]] = None,
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

                tab = BaseTransformerArray(base_transformer=self.__base_transformer, parameters=self.__parameters)
                self.__transformers = [deepcopy(tab) for _ in range(len(dataset))]

                for i in range(len(dataset)):
                    self.__transformers[i].fit(dataset[i])
        self.__was_fitted = True

    def transform(self, dataset: Union[Dataset, DatasetArray]) -> DatasetArray:
        # Single dataset case
        if isinstance(dataset, Dataset):
            if not self.__input_shape == 1:
                raise Exception('DatasetArray was fitted, but single Dataset was provided!')
            return DatasetArray(
                [transformator.transform(dataset) for transformator in self.__transformers],
                name=dataset.name + self.transformer_sufix
            )
        # DatasetArray case
        else:
            if not self.__input_shape == len(dataset):
                raise Exception('Dataset was fitted, but DatasetArray was provided!')
            return DatasetArray(
                [self.__transformers[i].transform(dataset[i]) for i in range(len(dataset))],
                name=dataset.name + self.transformer_sufix
            )

    """
    Each parameter has to be a list!!
    """

    def set_params(self, **params) -> None:
        lengths = [len(val) for key, val in params.items()]
        if len(lengths) == 0:
            raise Exception('Parameters were not provided!')

        if not np.alltrue(np.array(lengths) == lengths[0]):
            raise Exception('Parameters do not have the same length!')

        tmp = []
        for i in range(lengths[0]):
            tmp_dict = {}
            for key in params:
                tmp_dict[key] = params[key][i]
            tmp.append(tmp_dict)

        self.__parameters = tmp

        if len(self.__transformers) != 0:
            for i in range(len(self.__transformers)):
                self.__transformers[i].set_params(**tmp[i])

    def get_params(self) -> Optional[List[Dict[str, Any]]]:
        return self.__parameters

    @property
    def was_fitted(self) -> bool:
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
        return self.__parameters

    @parameters.setter
    def parameters(self, val: Optional[List[Dict[str, Any]]]) -> None:
        if not self.was_fitted:
            self.__parameters = val
        else:
            raise Exception('Parameters were not set since Transformer has already been fitted!')

    @property
    def transformers(self) -> List[Union[BaseTransformer, BaseTransformerArray, List[Any]]]:
        return self.__transformers

    @transformers.setter
    def transformers(self, val: List[Union[BaseTransformer, BaseTransformerArray, List[Any]]]) -> None:
        if not self.was_fitted:
            self.__transformers = val
        else:
            raise Exception('Transformers were not set since Transformer has already been fitted!')

    @property
    def base_transformer(self) -> BaseTransformer:
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
                    Union[BaseTransformer, BaseTransformerArray, List[Any]]]:
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
        return f"BaseTransformerArray {self.__class__.__name__ if self.__class__.__name__ != 'BaseTransformerArray' else ''} with {len(self.transformers)} transformers"

    def __repr__(self) -> str:
        return f"<BaseTransformerArray {self.__class__.__name__ if self.__class__.__name__ != 'BaseTransformerArray' else ''} with {len(self.transformers)} transformers>"
