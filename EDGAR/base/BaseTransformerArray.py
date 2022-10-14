from typing import List, Dict, Optional, Any, Union
from copy import deepcopy
import numpy as np
from EDGAR.data.Dataset import Dataset
from EDGAR.data.DatasetArray import DatasetArray
from EDGAR.base.BaseTransformer import BaseTransformer


class BaseTransformerArray:
    def __init__(self, base_transformer: BaseTransformer, parameters: Optional[List[Dict[str, Any]]] = None):
        self.__base_transformer = base_transformer
        self.__transformers = []
        self.__input_array = None
        self.__parameters = parameters

    def __create_new_transformer(self, param):
        t = deepcopy(self.__base_transformer)
        t.set_params(**param)
        return t

    def fit(self, dataset: Union[Dataset, DatasetArray]):
        # Single dataset case
        if isinstance(dataset, Dataset):
            self.__input_array = False
            if self.__parameters is None:
                self.__transformers = [deepcopy(self.__base_transformer)]
                self.__transformers[0].fit(dataset)
            else:
                self.__transformers = [self.__create_new_transformer(params) for params in self.__parameters]
                for i in range(len(self.__transformers)):
                    self.__transformers[i].fit(dataset)
        # DatasetArray case
        else:
            self.__input_array = True
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

    def transform(self, dataset: Union[Dataset, DatasetArray]):
        # Single dataset case
        if isinstance(dataset, Dataset):
            if self.__input_array:
                raise Exception('DatasetArray was fitted, but single Dataset was provided!')
            return DatasetArray(
                [transformator.transform(dataset) for transformator in self.__transformers],
                name=dataset.name + '_transformed_array'
            )
        # DatasetArray case
        else:
            if not self.__input_array:
                raise Exception('Dataset was fitted, but DatasetArray was provided!')
            return DatasetArray(
                [self.__transformers[i].transform(dataset[i]) for i in range(len(dataset))],
                name=dataset.name + '_transformed_array'
            )

    """
    Each parameter has to be a list!!
    """

    def set_params(self, **params):
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

    def get_params(self):
        return self.__parameters

    def get_transformers(self):
        return self.__transformers

    def get_base_transformer(self):
        return self.__base_transformer

    def __len__(self):
        return len(self.__transformers)

    def __getitem__(self, key: Union[int, List[int]]):
        if isinstance(key, list):
            out = [self.__getitem__(k) for k in key]
            if len(out) == 0:
                return None
            else:
                return out
        elif isinstance(key, int):
            if key <= len(self.__transformers):
                return self.__transformers[key]
        return None
