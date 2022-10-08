from typing import List, Dict, Optional, Any, Union
from copy import deepcopy
import numpy as np
from EDGAR.data.Dataset import Dataset
from EDGAR.data.DatasetArray import DatasetArray
from EDGAR.base.BaseTransformer import BaseTransformer


# TODO:
# DatasetArray of DatasetArray implementation

class BaseTransformerArray:
    def __init__(self, base_transfer: BaseTransformer, parameters: Optional[List[Dict[str, Any]]] = None):
        self.__base_transformer = base_transfer
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
                    [deepcopy(self.__base_transformer)] for _ in range(len(dataset))
                    # dodać tutaj stworzenie TransformerArray jeśli któryś z DatasetArray jest też DatasetArray
                ]
                for i in range(len(self.__transformers)):
                    self.__transformers[i][0].fit(dataset[i])
            else:
                if not len(dataset) == len(self.__parameters):
                    raise Exception('Not enough parameters were provided!')
                tab = [self.__create_new_transformer(params) for params in self.__parameters]
                self.__transformers = [deepcopy(tab) for _ in range(len(dataset))]  # dodać tutaj stworzenie
                # TransformerArray jeśli któryś z DatasetArray jest też DatasetArray
                for i in range(len(dataset)):
                    for j in range(len(self.__transformers[i])):
                        self.__transformers[i][j].fit(dataset[i])

    def transform(self, dataset: Union[Dataset, DatasetArray]):
        # Single dataset case
        if isinstance(dataset, Dataset):
            if self.__input_array:
                raise Exception('DatasetArray was fitted, but single Dataset was provided!')
            return DatasetArray(
                [transformator.transform(dataset) for transformator in self.__transformers],
                name=dataset.name + '_array'
            )
        # DatasetArray case
        else:
            if not self.__input_array:
                raise Exception('Dataset was fitted, but DatasetArray was provided!')
            return [
                DatasetArray(
                    [transformator.transform(dataset[i]) for transformator in self.__transformers[i]],
                    name=dataset.name + '_array'
                ) for i in range(len(dataset))
            ]

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
                for j in range(len(self.__transformers[i])):
                    self.__transformers[i][j].set_params(**tmp[j])

    def get_params(self):
        return self.__parameters

    def get_transformers(self):
        return self.__transformers
