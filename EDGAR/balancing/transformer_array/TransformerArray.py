from typing import List, Dict, Optional, Any, Union
from copy import deepcopy
import numpy as np
from EDGAR.data.dataset.Dataset import Dataset
from EDGAR.data.dataset_array.DatasetArray import DatasetArray
from EDGAR.balancing.transformer.Transformer import Transformer


# TODO:
# DatasetArray of DatasetArray implementation

class TransformerArray(Transformer):
    def __init__(self, base_transfer: Transformer, parameters: Optional[List[Dict[str, Any]]] = None,
                 name_sufix: Union[str, List[str]] = '_transformed'):
        super().__init__(name_sufix='')
        self.__base_transformer = base_transfer
        self.__transformers = []
        self.__input_array = None
        self.__parameters = parameters

        self.__name_sufix = None
        self.set_name_sufix(name_sufix)

    def set_name_sufix(self, name_sufix: Union[str, List[str]]):
        if self.__parameters is None:
            if not (isinstance(name_sufix, str) or (isinstance(name_sufix, list) and len(name_sufix) == 1)):
                raise Exception('Parameter name_sufix has invalid length!')
            if isinstance(name_sufix, str):
                self.__name_sufix = [self.__name_sufix]
        else:
            if not (isinstance(self.__name_sufix, str) or (
                    isinstance(self.__name_sufix, list) and len(self.__name_sufix) == len(self.__parameters))):
                raise Exception('Parameter name_sufix has invalid length!')
            if isinstance(self.__name_sufix, str):
                if len(self.__parameters) == 1:
                    self.__name_sufix = [self.__name_sufix]
                else:
                    self.__name_sufix = [self.__name_sufix + self.__name_sufix + '_' + str(i) for i in
                                         range(len(self.__name_sufix))]

    def __create_new_transformer(self, param):
        t = deepcopy(self.__base_transformer)
        t.set_params(**param)
        return t

    def fit(self, dataset: Union[Dataset, DatasetArray]):
        # Single dataset case
        if isinstance(dataset, Dataset):
            self.__input_array = False
            if self.__parameters is None:
                self.__transformers = [self.__base_transformer]
            else:
                self.__transformers = [self.__create_new_transformer(params) for params in self.__parameters]
                for i in range(len(self.__transformers)):
                    self.__transformers[i].set_name_sufix(self.__name_sufix[i])
                    self.__transformers[i].fit(dataset)
        # DatasetArray case
        else:
            self.__input_array = True
            if self.__parameters is None:
                self.__transformers = [
                    [deepcopy(self.__base_transformer)] for _ in range(len(dataset))
                    # dodać tutaj stworzenie TransformerArray jeśli któryś z DatasetArray jest też DatasetArray
                ]
            else:
                if not len(dataset) == len(self.__parameters):
                    raise Exception('Not enough parameters were provided!')
                tab = [self.__create_new_transformer(params) for params in self.__parameters]
                self.__transformers = [deepcopy(tab) for _ in range(
                    len(dataset))]  # dodać tutaj stworzenie TransformerArray jeśli któryś z DatasetArray jest też DatasetArray
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
                self.__transformers[i].set_params(**tmp[i])

    def get_params(self):
        return self.__parameters

    def get_imblearn_transformer(self):
        return self.__transformers
