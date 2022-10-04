from typing import List, Dict, Optional, Any, Union
from EDGAR.data.Dataset import Dataset
from EDGAR.data.DatasetArray import DatasetArray
from EDGAR.balancing.Transformer import Transformer
from EDGAR.base.BaseTransformerArray import BaseTransformerArray


class TransformerArray(BaseTransformerArray):
    def __init__(self, base_transfer: Transformer, parameters: Optional[List[Dict[str, Any]]] = None,
                 name_sufix: Union[str, List[str]] = '_transformed'):
        super().__init__(base_transfer=base_transfer, parameters=parameters)
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

    def fit(self, dataset: Union[Dataset, DatasetArray]):
        super().fit(dataset)

        # Setting suffixes
        if isinstance(dataset, Dataset):
            if self.__parameters is None:
                self.__transformers[0].set_name_sufix(self.__name_sufix[0])
            else:
                for i in range(len(self.__transformers)):
                    self.__transformers[i].set_name_sufix(self.__name_sufix[i])
        else:
            if self.__parameters is None:
                for i in range(len(self.__transformers)):
                    self.__transformers[i][0].set_name_sufix(self.__name_sufix[i])
            else:
                for i in range(len(dataset)):
                    for j in range(len(self.__transformers[i])):
                        self.__transformers[i][j].set_name_sufix(self.__name_sufix[i])


