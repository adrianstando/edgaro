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
        if self.get_params() is None:
            if isinstance(name_sufix, str):
                self.__name_sufix = [name_sufix]
            elif isinstance(name_sufix, list):
                self.__name_sufix = name_sufix
            else:
                raise Exception('Wrong sufix names!')
        else:
            if isinstance(name_sufix, str):
                if len(self.get_params()) == 1:
                    self.__name_sufix = [name_sufix]
                else:
                    self.__name_sufix = [name_sufix + '_' + str(i) for i in range(len(name_sufix))]
            elif isinstance(name_sufix, list) and len(name_sufix) == len(self.get_params()):
                self.__name_sufix = name_sufix
            else:
                raise Exception('Parameter name_sufix has invalid length!')

    def get_name_sufix(self):
        return self.__name_sufix

    def fit(self, dataset: Union[Dataset, DatasetArray]):
        super().fit(dataset)

        # Setting suffixes
        if isinstance(dataset, Dataset):
            if self.get_params() is None:
                self.get_transformers()[0].set_name_sufix(self.__name_sufix[0])
            elif len(self.__name_sufix) == len(self.get_transformers()):
                for i in range(len(self.get_transformers())):
                    self.get_transformers()[i].set_name_sufix(self.__name_sufix[i])
            else:
                raise Exception('Wrong length of name_sufix!')
        else:
            if self.get_params() is None:
                for i in range(len(self.get_transformers())):
                    if len(self.__name_sufix) == 1:
                        self.get_transformers()[i][0].set_name_sufix(self.__name_sufix[0])
                    else:
                        self.get_transformers()[i][0].set_name_sufix(self.__name_sufix[i])
            else:
                for i in range(len(dataset)):
                    for j in range(len(self.get_transformers()[i])):
                        if len(self.__name_sufix) == 1:
                            self.get_transformers()[i][j].set_name_sufix(self.__name_sufix[0] + '_' + str(j))
                        elif len(self.__name_sufix) == len(self.get_transformers()):
                            self.get_transformers()[i][j].set_name_sufix(self.__name_sufix[i])
                        else:
                            raise Exception('Wrong length of name_sufix!')
