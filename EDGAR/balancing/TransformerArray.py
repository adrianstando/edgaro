from typing import List, Dict, Optional, Any, Union
from EDGAR.data.Dataset import Dataset
from EDGAR.data.DatasetArray import DatasetArray
from EDGAR.balancing.Transformer import Transformer
from EDGAR.base.BaseTransformerArray import BaseTransformerArray


class TransformerArray(BaseTransformerArray):
    def __init__(self, base_transformer: Transformer, parameters: Optional[List[Dict[str, Any]]] = None,
                 name_sufix: Union[str, List[str]] = '_transformed'):
        super().__init__(base_transformer=base_transformer, parameters=parameters)
        self.__name_sufix = None
        self.set_name_sufix(name_sufix)

    def set_name_sufix(self, name_sufix: Union[str, List[str]]):
        params = self.get_params()
        length_params = len(params) if params is not None else 0
        set_names = True if len(self.get_transformers()) > 0 else False
        transformers = self.get_transformers()

        if params is None:
            if isinstance(name_sufix, str):
                self.__name_sufix = [name_sufix]
                if set_names:
                    transformers[0].set_name_sufix(name_sufix)

            elif isinstance(name_sufix, list) and len(name_sufix) == 1 and isinstance(name_sufix[0], str):
                self.__name_sufix = name_sufix
                if set_names:
                    transformers[0].set_name_sufix(name_sufix)

            elif isinstance(name_sufix, list) and (length_params == 0 or len(name_sufix) == length_params):
                self.__name_sufix = name_sufix
                if set_names:
                    for i in range(len(transformers)):
                        transformers[i].set_name_sufix(name_sufix[i])

            else:
                raise Exception('Wrong sufix names!')

        else:
            if isinstance(name_sufix, list) and len(name_sufix) == 1 and isinstance(name_sufix[0], str):
                name_sufix = name_sufix[0]

            if isinstance(name_sufix, str):
                if length_params == 1:
                    self.__name_sufix = [name_sufix]
                    if set_names:
                        transformers[0].set_name_sufix(name_sufix)

                else:
                    self.__name_sufix = [name_sufix + '_' + str(i) for i in range(length_params)]
                    if set_names:
                        for i in range(len(transformers)):
                            transformers[i].set_name_sufix(self.__name_sufix[i])

            elif isinstance(name_sufix, list) and len(name_sufix) == length_params:
                self.__name_sufix = name_sufix
                if set_names:
                    for i in range(len(transformers)):
                        transformers[i].set_name_sufix(self.__name_sufix[i])

            else:
                raise Exception('Parameter name_sufix has invalid length!')

    def get_name_sufix(self):
        return self.__name_sufix

    def fit(self, dataset: Union[Dataset, DatasetArray]):
        super().fit(dataset)

        # Setting suffixes
        params = self.get_params()

        # Dataset case
        if isinstance(dataset, Dataset):
            if params is None:
                self.get_transformers()[0].set_name_sufix(self.__name_sufix[0])
            elif len(self.__name_sufix) == len(self.get_transformers()):
                for i in range(len(self.get_transformers())):
                    self.get_transformers()[i].set_name_sufix(self.__name_sufix[i])
            else:
                raise Exception('Wrong length of name_sufix!')
        # DatasetArray case
        else:
            if params is None:
                for i in range(len(self.get_transformers())):
                    if len(self.__name_sufix) == 1:
                        self.get_transformers()[i] = self.__base_transformer_array_to_balancing_transformer_array(
                            self.get_transformers()[i])
                        self.get_transformers()[i].set_name_sufix(self.__name_sufix[0])
                    else:
                        self.get_transformers()[i] = self.__base_transformer_array_to_balancing_transformer_array(
                            self.get_transformers()[i])
                        self.get_transformers()[i].set_name_sufix(self.__name_sufix[i])
            else:
                for i in range(len(dataset)):
                    self.get_transformers()[i] = self.__base_transformer_array_to_balancing_transformer_array(
                        self.get_transformers()[i])
                    for j in range(len(self.get_transformers()[i])):
                        if len(self.__name_sufix) == 1:
                            self.get_transformers()[i][j].set_name_sufix(self.__name_sufix[0] + '_' + str(j))
                        elif len(self.__name_sufix) == len(self.get_transformers()):
                            self.get_transformers()[i][j].set_name_sufix(self.__name_sufix[j])
                        else:
                            raise Exception('Wrong length of name_sufix!')

    def __base_transformer_array_to_balancing_transformer_array(self, base):
        if isinstance(base, Transformer):
            return base
        elif not isinstance(base, TransformerArray):
            out = TransformerArray(base_transformer=self.get_base_transformer())
            out.__class__ = self.__class__
            for key, val in base.__dict__.items():
                out.__dict__[key] = val

            return out
        else:
            return base
