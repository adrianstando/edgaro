from __future__ import annotations

from typing import List, Dict, Optional, Any, Union
from EDGAR.data.Dataset import Dataset
from EDGAR.data.DatasetArray import DatasetArray
from EDGAR.balancing.Transformer import Transformer
from EDGAR.base.BaseTransformerArray import BaseTransformerArray


class TransformerArray(BaseTransformerArray):
    def __init__(self, base_transformer: Transformer, parameters: Optional[List[Dict[str, Any]]] = None,
                 keep_original_dataset: bool = False, dataset_suffixes: Union[str, List[str]] = '_transformed',
                 result_array_sufix: str = '_transformed_array', allow_dataset_array_sufix_change: bool = True) -> None:
        super().__init__(base_transformer=base_transformer, parameters=parameters, transformer_sufix=result_array_sufix)
        self.__dataset_suffixes = None
        self.set_dataset_suffixes(dataset_suffixes)
        self.keep_original_dataset = keep_original_dataset
        self.allow_dataset_array_sufix_change = allow_dataset_array_sufix_change

    def set_dataset_suffixes(self, name_sufix: Union[str, List[str]]) -> None:
        params = self.get_params()
        length_params = len(params) if params is not None else 0
        set_names = True if len(self.transformers) > 0 else False
        transformers = self.transformers

        if params is None:
            if isinstance(name_sufix, str):
                self.__dataset_suffixes = [name_sufix]
                if set_names:
                    transformers[0].set_dataset_suffixes(name_sufix)

            elif isinstance(name_sufix, list) and len(name_sufix) == 1 and isinstance(name_sufix[0], str):
                self.__dataset_suffixes = name_sufix
                if set_names:
                    transformers[0].set_dataset_suffixes(name_sufix)

            elif isinstance(name_sufix, list) and (length_params == 0 or len(name_sufix) == length_params):
                self.__dataset_suffixes = name_sufix
                if set_names:
                    for i in range(len(transformers)):
                        transformers[i].set_dataset_suffixes(name_sufix[i])

            else:
                raise Exception('Wrong sufix names!')

        else:
            if isinstance(name_sufix, list) and len(name_sufix) == 1 and isinstance(name_sufix[0], str):
                name_sufix = name_sufix[0]

            if isinstance(name_sufix, str):
                if length_params == 1:
                    self.__dataset_suffixes = [name_sufix]
                    if set_names:
                        transformers[0].set_dataset_suffixes(name_sufix)

                else:
                    self.__dataset_suffixes = [name_sufix + '_' + str(i) for i in range(length_params)]
                    if set_names:
                        for i in range(len(transformers)):
                            transformers[i].set_dataset_suffixes(self.__dataset_suffixes[i])

            elif isinstance(name_sufix, list) and len(name_sufix) == length_params:
                self.__dataset_suffixes = name_sufix
                if set_names:
                    for i in range(len(transformers)):
                        transformers[i].set_dataset_suffixes(self.__dataset_suffixes[i])

            else:
                raise Exception('Parameter dataset_suffixes has invalid length!')

    def get_dataset_suffixes(self) -> Optional[Union[str, List[str]]]:
        return self.__dataset_suffixes

    def set_params(self, **params) -> None:
        super().set_params(**params)
        if len(self.__dataset_suffixes) == 1:
            self.set_dataset_suffixes(self.__dataset_suffixes[0])

    def fit(self, dataset: Union[Dataset, DatasetArray]) -> None:
        super().fit(dataset)
        self.__fix_classes()

        # Setting suffixes
        params = self.get_params()

        # Dataset case
        if isinstance(dataset, Dataset):
            if params is None:
                self.transformers[0].set_dataset_suffixes(self.__dataset_suffixes[0])
            elif len(self.__dataset_suffixes) == len(self.transformers):
                for i in range(len(self.transformers)):
                    self.transformers[i].set_dataset_suffixes(self.__dataset_suffixes[i])
            else:
                raise Exception('Wrong length of dataset_suffixes!')
        # DatasetArray case
        else:
            if params is None:
                for i in range(len(self.transformers)):
                    if len(self.__dataset_suffixes) == 1:
                        self.transformers[i].set_dataset_suffixes(self.__dataset_suffixes[0])
                    else:
                        self.transformers[i].set_dataset_suffixes(self.__dataset_suffixes[i])
            else:
                for i in range(len(dataset)):
                    for j in range(len(self.transformers[i])):
                        if len(self.__dataset_suffixes) == 1:
                            self.transformers[i][j].set_dataset_suffixes(self.__dataset_suffixes[0] + '_' + str(j))
                        elif len(self.__dataset_suffixes) == len(self.transformers):
                            self.transformers[i][j].set_dataset_suffixes(self.__dataset_suffixes[j])
                        else:
                            raise Exception('Wrong length of dataset_suffixes!')

        if isinstance(self.__dataset_suffixes, list) and len(self.__dataset_suffixes) == 1 and isinstance(self.__dataset_suffixes[0], str):
            if self.allow_dataset_array_sufix_change:
                self.transformer_sufix = self.__dataset_suffixes[0]

    def transform(self, dataset: Union[Dataset, DatasetArray]) -> Union[Dataset, DatasetArray]:
        out = super().transform(dataset=dataset)
        if self.keep_original_dataset:
            out.append(dataset)
        return out

    def __fix_classes(self) -> None:
        for i in range(len(self.transformers)):
            self.transformers[i] = self.__base_transformer_array_to_balancing_transformer_array(
                self.transformers[i])

    def __base_transformer_array_to_balancing_transformer_array(self, base: Any) -> Any:
        if isinstance(base, Transformer):
            return base
        elif not isinstance(base, TransformerArray):
            out = TransformerArray(base_transformer=self.base_transformer)
            out.__class__ = self.__class__
            for key, val in base.__dict__.items():
                out.__dict__[key] = val

            return out
        else:
            return base

    @property
    def transformers(self) -> List[Union[Transformer, TransformerArray, List[Any]]]:
        return super().transformers

    @transformers.setter
    def transformers(self, val: List[Union[Transformer, TransformerArray, List[Any]]]) -> None:
        super().transformers = val

    @property
    def base_transformer(self) -> Union[Transformer, TransformerArray, List[Any]]:
        return super().base_transformer

    @base_transformer.setter
    def base_transformer(self, val: Union[Transformer, TransformerArray, List[Any]]):
        super().base_transformer = val

    def __str__(self) -> str:
        return f"TransformerArray{(' ' + self.__class__.__name__) if self.__class__.__name__ != 'TransformerArray' else ''} with {len(self.transformers)} transformers"

    def __repr__(self) -> str:
        return f"<TransformerArray{(' ' + self.__class__.__name__) if self.__class__.__name__ != 'TransformerArray' else ''} with {len(self.transformers)} transformers>"
