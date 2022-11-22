from __future__ import annotations

import numpy as np

from typing import List, Dict, Optional, Any, Union

from EDGAR.data.dataset import Dataset
from EDGAR.data.dataset_array import DatasetArray
from EDGAR.base.base_transformer_array import BaseTransformerArray
from EDGAR.balancing.transformer import Transformer, RandomUnderSampler, RandomOverSampler, SMOTE
from EDGAR.base.utils import print_unbuffered


class TransformerArray(BaseTransformerArray):
    def __init__(self, base_transformer: Transformer, parameters: Optional[List[Dict[str, Any]]] = None,
                 keep_original_dataset: bool = False, dataset_suffixes: Union[str, List[str]] = '_transformed',
                 result_array_sufix: str = '_transformed_array', allow_dataset_array_sufix_change: bool = True,
                 verbose: bool = False) -> None:
        super().__init__(base_transformer=base_transformer, parameters=parameters, transformer_sufix=result_array_sufix)
        self.__dataset_suffixes = None
        self.set_dataset_suffixes(dataset_suffixes)
        self.keep_original_dataset = keep_original_dataset
        self.allow_dataset_array_sufix_change = allow_dataset_array_sufix_change
        self.verbose = verbose

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
        if self.__dataset_suffixes is not None and len(self.__dataset_suffixes) == 1:
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

        if isinstance(self.__dataset_suffixes, list) and len(self.__dataset_suffixes) == 1 and isinstance(
                self.__dataset_suffixes[0], str):
            if self.allow_dataset_array_sufix_change:
                self.transformer_sufix = self.__dataset_suffixes[0]

        if self.verbose:
            print_unbuffered(f'TransformerArray {self.__repr__()} was fitted with {dataset.name}')

    def transform(self, dataset: Union[Dataset, DatasetArray]) -> Union[Dataset, DatasetArray]:
        out = super().transform(dataset=dataset)
        if self.keep_original_dataset:
            out.append(dataset)

        if self.verbose:
            print_unbuffered(f'TransformerArray {self.__repr__()} transformed with {dataset.name}')

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
    def base_transformer(self, val: Union[Transformer, TransformerArray, List[Any]]) -> None:
        super().base_transformer = val

    def __str__(self) -> str:
        return f"TransformerArray{(' ' + self.__class__.__name__) if self.__class__.__name__ != 'TransformerArray' else ''} with {len(self.transformers)} transformers"

    def __repr__(self) -> str:
        return f"<TransformerArray{(' ' + self.__class__.__name__) if self.__class__.__name__ != 'TransformerArray' else ''} with {len(self.transformers)} transformers>"


class AutomaticTransformerArray(Transformer):
    def __init__(self, keep_original_dataset: bool = False, result_array_sufix: str = '_automatic_transformed_array',
                 n_per_method: int = 5, random_state: Optional[int] = None, IR_round_precision: int = 2,
                 min_samples_to_modify: Optional[int] = None, verbose: bool = False) -> None:
        self.keep_original_dataset = keep_original_dataset
        self.n_per_method = n_per_method
        self.random_state = random_state
        self.result_array_sufix = result_array_sufix
        self.IR_round_precision = IR_round_precision
        self.__transformers = None
        self.__was_fitted = False
        self.min_samples_to_modify = min_samples_to_modify
        self.verbose = verbose

        super(Transformer, self).__init__()

    def fit(self, dataset: Dataset) -> None:
        IR = dataset.imbalance_ratio
        n_per_method = self.n_per_method
        IR_range = IR - 1
        IR_step = IR_range / n_per_method

        if self.min_samples_to_modify is not None:
            _, counts = np.unique(dataset.target, return_counts=True)
            n_rows_minority = min(counts)
            n_rows_majority = max(counts)

            n_rows_modified_under = IR_step * n_rows_minority
            n_rows_modified_over = (n_rows_majority * n_rows_minority) / (
                        n_rows_majority - IR_step * n_rows_minority) - n_rows_minority
            min_n_rows_modified = min(n_rows_modified_under, n_rows_modified_over)

            if min_n_rows_modified < self.min_samples_to_modify:
                IR_step_under = self.min_samples_to_modify / n_rows_minority
                IR_step_over = n_rows_majority * (
                            1 / n_rows_minority - 1 / (n_rows_minority + self.min_samples_to_modify))

                IR_step = max(IR_step_under, IR_step_over)
                n_per_method = int((IR - 1) // IR_step + 1)

                if round(IR_step, 3) == round(IR_range, 3):
                    n_per_method = 1

        IR_values = [
            IR - i * IR_step if IR - i * IR_step >= 1
            else 1
            for i in range(1, int(n_per_method) + 1)
        ]

        random_under_sampler_array = TransformerArray(
            RandomUnderSampler(random_state=self.random_state),
            parameters=[{'IR': ir} for ir in IR_values],
            dataset_suffixes=[f'_transformed_under_sampler_IR={round(ir, self.IR_round_precision)}' for ir in
                              IR_values],
            result_array_sufix='_RANDOM_UNDERSAMPLING'
        )
        random_under_sampler_array.fit(dataset)

        random_over_sampler_array = TransformerArray(
            RandomOverSampler(random_state=self.random_state),
            parameters=[{'IR': ir} for ir in IR_values],
            dataset_suffixes=[f'_transformed_over_sampler_IR={round(ir, self.IR_round_precision)}' for ir in IR_values],
            result_array_sufix='_RANDOM_OVERSAMPLING'
        )
        random_over_sampler_array.fit(dataset)

        smote_array = TransformerArray(
            SMOTE(random_state=self.random_state),
            parameters=[{'IR': ir} for ir in IR_values],
            dataset_suffixes=[f'_transformed_smote_IR={round(ir, self.IR_round_precision)}' for ir in IR_values],
            result_array_sufix='_SMOTE'
        )
        smote_array.fit(dataset)

        self.__transformers = [random_under_sampler_array, random_over_sampler_array, smote_array]
        self.__was_fitted = True

        if self.verbose:
            print_unbuffered(f'AutomaticTransformerArray {self.__repr__()} was fitted with {dataset.name}')

    def transform(self, dataset: Dataset) -> DatasetArray:
        if self.__transformers is None:
            return DatasetArray([])
        else:
            out = DatasetArray(
                [t.transform(dataset) for t in self.__transformers],
                name=dataset.name + self.result_array_sufix
            )
            if self.keep_original_dataset:
                out.append(dataset)

            if self.verbose:
                print_unbuffered(f'AutomaticTransformerArray {self.__repr__()} transformed {dataset.name}')

            return out

    def set_dataset_suffixes(self, name_sufix: Union[str, List[Union[str, List[str]]]]) -> None:
        if isinstance(self.__transformers, list) and np.all(
                [isinstance(t, TransformerArray) for t in self.__transformers]):
            for t in self.__transformers:
                t.set_dataset_suffixes(name_sufix)

    def get_dataset_suffixes(self) -> Optional[Union[str, List[str]]]:
        if isinstance(self.__transformers, list) and np.all(
                [isinstance(t, TransformerArray) for t in self.__transformers]):
            return [t.get_dataset_suffixes() for t in self.__transformers]
        else:
            return []

    def set_params(self, **params) -> None:
        if isinstance(self.__transformers, list) and np.all(
                [isinstance(t, TransformerArray) for t in self.__transformers]):
            for i in range(len(self.__transformers)):
                tmp = {}
                for key, val in params.items():
                    tmp[key] = val[i]
                self.__transformers[i].set_params(**tmp)

    def get_params(self) -> List[Union[Dict, List[Any]]]:
        if isinstance(self.__transformers, list) and np.all(
                [isinstance(t, TransformerArray) for t in self.__transformers]):
            return [t.parameters for t in self.__transformers]
        else:
            return []

    @property
    def was_fitted(self) -> bool:
        return self.__was_fitted

    @property
    def transformers(self) -> List[Union[Transformer, TransformerArray, List[Any]]]:
        if self.__transformers is None:
            return []
        else:
            return [t.transformers for t in self.__transformers]

    @transformers.setter
    def transformers(self, val: List[Union[Transformer, TransformerArray, List[Any]]]) -> None:
        if not self.was_fitted:
            self.__transformers = val
        else:
            raise Exception('Transformers were not set since Transformer has already been fitted!')

    @property
    def base_transformer(self) -> Union[Transformer, TransformerArray, List[Any]]:
        if self.__transformers is None:
            return []
        else:
            return [t.base_transformer for t in self.__transformers]

    @base_transformer.setter
    def base_transformer(self, val: Union[Transformer, TransformerArray, List[Any]]) -> None:
        raise Exception('Base Transformer can\'t be set for AutomaticTransformerArray')

    def _fit(self, dataset: Dataset) -> None:
        pass

    def _transform(self, dataset: Dataset) -> Dataset:
        return Dataset('', None, None)

    def __len__(self) -> int:
        if self.__transformers is None:
            return 0
        else:
            return len(self.__transformers)

    def __getitem__(self, key: Union[int, List[int]]) -> Optional[
                    Union[Optional[Transformer], TransformerArray, List[Union[Optional[Transformer], TransformerArray]]]]:
        if self.__transformers is None:
            return None
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

    def __str__(self) -> str:
        out = f"AutomaticTransformerArray transformers" + "\n"
        if self.__transformers is not None:
            for t in self.__transformers:
                out += t.__str__()
                out += '\n'
        return out

    def __repr__(self) -> str:
        return "<AutomaticTransformerArray transformers>"
