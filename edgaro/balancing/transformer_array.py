from __future__ import annotations

from typing import List, Dict, Optional, Any, Union

from edgaro.data.dataset import Dataset
from edgaro.data.dataset_array import DatasetArray
from edgaro.base.base_transformer_array import BaseTransformerArray
from edgaro.balancing.transformer import Transformer
from edgaro.base.utils import print_unbuffered


class TransformerArray(BaseTransformerArray):
    """
    Create a class to apply Transformer transformation with more than one set of parameters and/or
    to each of the Dataset objects in DatasetArray.

    Note: If you use NestedAutomaticTransformer (or children class) as a parameter to TransformerArray, it is advisable
    to pass parameter `set_suffixes=False` in TransformerArray object. Otherwise, the suffixes will be distorted.

    Parameters
    ----------
    base_transformer : Transformer
        The object defining the transformation procedure.
    parameters : list[list, Dict[str, Any]]], optional
        The list of parameters for base_transformer. If the object is used for a DatasetArray object,
        the parameter list should be nested. For details, see Examples section.
    keep_original_dataset : bool, default=False
        Keep the original Dataset after transformations or not.
    dataset_suffixes : list, str, default='_transformed'
        Suffixes to be set to a transformed objects.
    result_array_sufix : str, default='_transformed_array'
        Suffix of the main transformed DatasetArray object.
    allow_dataset_array_sufix_change : bool, default=True
        Allow changing passed value of `result_array_sufix` according to `dataset_suffixes`.
    verbose : bool, default=False
        Print messages during calculations.
    set_suffixes : str, default=True
        Information whether suffixes for sub-Transformers should be set.

    Examples
    ----------
    Example 1

    >>> from test.resources.objects import *
    >>> from edgaro.data.dataset import Dataset
    >>> from edgaro.data.dataset_array import DatasetArray
    >>> from edgaro.balancing.transformer import RandomUnderSampler
    >>> from edgaro.balancing.transformer_array import TransformerArray
    >>> df = Dataset(name_1, df_1, target_1)
    >>> params = [{'sampling_strategy': 0.98}, {'sampling_strategy': 1}]
    >>> transformer = RandomUnderSampler()
    >>> array = TransformerArray(transformer, parameters=params)
    >>> array.fit(df)
    >>> array.transform(df)

    Example 2

    >>> from test.resources.objects import *
    >>> from edgaro.data.dataset import Dataset
    >>> from edgaro.data.dataset_array import DatasetArray
    >>> from edgaro.balancing.transformer import RandomUnderSampler
    >>> from edgaro.balancing.transformer_array import TransformerArray
    >>> df = DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)])
    >>> params = [ [{'sampling_strategy': 0.98}, {'sampling_strategy': 1}] for _ in range(len(df)) ]
    >>> transformer = RandomUnderSampler()
    >>> array = TransformerArray(transformer, parameters=params)
    >>> array.fit(df)
    >>> array.transform(df)
    """
    def __init__(self, base_transformer: Transformer, parameters: Optional[List[Union[List, Dict[str, Any]]]] = None,
                 keep_original_dataset: bool = False, dataset_suffixes: Union[str, List[str]] = '_transformed',
                 result_array_sufix: str = '_transformed_array', allow_dataset_array_sufix_change: bool = True,
                 verbose: bool = False, set_suffixes: bool = True) -> None:
        super().__init__(base_transformer=base_transformer, parameters=parameters, transformer_sufix=result_array_sufix)
        self.__dataset_suffixes = None

        self.set_suffixes = set_suffixes

        self.set_dataset_suffixes(dataset_suffixes)
        self.keep_original_dataset = keep_original_dataset
        self.allow_dataset_array_sufix_change = allow_dataset_array_sufix_change
        self.verbose = verbose

    def set_dataset_suffixes(self, name_sufix: Union[str, List[str]]) -> None:
        """
        Set suffixes to be set to transformed Dataset.

        Parameters
        ----------
        name_sufix : str, list
            Suffixes to be set to a transformed Dataset.
        """
        if self.set_suffixes:
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
        """
        Get suffixes for transformed Dataset.

        Returns
        -------
        str, list
            Suffixes for a transformed Dataset.
        """
        return self.__dataset_suffixes

    def set_params(self, **params) -> None:
        """
        Set params for Transformer.

        Parameters
        ----------
        params : dict
            The parameters to be set.
        """
        super().set_params(**params)
        if self.__dataset_suffixes is not None and len(self.__dataset_suffixes) == 1:
            self.set_dataset_suffixes(self.__dataset_suffixes[0])

    def fit(self, dataset: Union[Dataset, DatasetArray]) -> None:
        """
        Fit the transformer.

        Parameters
        ----------
        dataset : Dataset, DatasetArray
            The object to fit Transformer on.
        """
        super().fit(dataset)
        self.__fix_classes()

        if self.set_suffixes:
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
                            elif len(self.__dataset_suffixes[i]) == len(self.transformers[i]):
                                self.transformers[i].set_dataset_suffixes(self.__dataset_suffixes[i])
                            elif len(self.__dataset_suffixes) == len(self.transformers[i]):
                                self.transformers[i].set_dataset_suffixes(self.__dataset_suffixes[i])
                            else:
                                raise Exception('Wrong length of dataset_suffixes!')

            if isinstance(self.__dataset_suffixes, list) and len(self.__dataset_suffixes) == 1 and isinstance(
                    self.__dataset_suffixes[0], str):
                if self.allow_dataset_array_sufix_change:
                    self.transformer_sufix = self.__dataset_suffixes[0]

        if self.verbose:
            print_unbuffered(f'TransformerArray {self.__repr__()} was fitted with {dataset.name}')

    def transform(self, dataset: Union[Dataset, DatasetArray]) -> Union[Dataset, DatasetArray]:
        """
        Transform the object.

        Parameters
        ----------
        dataset : Dataset, DatasetArray
            The object to be transformed.

        Returns
        -------
        Dataset, DatasetArray
            The transformed object.
        """
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

    def __base_transformer_array_to_balancing_transformer_array(self, base: Union[
            Transformer, TransformerArray, BaseTransformerArray]) -> Union[Transformer, TransformerArray]:
        if isinstance(base, Transformer):
            return base
        elif not isinstance(base, TransformerArray) and isinstance(base, BaseTransformerArray):
            out = TransformerArray(base_transformer=self.base_transformer)
            out.__class__ = self.__class__
            for key, val in base.__dict__.items():
                out.__dict__[key] = val

            return out
        else:
            return base

    @property
    def transformers(self) -> List[Union[Transformer, TransformerArray, List]]:
        """
        All the Transformer objects used by this object.

        Returns
        -------
        list[Transformer, TransformerArray, list]
        """
        return super().transformers

    @transformers.setter
    def transformers(self, val: List[Union[Transformer, TransformerArray, List]]) -> None:
        super().transformers = val

    @property
    def base_transformer(self) -> Transformer:
        """
        Base transformers for creation of this object.

        Returns
        -------
        Transformer
        """
        out = super().base_transformer
        if isinstance(out, Transformer):
            return out
        else:
            raise Exception('Wrong base_transformer attribute')

    @base_transformer.setter
    def base_transformer(self, val: Transformer) -> None:
        BaseTransformerArray.base_transformer.fset(self, val)

    def __str__(self) -> str:
        return f"TransformerArray{(' ' + self.__class__.__name__) if self.__class__.__name__ != 'TransformerArray' else ''} with {len(self.transformers)} transformers"

    def __repr__(self) -> str:
        return f"<TransformerArray{(' ' + self.__class__.__name__) if self.__class__.__name__ != 'TransformerArray' else ''} with {len(self.transformers)} transformers>"
