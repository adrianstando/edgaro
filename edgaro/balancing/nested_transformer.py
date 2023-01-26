from __future__ import annotations

import numpy as np

from typing import List, Dict, Optional, Union
from copy import deepcopy
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.under_sampling import ClusterCentroids, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek

from edgaro.balancing.transformer import TransformerFromIMBLEARN
from edgaro.balancing.transformer_array import TransformerArray
from edgaro.data.dataset import Dataset
from edgaro.data.dataset_array import DatasetArray
from edgaro.balancing.transformer import Transformer, RandomUnderSampler, RandomOverSampler, SMOTE
from edgaro.base.utils import print_unbuffered


class NestedAutomaticTransformer(Transformer):
    """
    Create NestedAutomaticTransformer.

    This object creates an object that behaves like a Transformer, but transforms a single Dataset using not only one,
    but more methods. Moreover, you do not specify Imbalance Ratio values, but they are automatically calculated. You
    pass only an argument `n_per_method` and the class will balance the Dataset
    with that number of Imbalance Raio values.

    Note: If you use NestedAutomaticTransformer (or children class) as a parameter to TransformerArray, it is advisable
    to pass parameter `set_suffixes=False` in TransformerArray object. Otherwise, the suffixes will be distorted.

    Parameters
    ----------
    base_transformers : list(Transformer)
        List of base transformers.
    base_transformers_names : list(str)
        List of names of base_transformers.
    keep_original_dataset : bool, default=False
        Keep the original Dataset after transformations or not.
    result_array_sufix str, default='_automatic_transformed_array'
        Suffix of the transformed DatasetArray.
    n_per_method : int, default=5
        Number of intermediate Imbalance Ratio values.
    random_state : int, optional, default=None,
        Random state seed.
    IR_round_precision : int, default=2
        Round precision of Imbalance Ratio when printing.
    min_samples_to_modify : int, optional, default=None
        Minimal number of samples to modify to create an intermediate Imbalance Ratio. If the number of modified
        observations is less than this number, the `n_per_method` parameter will be modified.
    verbose : bool, default=False
        Print messages during calculations.

    Attributes
    ----------
    keep_original_dataset : bool, default=False
        Keep the original Dataset after transformations or not.
    n_per_method : int, default=5
        Number of intermediate Imbalance Ratio values.
    random_state : int, optional, default=None,
        Random state seed.
    result_array_sufix str, default='_automatic_transformed_array'
        Suffix of the transformed DatasetArray.
    IR_round_precision : int, default=2
        Round precision of Imbalance Ratio when printing.
    min_samples_to_modify : int, optional, default=None
        Minimal number of samples to modify to create an intermediate Imbalance Ratio. If the number of modified
        observations is less than this number, the `n_per_method` parameter will be modified.
    verbose : bool, default=False
        Print messages during calculations.

    """
    def __init__(self, base_transformers: List[Transformer], base_transformers_names: List[str],
                 keep_original_dataset: bool = False, result_array_sufix: str = '_automatic_transformed_array',
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
        self.__base_transformers = base_transformers
        self.__base_transformers_names = base_transformers_names

        super(Transformer, self).__init__()

    def __find_IR_values(self, dataset: Dataset) -> List[float]:
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

        return IR_values

    def fit(self, dataset: Dataset) -> None:
        """
        Fit the transformer.

        Parameters
        ----------
        dataset : Dataset
            The object to fit Transformer on.
        """
        IR_values = self.__find_IR_values(dataset)

        for i in range(len(self.__base_transformers)):
            try:
                self.__base_transformers[i] = deepcopy(self.__base_transformers[i])
                self.__base_transformers[i].set_params(random_state=self.random_state)
            except (Exception,):
                pass

        self.__transformers = [
            TransformerArray(
                self.__base_transformers[i],
                parameters=[
                    {'IR': ir}
                    for ir in IR_values
                ],
                dataset_suffixes=[
                    f'_transformed_{self.__base_transformers_names[i]}_IR='
                    f'{round(ir, self.IR_round_precision)}='
                    f'{round(ir/dataset.imbalance_ratio, self.IR_round_precision)}BASE_IR='
                    f'{round((ir-1)/(dataset.imbalance_ratio-1), self.IR_round_precision)}IR_RANGE'
                    for ir in IR_values
                ],
                result_array_sufix=f'_{self.__base_transformers_names[i]}'
            )
            for i in range(len(self.__base_transformers))
        ]

        for transformer in self.__transformers:
            transformer.fit(dataset)

        self.__was_fitted = True

        if self.verbose:
            print_unbuffered(f'AutomaticTransformerArray {self.__repr__()} was fitted with {dataset.name}')

    def transform(self, dataset: Dataset) -> DatasetArray:
        """
        Transform the object.

        Parameters
        ----------
        dataset : Dataset
            The object to be transformed.

        Returns
        -------
        DatasetArray
            The transformed object.
        """
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
        """
        Set suffixes to be set to transformed Dataset.

        Parameters
        ----------
        name_sufix : str, list
            Suffixes to be set to a transformed Dataset.
        """
        if isinstance(self.__transformers, list) and np.all(
                [isinstance(t, TransformerArray) for t in self.__transformers]):
            for t in self.__transformers:
                t.set_dataset_suffixes(name_sufix)

    def get_dataset_suffixes(self) -> Optional[Union[str, List[str]]]:
        """
        Get suffixes for transformed Dataset.

        Returns
        -------
        str, list
            Suffixes for a transformed Dataset.
        """
        if isinstance(self.__transformers, list) and np.all(
                [isinstance(t, TransformerArray) for t in self.__transformers]):
            return [t.get_dataset_suffixes() for t in self.__transformers]
        else:
            return []

    def set_params(self, **params) -> None:
        """
        Set params for Transformer.

        Parameters
        ----------
        params : dict
            The parameters to be set.
        """
        if isinstance(self.__transformers, list) and np.all(
                [isinstance(t, TransformerArray) for t in self.__transformers]):
            for i in range(len(self.__transformers)):
                tmp = {}
                for key, val in params.items():
                    tmp[key] = val[i]
                self.__transformers[i].set_params(**tmp)

    def get_params(self) -> List[Union[Dict, List]]:
        """
        Get parameters of Transformer.

        Returns
        -------
        Dict, list
            The parameters.
        """
        if isinstance(self.__transformers, list) and np.all(
                [isinstance(t, TransformerArray) for t in self.__transformers]):
            return [t.parameters for t in self.__transformers]
        else:
            return []

    @property
    def was_fitted(self) -> bool:
        return self.__was_fitted

    @property
    def transformers(self) -> List[Union[Transformer, TransformerArray, List]]:
        """
        All the Transformer objects used by this object.

        Returns
        -------
        list[Transformer, TransformerArray, list]
        """
        if self.__transformers is None:
            return []
        else:
            return self.__transformers

    @transformers.setter
    def transformers(self, val: List[Union[Transformer, TransformerArray, List]]) -> None:
        if not self.was_fitted:
            self.__transformers = val
        else:
            raise Exception('Transformers were not set since Transformer has already been fitted!')

    @property
    def base_transformer(self) -> Union[Transformer, TransformerArray, List]:
        """
        Base transformers for creation of this object.

        Returns
        -------
        list[Transformer, TransformerArray, list]
        """
        if self.__base_transformers is None:
            return []
        else:
            return self.__base_transformers

    @base_transformer.setter
    def base_transformer(self, val: Union[Transformer, TransformerArray, List]) -> None:
        if not self.was_fitted:
            self.__base_transformers = val
        else:
            raise Exception('BaseTransformers were not set since Transformer has already been fitted!')

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
        out = f"NestedTransformer transformers" + "\n"
        if self.__transformers is not None:
            for t in self.__transformers:
                out += t.__str__()
                out += '\n'
        return out

    def __repr__(self) -> str:
        return "<NestedTransformer transformer>"


class BasicAutomaticTransformer(NestedAutomaticTransformer):
    """
    Create BasicAutomaticTransformer.

    This object contains three most popular methods - Random Under Sampler, Random Over Sampler and SMOTE.

    This class can be used for both continuous (numerical) and categorical data.

    Parameters
    ----------
    keep_original_dataset : bool, default=False
        Keep the original Dataset after transformations or not.
    result_array_sufix str, default='_automatic_transformed_array'
        Suffix of the transformed DatasetArray.
    n_per_method : int, default=5
        Number of intermediate Imbalance Ratio values.
    random_state : int, optional, default=None,
        Random state seed.
    IR_round_precision : int, default=2
        Round precision of Imbalance Ratio when printing.
    min_samples_to_modify : int, optional, default=None
        Minimal number of samples to modify to create an intermediate Imbalance Ratio. If the number of modified
        observations is less than this number, the `n_per_method` parameter will be modified.
    verbose : bool, default=False
        Print messages during calculations.

    """
    def __init__(self, keep_original_dataset: bool = False, result_array_sufix: str = '_automatic_transformed_array',
                 n_per_method: int = 5, random_state: Optional[int] = None, IR_round_precision: int = 2,
                 min_samples_to_modify: Optional[int] = None, verbose: bool = False) -> None:

        super().__init__(
            base_transformers=[RandomUnderSampler(), RandomOverSampler(), SMOTE()],
            base_transformers_names=['UNDERSAMPLING__Random', 'OVERSAMPLING__Random', 'OVERSAMPLING__SMOTE'],
            keep_original_dataset=keep_original_dataset, result_array_sufix=result_array_sufix,
            n_per_method=n_per_method, random_state=random_state, IR_round_precision=IR_round_precision,
            min_samples_to_modify=min_samples_to_modify, verbose=verbose)

    def __str__(self) -> str:
        out = f"BasicAutomaticTransformer transformers" + "\n"
        if self.transformers is not None:
            for t in self.transformers:
                out += t.__str__()
                out += '\n'
        return out

    def __repr__(self) -> str:
        return "<BasicAutomaticTransformer transformer>"


class ExtensionAutomaticTransformer(NestedAutomaticTransformer):
    """
    Create ExtensionAutomaticTransformer.

    This object contains three more complex methods implemented in imblearn. There are methods used for oversampling
    (BorderlineSMOTE), undersampling (NearMiss) and there is also a hybrid method (SMOTETomek).

    This class can be used only for continuous (numerical) data.

    Parameters
    ----------
    keep_original_dataset : bool, default=False
        Keep the original Dataset after transformations or not.
    result_array_sufix str, default='_automatic_transformed_array'
        Suffix of the transformed DatasetArray.
    n_per_method : int, default=5
        Number of intermediate Imbalance Ratio values.
    random_state : int, optional, default=None,
        Random state seed.
    IR_round_precision : int, default=2
        Round precision of Imbalance Ratio when printing.
    min_samples_to_modify : int, optional, default=None
        Minimal number of samples to modify to create an intermediate Imbalance Ratio. If the number of modified
        observations is less than this number, the `n_per_method` parameter will be modified.
    verbose : bool, default=False
        Print messages during calculations.

    """
    def __init__(self, keep_original_dataset: bool = False, result_array_sufix: str = '_automatic_transformed_array',
                 n_per_method: int = 5, random_state: Optional[int] = None, IR_round_precision: int = 2,
                 min_samples_to_modify: Optional[int] = None, verbose: bool = False) -> None:

        super().__init__(
            base_transformers=[
                TransformerFromIMBLEARN(NearMiss()),
                TransformerFromIMBLEARN(BorderlineSMOTE()),
                TransformerFromIMBLEARN(SMOTETomek())
            ],
            base_transformers_names=[
                'UNDERSAMPLING__Near_Miss',
                'OVERSAMPLING__Borderline_SMOTE',
                'COMBINED__SMOTE_Tomek'
            ],
            keep_original_dataset=keep_original_dataset, result_array_sufix=result_array_sufix,
            n_per_method=n_per_method, random_state=random_state, IR_round_precision=IR_round_precision,
            min_samples_to_modify=min_samples_to_modify, verbose=verbose)

    def __str__(self) -> str:
        out = f"ExtensionAutomaticTransformer transformers" + "\n"
        if self.transformers is not None:
            for t in self.transformers:
                out += t.__str__()
                out += '\n'
        return out

    def __repr__(self) -> str:
        return "<ExtensionAutomaticTransformer transformer>"


class AutomaticTransformer(NestedAutomaticTransformer):
    """
    Create AutomaticTransformer.

    This object contains six methods implemented in imblearn. There are methods used for oversampling
    (RandomOverSampling, BorderlineSMOTE), undersampling (RandomUnderSampling, NearMiss) and there is also a
    hybrid method (SMOTETomek).

    This class can be used only for continuous (numerical) data.

    Parameters
    ----------
    keep_original_dataset : bool, default=False
        Keep the original Dataset after transformations or not.
    result_array_sufix str, default='_automatic_transformed_array'
        Suffix of the transformed DatasetArray.
    n_per_method : int, default=5
        Number of intermediate Imbalance Ratio values.
    random_state : int, optional, default=None,
        Random state seed.
    IR_round_precision : int, default=2
        Round precision of Imbalance Ratio when printing.
    min_samples_to_modify : int, optional, default=None
        Minimal number of samples to modify to create an intermediate Imbalance Ratio. If the number of modified
        observations is less than this number, the `n_per_method` parameter will be modified.
    verbose : bool, default=False
        Print messages during calculations.

    """
    def __init__(self, keep_original_dataset: bool = False, result_array_sufix: str = '_automatic_transformed_array',
                 n_per_method: int = 5, random_state: Optional[int] = None, IR_round_precision: int = 2,
                 min_samples_to_modify: Optional[int] = None, verbose: bool = False) -> None:

        super().__init__(
            base_transformers=[
                RandomUnderSampler(),
                TransformerFromIMBLEARN(NearMiss()),
                RandomOverSampler(),
                SMOTE(),
                TransformerFromIMBLEARN(BorderlineSMOTE()),
                TransformerFromIMBLEARN(SMOTETomek())
            ],
            base_transformers_names=[
                'UNDERSAMPLING__Random',
                'UNDERSAMPLING__Near_Miss',
                'OVERSAMPLING__Random',
                'OVERSAMPLING__SMOTE',
                'OVERSAMPLING__Borderline_SMOTE',
                'COMBINED__SMOTE_Tomek'
            ],
            keep_original_dataset=keep_original_dataset, result_array_sufix=result_array_sufix,
            n_per_method=n_per_method, random_state=random_state, IR_round_precision=IR_round_precision,
            min_samples_to_modify=min_samples_to_modify, verbose=verbose)

    def __str__(self) -> str:
        out = f"AutomaticTransformer transformers" + "\n"
        if self.transformers is not None:
            for t in self.transformers:
                out += t.__str__()
                out += '\n'
        return out

    def __repr__(self) -> str:
        return "<AutomaticTransformer transformer>"
