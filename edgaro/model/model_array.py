from __future__ import annotations

import pandas as pd

from typing import Optional, List, Dict, Any, Union, Callable

from edgaro.base.base_transformer_array import BaseTransformerArray
from edgaro.model.model import Model
from edgaro.data.dataset import Dataset
from edgaro.data.dataset_array import DatasetArray
from edgaro.base.utils import print_unbuffered


class ModelArray(BaseTransformerArray):
    """
    Create a class to train Models for each of the Dataset in DatasetArray.

    Parameters
    ----------
    base_model : Model
        The object defining the basic Model training procedure.
        The base_model object has to be clean - it cannot be fitted earlier.
    parameters : list[list, Dict[str, Any]]], optional
        The list of parameters for base_model. If the object is used for a DatasetArray object,
        the parameter list should be nested. For details, see Examples section.
    name : str
        A name of the ModelArray.
    verbose : bool, default=False
        Print messages during calculations.

    Attributes
    ----------
    name : str
        A name of the ModelArray.
    verbose : bool
        Print messages during calculations.

    Examples
    ----------
    Example 1

    >>> from test.resources.objects import *
    >>> from edgaro.data.dataset import Dataset
    >>> from edgaro.data.dataset_array import DatasetArray
    >>> from edgaro.model.model import RandomForest
    >>> from edgaro.model.model_array import ModelArray
    >>> df = Dataset(name_1, df_1, target_1)
    >>> params = [{'n_estimators': 20}]
    >>> model = RandomForest()
    >>> array = ModelArray(model, parameters=params)
    >>> array.fit(df)
    >>> array.predict(df)

    Example 2

    >>> from test.resources.objects import *
    >>> from edgaro.data.dataset import Dataset
    >>> from edgaro.data.dataset_array import DatasetArray
    >>> from edgaro.model.model import RandomForest
    >>> from edgaro.model.model_array import ModelArray
    >>> df = DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)])
    >>> params = [[{'n_estimators': 20}] for _ in range(len(df)) ]
    >>> model = RandomForest()
    >>> array = ModelArray(model, parameters=params)
    >>> array.fit(df)
    >>> array.predict(df)

    Example 3

    >>> from test.resources.objects import *
    >>> from edgaro.data.dataset import Dataset
    >>> from edgaro.data.dataset_array import DatasetArray
    >>> from edgaro.model.model import RandomForest
    >>> from edgaro.model.model_array import ModelArray
    >>> df = DatasetArray([
    ...         Dataset(name_2, df_1, target_1),
    ...         DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)])
    ... ])
    >>> params = [[{'n_estimators': 20}], [{'n_estimators': 10}, {'n_estimators': 30}]]
    >>> model = RandomForest()
    >>> array = ModelArray(model, parameters=params)
    >>> array.fit(df)
    >>> array.predict(df)

    """

    def __init__(self, base_model: Model, parameters: Optional[List[Union[Dict[str, Any], List]]] = None,
                 name: str = '', verbose: bool = False) -> None:
        super().__init__(base_transformer=base_model, parameters=parameters)
        self.name = name
        self.verbose = verbose

    def fit(self, dataset: Union[Dataset, DatasetArray]) -> None:
        """
        Fit the ModelArray.

        The fitting process includes encoding the categorical variables with OrdinalEncoder (from scikit-learn library)
        and target encoding (custom encoding, the minority class is encoded as 1, the majority class as 0).

        The method assumes that categorical variables, which has to be encoded,
        are one of the types: 'category', 'object'.

        Parameters
        ----------
        dataset : Dataset, DatasetArray
            The object to fit Model on.

        """
        if self.name == '':
            self.name = dataset.name

        if self.verbose:
            print_unbuffered(f'ModelArray {self.__repr__()} is being fitted with {dataset.name}')

        super().fit(dataset)
        self.__fix_classes(dataset)

        if self.verbose:
            print_unbuffered(f'ModelArray {self.__repr__()} was fitted with {dataset.name}')

    def predict(self, dataset: Union[Dataset, DatasetArray]) -> Union[Dataset, DatasetArray]:
        """
        Predict the class for a Dataset/DatasetArray object.

        Parameters
        ----------
        dataset : Dataset, DatasetArray
            A Dataset/DatasetArray object to make predictions on.

        Returns
        -------
        Dataset, DatasetArray

        """
        out = super().transform(dataset)

        if self.verbose:
            print_unbuffered(f'ModelArray {self.__repr__()} predicted on {dataset.name}')

        return out

    @staticmethod
    def __set_to_probs(inp: Union[List, Model, ModelArray]):
        if isinstance(inp, list):
            for i in inp:
                ModelArray.__set_to_probs(i)
        else:
            inp.set_transform_to_probabilities()

    @staticmethod
    def __set_to_class(inp: Union[List, Model, ModelArray]):
        if isinstance(inp, list):
            for i in inp:
                ModelArray.__set_to_class(i)
        else:
            inp.set_transform_to_classes()

    def predict_proba(self, dataset: Union[Dataset, DatasetArray]) -> Union[Dataset, DatasetArray]:
        """
        Predict the probability of class `1` for a Dataset/DatasetArray object.

        Parameters
        ----------
        dataset : Dataset, DatasetArray
            A Dataset/DatasetArray object to make predictions on.

        Returns
        -------
        Dataset, DatasetArray

        """
        ModelArray.__set_to_probs(self.get_models())
        out = super().transform(dataset)
        ModelArray.__set_to_class(self.get_models())

        if self.verbose:
            print_unbuffered(f'ModelArray {self.__repr__()} predicted probabilities on {dataset.name}')

        return out

    def get_models(self) -> List[Union[Model, ModelArray, List[Union[Model, ModelArray]]]]:
        """
        All the Model/ModelArray objects used by this object.

        Returns
        -------
        list[Model, ModelArray, list]
        """
        return self.transformers

    def set_transform_to_probabilities(self) -> None:
        """
        Make `transform` function return probabilities.
        """
        ModelArray.__set_to_probs(self.get_models())

    def set_transform_to_classes(self) -> None:
        """
        Make `transform` function return classes..
        """
        ModelArray.__set_to_class(self.get_models())

    def __fix_classes(self, dataset: Union[Dataset, DatasetArray]) -> None:
        for i in range(len(self.transformers)):
            self.transformers[i] = self.__base_transformer_array_to_model_array(
                self.transformers[i],
                dataset[i].name if isinstance(dataset, DatasetArray) else dataset.name
            )
            if isinstance(self.transformers[i], BaseTransformerArray):
                self.transformers[i].__fix_classes(dataset if isinstance(dataset, Dataset) else dataset[i])

    def __base_transformer_array_to_model_array(self, base: Union[Model, ModelArray, BaseTransformerArray],
                                                name: str) -> Union[Model, ModelArray]:
        if isinstance(base, Model):
            return base
        elif not isinstance(base, ModelArray) and isinstance(base, BaseTransformerArray):
            out = ModelArray(base_model=self.base_transformer)
            out.__class__ = self.__class__
            for key, val in base.__dict__.items():
                out.__dict__[key] = val
            out.name = name

            return out
        else:
            return base

    def evaluate(self, metrics_output_class: Optional[List[Callable[[pd.Series, pd.Series], float]]] = None,
                 metrics_output_probabilities: Optional[List[Callable[[pd.Series, pd.Series], float]]] = None,
                 ds: Optional[Union[Dataset, DatasetArray]] = None) -> pd.DataFrame:
        """
        Evaluate model.

        Parameters
        ----------
        metrics_output_class : list[Callable[[pd.Series, pd.Series], float]], optional, default=None
            List of functions to calculate metrics on predicted classes. If None is passed, accuracy, balanced accuracy,
            precision, recall, specificity, f1, f1_weighted, geometric mean score are used.
        metrics_output_probabilities : list[Callable[[pd.Series, pd.Series], float]], optional, default=None
            List of functions to calculate metrics on predicted probabilities. If None is passed, ROC AUC is used.
        ds : Dataset, DatasetArray, optional, default=None
            A Dataset/DatasetArray object to calculate metric on. If None is passed,
            test Dataset/DatasetArray from fitting is used.

        Returns
        -------
        pd.DataFrame
        """

        def _evaluate(mod, ds_, out_in):
            m = mod
            data = ds_ if isinstance(ds_, Dataset) else None
            eval_model = m.evaluate(metrics_output_class=metrics_output_class,
                                    metrics_output_probabilities=metrics_output_probabilities, ds=data)
            eval_model['model'] = m.name
            return pd.concat([out_in, eval_model])

        def _eval_all(mod, ds_, out_in):
            out_out = None
            if isinstance(mod, Model):
                out_out = _evaluate(mod, ds_, out_in)
            elif isinstance(mod, ModelArray):
                tmp = []
                for j in range(len(mod.get_models())):
                    m = mod.get_models()[j]
                    if isinstance(ds_, DatasetArray):
                        tmp.append(_eval_all(m, ds_[j], out_out))
                    else:
                        tmp.append(_eval_all(m, ds_, out_in))
                tmp.append(out_out)
                out_out = pd.concat(tmp)
            elif isinstance(mod, list):
                tmp = []
                for j in range(len(mod)):
                    if isinstance(ds_, DatasetArray):
                        tmp.append(_eval_all(mod[j], ds_[j], out_out))
                    else:
                        tmp.append(_eval_all(mod[j], ds_, out_out))
                tmp.append(out_out)
                out_out = pd.concat(tmp)
            return out_out

        if self.verbose:
            print_unbuffered(f'ModelArray {self.__repr__()} is being evaluated')

        out = pd.DataFrame({'model': [], 'metric': [], 'value': []})
        out = _eval_all(self.get_models(), ds, out)
        out = out[['model', 'metric', 'value']]
        out = out.reset_index(drop=True)

        if self.verbose:
            print_unbuffered(f'ModelArray {self.__repr__()} was evaluated')

        return out

    @property
    def transformers(self) -> List[Union[Model, ModelArray, List]]:
        """
        All the Model objects used by this object.

        Returns
        -------
        list[Model, ModelArray, list]
        """
        return super().transformers

    @transformers.setter
    def transformers(self, val: List[Union[Model, ModelArray, List]]) -> None:
        BaseTransformerArray.transformers.fset(self, val)

    @property
    def base_transformer(self) -> Model:
        """
        Base transformers for creation of this object.

        Returns
        -------
        Model
        """
        out = super().base_transformer
        if isinstance(out, Model):
            return out
        else:
            raise Exception('Wrong base_transformer attribute')

    @base_transformer.setter
    def base_transformer(self, val: Model) -> None:
        BaseTransformerArray.base_transformer.fset(self, val)

    def __str__(self) -> str:
        return f"ModelArray {self.name} with {len(self.get_models())} models"

    def __repr__(self) -> str:
        return f"<ModelArray {self.name} with {len(self.get_models())} models>"
