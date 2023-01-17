from __future__ import annotations

import numpy as np
import warnings
import pandas as pd
import xgboost as xgb

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Protocol, Any, Dict, List, Callable
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.utils.validation import check_is_fitted
from imblearn.metrics import geometric_mean_score, specificity_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV as RS
from sklearn.model_selection import GridSearchCV as GS

from edgaro.data.dataset import Dataset
from edgaro.base.base_transformer import BaseTransformer
from edgaro.base.utils import print_unbuffered


class Model(BaseTransformer, ABC):
    """
    The abstract class to define a Machine Learning model for a single Dataset.

    Parameters
    ----------
    name : str
        A name of the Model.
    test_size : float, optional, default=None
        Test size for a Dataset object in case it was not train-test-split. If a Dataset object was not train-test-split
        and the parameter has value None, the training will be done on the all data.
    random_state : int, optional, default=None
        Random state seed.
    verbose : bool, default=False
        Print messages during calculations.

    Attributes
    ----------
    name : str
        A name of the Model.
    test_size : float, optional
        Test size for a Dataset object in case it was not train-test-split. If a Dataset object was not train-test-split
        and the parameter has value None, the training will be done on the all data.
    random_state : int, optional
        Random state seed.
    verbose : bool
        Print messages during calculations.
    majority_class_label : str, optional, default=None
        The label of the majority class. It is recommended to pass this argument; otherwise, it will be guessed.
        The guess may be wrong if the dataset is balanced - consequently, experiment results may be wrong. If None,
        it will be tried to extract the information from the majority_class_label attribute of the Dataset object.

    """

    def __init__(self, name: str = '', test_size: Optional[float] = None, random_state: Optional[int] = None,
                 verbose: bool = False, majority_class_label: Optional[str] = None) -> None:
        super().__init__()
        self.__transform_to_probabilities = False
        self.__train_dataset = None
        self.__test_dataset = None
        self.test_size = test_size
        self.name = name
        self.__label_encoders = {}
        self.__target_encoder = None
        self.random_state = random_state
        self.__was_fitted = False
        self.verbose = verbose
        self.majority_class_label = majority_class_label

    def fit(self, dataset: Dataset, print_scores: bool = False) -> None:
        """
        Fit the Model.

        The fitting process includes encoding the categorical variables with OrdinalEncoder (from scikit-learn library)
        and target encoding (custom encoding, the minority class is encoded as 1, the majority class as 0).

        The method assumes that categorical variables, which has to be encoded,
        are one of the types: 'category', 'object'.

        Parameters
        ----------
        dataset : Dataset
            The object to fit Model on.
        print_scores : bool, default=False
            Indicates whether model evaluation on a test dataset should be printed at the end of fitting.

        """
        if self.verbose:
            print_unbuffered(f'Model {self.__repr__()} is being fitted with {dataset.name}')

        if self.majority_class_label is None and dataset.majority_class_label is not None:
            self.majority_class_label = dataset.majority_class_label

        if dataset.data is None or (dataset.data is not None and len(dataset.data) == 0):
            raise Exception('The dataset has empty data!')
        elif dataset.target is None or (dataset.target is not None and len(dataset.target) == 0):
            raise Exception('The dataset has empty data!')
        else:
            if not dataset.check_binary_classification():
                raise Exception('Dataset does not represent binary classification task!')

            if not dataset.was_split and self.test_size is None:
                warnings.warn(
                    'Dataset was not train-test-split! The training dataset will be the same as the test dataset.')
                self.__train_dataset = deepcopy(dataset)
                self.__test_dataset = deepcopy(dataset)
            else:
                if dataset.was_split and self.test_size is not None:
                    warnings.warn('Dataset was train-test-split! Dataset will not be split the second time.')
                elif not dataset.was_split and self.test_size is not None:
                    dataset.train_test_split(test_size=self.test_size, random_state=self.random_state)
                self.__train_dataset = deepcopy(dataset.train)
                self.__test_dataset = deepcopy(dataset.test)

            ds = deepcopy(self.__train_dataset)

            columns_to_encode = list(dataset.data.select_dtypes(include=['category', 'object']))
            for col in columns_to_encode:
                le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                ds.data[col] = le.fit_transform(ds.data[[col]])
                self.__label_encoders[col] = le

            self.__target_encoder = _TargetEncode(majority_class_label=self.majority_class_label)
            ds.target = self.__target_encoder.fit_transform(ds.target)
            self.__train_dataset.target = self.__target_encoder.transform(self.__train_dataset.target)

            if self.name == '':
                self.name = dataset.name

            self._fit(ds)

            if print_scores:
                print_unbuffered(self.evaluate())

            self.__was_fitted = True

            if self.verbose:
                print_unbuffered(f'Model {self.__repr__()} was fitted with {dataset.name}')

    @abstractmethod
    def _fit(self, dataset: Dataset) -> None:
        pass

    @property
    def was_fitted(self) -> bool:
        """
        The information whether the Model was fitted.

        Returns
        -------
        bool
        """
        return self.__was_fitted

    def transform_data(self, dataset: Dataset) -> Dataset:
        """
        Encode dataset.data with the rules generated after fitting this object.

        Parameters
        ----------
        dataset : Dataset
            A Dataset object, where `.data` attribute will be encoded. The method returns a new object.

        Returns
        -------
        Dataset

        """
        if dataset.data is None or (dataset.data is not None and len(dataset.data) == 0):
            raise Exception('The dataset has empty data!')
        else:
            df = Dataset(
                name=dataset.name,
                dataframe=deepcopy(dataset.data),
                target=deepcopy(dataset.target)
            )
            if len(self.__label_encoders) > 0:
                for key, le in self.__label_encoders.items():
                    df.data[key] = le.transform(pd.DataFrame(df.data[key]))
            return df

    def transform_target(self, dataset: Dataset) -> Dataset:
        """
        Encode dataset.target with the rules generated after fitting this object.

        Parameters
        ----------
        dataset : Dataset
            A Dataset object, where `.target` attribute will be encoded. The method returns a new object.

        Returns
        -------
        Dataset

        """
        if dataset.target is None or (dataset.target is not None and len(dataset.target) == 0):
            raise Exception('The dataset has empty data!')
        else:
            df = Dataset(
                name=dataset.name,
                dataframe=deepcopy(dataset.data),
                target=deepcopy(dataset.target)
            )
            if self.__target_encoder is not None:
                df.target = self.__target_encoder.transform(df.target)
            return df

    def predict(self, dataset: Dataset) -> Dataset:
        """
        Predict the class for a Dataset object.

        Parameters
        ----------
        dataset : Dataset
            A Dataset object to make predictions on.

        Returns
        -------
        Dataset

        """
        if dataset.data is None:
            raise Exception('Data in dataset is not provided!')

        df = self.transform_data(dataset)
        model_name = '_' + self.name if not self.name == dataset.name else ''
        name = dataset.name + model_name + '_predicted'

        if self.verbose:
            print_unbuffered(f'Model {self.__repr__()} predicted on {dataset.name}')

        return self._predict(df, output_name=name)

    @abstractmethod
    def _predict(self, dataset: Dataset, output_name: str) -> Dataset:
        pass

    def predict_proba(self, dataset: Dataset) -> Dataset:
        """
        Predict the probability of class `1` for a Dataset object.

        Parameters
        ----------
        dataset : Dataset
            A Dataset object to make predictions on.

        Returns
        -------
        Dataset

        """
        if dataset.data is None:
            raise Exception('Data in dataset is not provided!')

        df = self.transform_data(dataset)
        model_name = '_' + self.name if not self.name == dataset.name else ''
        name = dataset.name + model_name + '_predicted_probabilities'

        if self.verbose:
            print_unbuffered(f'Model {self.__repr__()} predicted probabilities on {dataset.name}')

        return self._predict_proba(df, output_name=name)

    @abstractmethod
    def _predict_proba(self, dataset: Dataset, output_name: str) -> Dataset:
        pass

    def set_params(self, **params) -> None:
        """
        Set params for Model.

        Parameters
        ----------
        params : dict
            The parameters to be set.

        """
        if 'test_size_model' in params.keys():
            self.test_size = params.pop('test_size_model')
        self._set_params(**params)

    @abstractmethod
    def _set_params(self, **params) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        """
        Get parameters of Model.

        Returns
        -------
        Dict, list
            The parameters.

        """
        pass

    def transform(self, dataset: Dataset) -> Dataset:
        """
        A function to make the Model compatible with BaseTransformer.

        It can either return predicted classes or predicted probabilities - it can be set using
        `set_transform_to_probabilities` and `set_transform_to_classes` functions.

        Parameters
        ----------
        dataset : Dataset
            A Dataset object to be transformed.
        Returns
        -------
        Dataset
        """
        return self.__transform(dataset)

    def __transform(self, dataset: Dataset) -> Dataset:
        if self.__transform_to_probabilities:
            return self.predict_proba(dataset)
        else:
            return self.predict(dataset)

    def set_transform_to_probabilities(self) -> None:
        """
        Make `transform` function return probabilities.
        """
        self.__transform_to_probabilities = True

    def set_transform_to_classes(self) -> None:
        """
        Make `transform` function return classes..
        """
        self.__transform_to_probabilities = False

    def get_train_dataset(self) -> Optional[Dataset]:
        """
        Get a Dataset used for a training process.

        Returns
        -------
        Dataset

        """
        return self.__train_dataset

    def get_test_dataset(self) -> Optional[Dataset]:
        """
        Get a Dataset used for a test process.

        Returns
        -------
        Dataset

        """
        return self.__test_dataset

    def get_category_colnames(self) -> List[str]:
        """
        Get category column names, which were encoded during the fitting process.

        Returns
        -------
        list(str)
        """
        if len(self.__label_encoders) > 0:
            return list(self.__label_encoders.keys())
        else:
            return []

    def evaluate(self, metrics_output_class: Optional[List[Callable[[pd.Series, pd.Series], float]]] = None,
                 metrics_output_probabilities: Optional[List[Callable[[pd.Series, pd.Series], float]]] = None,
                 ds: Optional[Dataset] = None) -> pd.DataFrame:
        """
        Evaluate model.

        Parameters
        ----------
        metrics_output_class : list[Callable[[pd.Series, pd.Series], float]], optional, default=None
            List of functions to calculate metrics on predicted classes. If None is passed, accuracy, balanced accuracy,
            precision, recall, specificity, f1, f1_weighted, geometric mean score are used.
        metrics_output_probabilities : list[Callable[[pd.Series, pd.Series], float]], optional, default=None
            List of functions to calculate metrics on predicted probabilities. If None is passed, ROC AUC is used.
        ds : Dataset, optional, default=None
            A Dataset object to calculate metric on. If None is passed, test Dataset from fitting is used.

        Returns
        -------
        pd.DataFrame
        """

        if self.verbose:
            print_unbuffered(f'Model {self.__repr__()} is being evaluated')

        results = {}
        if ds is None:
            if self.__test_dataset is not None:
                ds = self.__test_dataset
            else:
                raise Exception('There is not test dataset and the ds argument was not provided!')
        if metrics_output_class is None and metrics_output_probabilities is None:
            def f1_weighted(y_true, y_pred):
                return f1_score(y_true, y_pred, average='weighted')

            metrics_output_class = [accuracy_score, balanced_accuracy_score, precision_score,
                                    recall_score, specificity_score, f1_score, f1_weighted, geometric_mean_score]
            metrics_output_probabilities = [roc_auc_score]
        if metrics_output_class is not None and len(metrics_output_class) > 0:
            y_hat = self.predict(ds)
            for f in metrics_output_class:
                results[f.__name__] = f(self.__target_encoder.transform(ds.target), y_hat.target)
        if metrics_output_probabilities is not None and len(metrics_output_probabilities) > 0:
            y_hat = self.predict_proba(ds)
            for f in metrics_output_probabilities:
                results[f.__name__] = f(self.__target_encoder.transform(ds.target), y_hat.target)

        if self.verbose:
            print_unbuffered(f'Model {self.__repr__()} was evaluated')

        return pd.DataFrame(results.items(), columns=['metric', 'value'])

    def __str__(self) -> str:
        return f"{self.__class__.__name__} model with name {self.name}"

    def __repr__(self) -> str:
        return f"<{self.name} {self.__class__.__name__} model>"


class _TargetEncode(BaseEstimator, TransformerMixin):
    def __init__(self, majority_class_label: Optional[str] = None) -> None:
        self.mapping = None
        super().__init__()
        self.majority_class_label = majority_class_label

    def fit(self, y: pd.Series) -> _TargetEncode:
        names, counts = np.unique(y, return_counts=True)
        if names is None:
            raise Exception('The input vector is wrong!')
        elif len(names) != 2:
            raise Exception('The input vector do not has two classes!')
        elif self.majority_class_label is not None:
            names = names.tolist()
            ind = names.index(self.majority_class_label)
            if counts[0] == counts[1]:
                self.mapping = {
                    names[0 if ind == 1 else 1]: 1,
                    names[ind]: 0
                }
                return self
            elif (ind == 1 and counts[0] < counts[1]) or (ind == 0 and counts[1] < counts[0]):
                raise Exception('Wrong majority class label!')

        if counts[0] < counts[1]:
            index_min = 0
            index_max = 1
        else:
            index_min = 1
            index_max = 0

        self.mapping = {
            names[index_min]: 1,
            names[index_max]: 0
        }
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        return y.map(self.mapping)


class SKLEARNModelProtocol(Protocol):
    """
    A Protocol to define the expected structure of a Model from `scikit-learn` library.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any:
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def get_params(self) -> Dict:
        ...

    def set_params(self, **params) -> Any:
        ...


class ModelFromSKLEARN(Model):
    """
    Create Model from a model in `scikit-learn` library.

    Parameters
    ----------
    base_model : SKLEARNModelProtocol
        A model from `scikit-learn` library. Note: this object has to be clean (not fitted).
    name : str
        A name of the Model.
    test_size : float, optional, default=None
        Test size for a Dataset object in case it was not train-test-split. If a Dataset object was not train-test-split
        and the parameter has value None, the training will be done on the all data.
    random_state : int, optional, default=None
        Random state seed.
    verbose : bool, default=False
        Print messages during calculations.

    """

    def __init__(self, base_model: SKLEARNModelProtocol, name: str = '', test_size: Optional[float] = None,
                 random_state: Optional[int] = None, verbose: bool = False) -> None:
        super().__init__(name=name, test_size=test_size, random_state=random_state)
        self._model = base_model
        self.verbose = verbose

    def _fit(self, dataset: Dataset) -> None:
        self._model.set_params(verbose=self.verbose)

        if 'random_state' in self._model.get_params().keys() and \
                self._model.get_params()['random_state'] is None and \
                self.random_state is not None:
            self._model.set_params(**{'random_state': self.random_state})

        self._model.fit(dataset.data, dataset.target)

    @property
    def was_fitted(self) -> bool:
        try:
            check_is_fitted(self._model)
            return True
        except (Exception,):
            return False

    def _predict(self, dataset: Dataset, output_name: str) -> Dataset:
        return Dataset(
            name=output_name,
            dataframe=None,
            target=pd.Series(self._model.predict(dataset.data))
        )

    def _predict_proba(self, dataset: Dataset, output_name: str) -> Dataset:
        return Dataset(
            name=output_name,
            dataframe=None,
            target=pd.Series(self._model.predict_proba(dataset.data)[:, 1])
        )

    def _set_params(self, **params) -> None:
        return self._model.set_params(**params)

    def get_params(self) -> Dict:
        return self._model.get_params()


class RandomForest(ModelFromSKLEARN):
    """
    Create RandomForest Model from a RandomForestClassifier implementation in `scikit-learn` library.

    Parameters
    ----------
    name : str
        A name of the Model.
    test_size : float, optional, default=None
        Test size for a Dataset object in case it was not train-test-split. If a Dataset object was not train-test-split
        and the parameter has value None, the training will be done on the all data.
    random_state : int, optional, default=None
        Random state seed.
    verbose : bool, default=False
        Print messages during calculations.
    *args : tuple, optional
        Additional parameters for RandomForestClassifier from `scikit-learn` library.
    **kwargs : dict, optional
        Additional parameters for RandomForestClassifier from `scikit-learn` library.

    """

    def __init__(self, name: str = '', test_size: Optional[float] = None, random_state: Optional[int] = None,
                 verbose: bool = False, *args, **kwargs) -> None:
        super().__init__(RandomForestClassifier(*args, **kwargs), name=name, test_size=test_size,
                         random_state=random_state, verbose=verbose)


class XGBoost(ModelFromSKLEARN):
    """
    Create XGBoost Model from a XGBClassifier implementation in `xgboost` library.

    Parameters
    ----------
    name : str
        A name of the Model.
    test_size : float, optional, default=None
        Test size for a Dataset object in case it was not train-test-split. If a Dataset object was not train-test-split
        and the parameter has value None, the training will be done on the all data.
    random_state : int, optional, default=None
        Random state seed.
    verbose : bool, default=False
        Print messages during calculations.
    *args : tuple, optional
        Additional parameters for XGBClassifier from `xgboost` library.
    **kwargs : dict, optional
        Additional parameters for XGBClassifier from `xgboost` library.

    """

    def __init__(self, name: str = '', test_size: Optional[float] = None, random_state: Optional[int] = None,
                 verbose: bool = False, *args, **kwargs) -> None:
        super().__init__(
            xgb.XGBClassifier(eval_metric='logloss' if 'eval_metric' not in kwargs.keys() else kwargs['eval_metric'],
                              *args, **kwargs),
            name=name, test_size=test_size, random_state=random_state, verbose=verbose
        )


class RandomSearchCV(ModelFromSKLEARN):
    """
    Create Model to perform Random Search on any of the model implementation matching SKLEARNModelProtocol.

    Parameters
    ----------
    base_model : SKLEARNModelProtocol
        A model from `scikit-learn` library. Note: this object has to be clean (not fitted).
    param_grid : Dict
        A parameter grid for searching.
    n_iter : int
        Number of iterations to be performed.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Name of a function to be used to choose the best model.
    name : str
        A name of the Model.
    test_size : float, optional, default=None
        Test size for a Dataset object in case it was not train-test-split. If a Dataset object was not train-test-split
        and the parameter has value None, the training will be done on the all data.
    random_state : int, optional, default=None
        Random state seed.
    verbose : bool, default=False
        Print messages during calculations.

    """

    def __init__(self, base_model: ModelFromSKLEARN, param_grid: Dict, n_iter: int = 10, cv: int = 5,
                 scoring: str = 'balanced_accuracy', name: str = '', test_size: Optional[float] = None,
                 random_state: Optional[int] = None, verbose: bool = False, *args, **kwargs) -> None:
        super().__init__(
            RS(base_model._model, param_grid, cv=cv, scoring=scoring, n_iter=n_iter, *args, **kwargs),
            name=name,
            test_size=test_size,
            random_state=random_state,
            verbose=verbose
        )


class GridSearchCV(ModelFromSKLEARN):
    """
    Create Model to perform Grid Search on any of the model implementation matching SKLEARNModelProtocol.

    Parameters
    ----------
    base_model : SKLEARNModelProtocol
        A model from `scikit-learn` library. Note: this object has to be clean (not fitted).
    param_grid : Dict
        A parameter grid for searching.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Name of a function to be used to choose the best model.
    name : str
        A name of the Model.
    test_size : float, optional, default=None
        Test size for a Dataset object in case it was not train-test-split. If a Dataset object was not train-test-split
        and the parameter has value None, the training will be done on the all data.
    random_state : int, optional, default=None
        Random state seed.
    verbose : bool, default=False
        Print messages during calculations.

    """

    def __init__(self, base_model: ModelFromSKLEARN, param_grid: Dict, cv: int = 5,
                 scoring: str = 'balanced_accuracy', name: str = '', test_size: Optional[float] = None,
                 random_state: Optional[int] = None, verbose: bool = False, *args, **kwargs) -> None:
        if 'random_state' in base_model._model.get_params().keys():
            base_model._model.set_params(**{'random_state': random_state})

        super().__init__(
            GS(base_model._model, param_grid, cv=cv, scoring=scoring, *args, **kwargs),
            name=name,
            test_size=test_size,
            random_state=random_state,
            verbose=verbose
        )
