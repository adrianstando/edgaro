from __future__ import annotations

import numpy as np
import warnings
import pandas as pd
import xgboost as xgb

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional, Protocol, Any, Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.utils.validation import check_is_fitted
from imblearn.metrics import geometric_mean_score, specificity_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV as RS
from sklearn.model_selection import GridSearchCV as GS

from EDGAR.data.dataset import Dataset
from EDGAR.base.base_transformer import BaseTransformer


class Model(BaseTransformer, ABC):
    def __init__(self, name: str = '', test_size: Optional[float] = None, random_state: Optional[int] = None) -> None:
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

    def fit(self, dataset: Dataset, print_scores: bool = False) -> None:
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
                if not dataset.was_split and self.test_size is not None:
                    dataset.train_test_split(test_size=self.test_size, random_state=self.random_state)
                if dataset.was_split and self.test_size is not None:
                    warnings.warn('Dataset was train-test-split! Dataset will not be split the second time.')
                self.__train_dataset = deepcopy(dataset.train)
                self.__test_dataset = deepcopy(dataset.test)

            ds = deepcopy(self.__train_dataset)

            columns_to_encode = list(dataset.data.select_dtypes(include=['category', 'object']))
            for col in columns_to_encode:
                le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                ds.data[col] = le.fit_transform(ds.data[[col]])
                self.__label_encoders[col] = le

            self.__target_encoder = _TargetEncode()
            ds.target = self.__target_encoder.fit_transform(ds.target)
            self.__train_dataset.target = self.__target_encoder.transform(self.__train_dataset.target)

            if self.name == '':
                self.name = dataset.name

            self._fit(ds)

            if print_scores:
                self.evaluate()

            self.__was_fitted = True

    @abstractmethod
    def _fit(self, dataset: Dataset) -> None:
        pass

    @property
    def was_fitted(self) -> bool:
        return self.__was_fitted

    def transform_data(self, dataset: Dataset) -> Dataset:
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
                    df.data[key] = le.transform(df.data[[key]])
            return df

    def transform_target(self, dataset: Dataset) -> Dataset:
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
        df = self.transform_data(dataset)
        model_name = '_' + self.name if not self.name == dataset.name else ''
        name = dataset.name + model_name + '_predicted'
        return self._predict(df, output_name=name)

    @abstractmethod
    def _predict(self, dataset: Dataset, output_name: str) -> Dataset:
        pass

    def predict_proba(self, dataset: Dataset) -> Dataset:
        df = self.transform_data(dataset)
        model_name = '_' + self.name if not self.name == dataset.name else ''
        name = dataset.name + model_name + '_predicted_probabilities'
        return self._predict_proba(df, output_name=name)

    @abstractmethod
    def _predict_proba(self, dataset: Dataset, output_name: str) -> Dataset:
        pass

    def set_params(self, **params) -> None:
        if 'test_size_model' in params.keys():
            self.test_size = params.pop('test_size_model')
        self._set_params(**params)

    @abstractmethod
    def _set_params(self, **params) -> None:
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        pass

    def transform(self, dataset: Dataset) -> Dataset:
        return self.__transform(dataset)

    def __transform(self, dataset: Dataset) -> Dataset:
        if self.__transform_to_probabilities:
            return self.predict_proba(dataset)
        else:
            return self.predict(dataset)

    def set_transform_to_probabilities(self) -> None:
        self.__transform_to_probabilities = True

    def set_transform_to_classes(self) -> None:
        self.__transform_to_probabilities = False

    def get_train_dataset(self) -> Optional[Dataset]:
        return self.__train_dataset

    def get_test_dataset(self) -> Optional[Dataset]:
        return self.__test_dataset

    def get_category_colnames(self) -> List[str]:
        return list(self.__label_encoders.keys())

    def evaluate(self, metrics_output_class=None, metrics_output_probabilities=None,
                 ds: Optional[Dataset] = None) -> pd.DataFrame:
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
        if len(metrics_output_class) > 0:
            y_hat = self.predict(ds)
            for f in metrics_output_class:
                results[f.__name__] = f(self.__target_encoder.transform(ds.target), y_hat.target)
        if len(metrics_output_probabilities) > 0:
            y_hat = self.predict_proba(ds)
            for f in metrics_output_probabilities:
                results[f.__name__] = f(self.__target_encoder.transform(ds.target), y_hat.target)
        return pd.DataFrame(results.items(), columns=['metric', 'value'])

    def __str__(self) -> str:
        return f"{self.__class__.__name__} model with name {self.name}"

    def __repr__(self) -> str:
        return f"<{self.name} {self.__class__.__name__} model>"


class _TargetEncode(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.mapping = None
        super().__init__()

    def fit(self, y: pd.Series) -> _TargetEncode:
        names, counts = np.unique(y, return_counts=True)
        if names is None:
            raise Exception('The input vector is wrong!')
        elif len(names) != 2:
            raise Exception('The input vector do not has two classes!')
        else:
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
    def __init__(self, base_model: SKLEARNModelProtocol, name: str = '', test_size: Optional[float] = None,
                 random_state: Optional[int] = None) -> None:
        super().__init__(name=name, test_size=test_size, random_state=random_state)
        self._model = base_model

    def _fit(self, dataset: Dataset) -> None:
        if dataset.target is None:
            raise Exception('Target data is not provided!')
        if dataset.data is None:
            raise Exception('Data in dataset is not provided!')

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
        if dataset.data is None:
            raise Exception('Data in dataset is not provided!')
        else:
            return Dataset(
                name=output_name,
                dataframe=None,
                target=pd.Series(self._model.predict(dataset.data))
            )

    def _predict_proba(self, dataset: Dataset, output_name: str) -> Dataset:
        if dataset.data is None:
            raise Exception('Data in dataset is not provided!')
        else:
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
    def __init__(self, name: str = '', test_size: Optional[float] = None, random_state: Optional[int] = None, *args,
                 **kwargs) -> None:
        super().__init__(RandomForestClassifier(*args, **kwargs), name=name, test_size=test_size,
                         random_state=random_state)


class XGBoost(ModelFromSKLEARN):
    def __init__(self, name: str = '', test_size: Optional[float] = None, random_state: Optional[int] = None, *args,
                 **kwargs) -> None:
        super().__init__(
            xgb.XGBClassifier(eval_metric='logloss' if 'eval_metric' not in kwargs.keys() else kwargs['eval_metric'],
                              *args, **kwargs),
            name=name, test_size=test_size, random_state=random_state
        )


class RandomSearchCV(ModelFromSKLEARN):
    def __init__(self, base_model: ModelFromSKLEARN, param_grid, n_iter=10, cv=5, scoring='balanced_accuracy',
                 name: str = '', test_size: Optional[float] = None, random_state: Optional[int] = None, *args,
                 **kwargs) -> None:
        super().__init__(
            RS(base_model._model, param_grid, cv=cv, scoring=scoring, n_iter=n_iter, *args, **kwargs),
            name=name,
            test_size=test_size,
            random_state=random_state
        )


class GridSearchCV(ModelFromSKLEARN):
    def __init__(self, base_model: ModelFromSKLEARN, param_grid, cv=5, scoring='balanced_accuracy', name: str = '',
                 test_size: Optional[float] = None, random_state: Optional[int] = None, *args, **kwargs) -> None:
        if 'random_state' in base_model._model.get_params().keys():
            base_model._model.set_params(**{'random_state': random_state})

        super().__init__(
            GS(base_model._model, param_grid, cv=cv, scoring=scoring, *args, **kwargs),
            name=name,
            test_size=test_size,
            random_state=random_state
        )
