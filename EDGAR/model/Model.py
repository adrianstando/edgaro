from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from EDGAR.base.BaseTransformer import BaseTransformer
from EDGAR.data.Dataset import Dataset


# TODO:
# XGBoost
# Predefined model tunning (GridSearch, BayesSearch, RandomSearch)
# train test split

class Model(BaseTransformer, ABC):
    def __init__(self, name: str = '', test_size: float = 0.2):
        super().__init__()
        self.__transform_to_probabilities = False
        self.__train_dataset = None
        self.__test_dataset = None
        self.test_size = test_size
        self.name = name
        self.__label_encoders = {}
        self.__target_encoder = None

    def fit(self, dataset: Dataset, random_state: Optional[int] = None, print_scores: bool = False):
        if not dataset.check_binary_classification():
            raise Exception('Dataset does not represent binary classification task!')

        ds = deepcopy(dataset)
        X_train, X_test, y_train, y_test = train_test_split(ds.data, ds.target, test_size=self.test_size, random_state=random_state, stratify=ds.target)
        self.__train_dataset = Dataset(dataset.name + '_train', X_train, y_train)
        self.__test_dataset = Dataset(dataset.name + '_test', X_test, y_test)

        ds = deepcopy(self.__train_dataset)

        columns_to_encode = list(dataset.data.select_dtypes(include=['category', 'object']))
        for col in columns_to_encode:
            le = LabelEncoder()
            ds.data[col] = le.fit_transform(ds.data[col])
            self.__label_encoders[col] = le

        self.__target_encoder = _TargetEncode()
        ds.target = self.__target_encoder.fit_transform(ds.target)
        self.__train_dataset.target = self.__target_encoder.transform(self.__train_dataset.target)

        if self.name == '':
            self.name = dataset.name

        out = self._fit(ds)

        if print_scores:
            self.evaluate()

        return out

    @abstractmethod
    def _fit(self, dataset: Dataset):
        pass

    def predict(self, dataset: Dataset):
        df = deepcopy(dataset)
        for key, le in self.__label_encoders.items():
            df.data[key] = le.transform(df.data[key])

        model_name = '_' + self.name if not self.name == dataset.name else ''
        name = dataset.name + model_name + '_predicted'
        return self._predict(df, output_name=name)

    @abstractmethod
    def _predict(self, dataset: Dataset, output_name: str):
        pass

    def predict_proba(self, dataset: Dataset):
        df = deepcopy(dataset)
        for key, le in self.__label_encoders.items():
            df.data[key] = le.transform(df.data[key])

        model_name = '_' + self.name if not self.name == dataset.name else ''
        name = dataset.name + model_name + '_predicted_probabilities'
        return self._predict_proba(df, output_name=name)

    @abstractmethod
    def _predict_proba(self, dataset: Dataset, output_name: str):
        pass

    @abstractmethod
    def set_params(self, **params):
        pass

    @abstractmethod
    def get_params(self):
        pass

    def transform(self, dataset: Dataset):
        return self.__transform(dataset)

    def __transform(self, dataset: Dataset):
        if self.__transform_to_probabilities:
            return self.predict_proba(dataset)
        else:
            return self.predict(dataset)

    def set_transform_to_probabilities(self):
        self.__transform_to_probabilities = True

    def set_transform_to_classes(self):
        self.__transform_to_probabilities = False

    def get_train_dataset(self):
        return self.__train_dataset

    def get_test_dataset(self):
        return self.__test_dataset

    def get_category_colnames(self):
        return list(self.__label_encoders.keys())

    def evaluate(self, metrics_output_class=None, metrics_output_probabilities=None, ds: Optional[Dataset] = None):
        results = {}
        if ds is None:
            ds = self.__test_dataset
        if metrics_output_class is None and metrics_output_probabilities is None:
            metrics_output_class = [accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score]
            metrics_output_probabilities = [roc_auc_score]
        if len(metrics_output_class) > 0:
            y_hat = self.predict(ds)
            for f in metrics_output_class:
                results[f.__name__] = f(ds.target, y_hat)
        if len(metrics_output_probabilities) > 0:
            y_hat = self.predict_proba(ds)
            for f in metrics_output_probabilities:
                results[f.__name__] = f(ds.target, y_hat)
        return pd.DataFrame(results)


class _TargetEncode(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping = None
        super().__init__()

    def fit(self, y):
        names, counts = np.unique(y, return_counts=True)
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

    def transform(self, y):
        return y.map(self.mapping)


class ModelFromSKLEARN(Model):
    def __init__(self, base_model: BaseEstimator, name: str = '', test_size: float = 0.2):
        super().__init__(name=name, test_size=test_size)
        self.__model = base_model

    def _fit(self, dataset: Dataset):
        if dataset.target is None:
            raise Exception('Target data is not provided!')

        return self.__model.fit(dataset.data, dataset.target)

    def _predict(self, dataset: Dataset, output_name: str):
        if isinstance(dataset.data, np.ndarray) and isinstance(self.get_train_dataset().data, pd.DataFrame):
            dataset = deepcopy(dataset)
            dataset.data = pd.DataFrame(dataset, columns=self.get_train_dataset().data.columns)

        return Dataset(
            name=output_name,
            dataframe=None,
            target=self.__model.predict(dataset.data)
        )

    def _predict_proba(self, dataset: Dataset, output_name: str):
        if isinstance(dataset.data, np.ndarray) and isinstance(self.get_train_dataset().data, pd.DataFrame):
            dataset = deepcopy(dataset)
            dataset.data = pd.DataFrame(dataset, columns=self.get_train_dataset().data.columns)

        return Dataset(
            name=output_name,
            dataframe=None,
            target=self.__model.predict_proba(dataset.data)
        )

    def set_params(self, **params):
        return self.__model.set_params(**params)

    def get_params(self):
        return self.__model.get_params()


class RandomForest(ModelFromSKLEARN):
    def __init__(self, name: str = '', test_size: float = 0.2, *args, **kwargs):
        super().__init__(RandomForestClassifier(*args, **kwargs), name=name, test_size=test_size)
