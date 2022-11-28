from __future__ import annotations

import openml
import pandas as pd
import numpy as np

from copy import deepcopy
from typing import Optional, Union
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split

from edgaro.base.utils import print_unbuffered


class Dataset:
    def __init__(self, name: str, dataframe: Optional[pd.DataFrame], target: Optional[pd.Series],
                 verbose: bool = False) -> None:
        if dataframe is not None and target is not None:
            if dataframe.shape[0] != target.shape[0]:
                raise Exception('Dataframe and target have different number of rows!')

        self.name = name
        self.verbose = verbose
        self.__data = dataframe
        self.__target = target

        self.__train_dataset: Optional[Dataset] = None
        self.__test_dataset: Optional[Dataset] = None

        if self.verbose:
            print_unbuffered(f'Dataset {self.__repr__()} created')

    @property
    def data(self) -> Optional[pd.DataFrame]:
        if self.__data is not None:
            return self.__data
        else:
            if self.__train_dataset is None and self.__test_dataset is None:
                return None
            elif self.__train_dataset is not None and self.__test_dataset is not None:
                if self.__train_dataset.data is not None and self.__test_dataset.data is not None:
                    return pd.concat([self.__train_dataset.data, self.__test_dataset.data])
                else:
                    return None
            elif self.__train_dataset is not None and self.__train_dataset.data is not None:
                return self.__train_dataset.data
            elif self.__test_dataset is not None and self.__test_dataset.data is not None:
                return self.__test_dataset.data
            else:
                return None

    @data.setter
    def data(self, val) -> None:
        if self.__train_dataset is None and self.__test_dataset is None:
            self.__data = val
        else:
            raise Exception('Data cannot be set since the dataset was train-test-split!')

    @property
    def target(self) -> Optional[pd.Series]:
        if self.__target is not None:
            return self.__target
        else:
            if self.__train_dataset is None and self.__test_dataset is None:
                return None
            elif self.__train_dataset is not None and self.__test_dataset is not None:
                if self.__train_dataset.target is not None and self.__test_dataset.target is not None:
                    return pd.concat([self.__train_dataset.target, self.__test_dataset.target])
                else:
                    return None
            elif self.__train_dataset is not None and self.__train_dataset.target is not None:
                return self.__train_dataset.target
            elif self.__test_dataset is not None and self.__test_dataset.target is not None:
                return self.__test_dataset.target
            else:
                return None

    @target.setter
    def target(self, val) -> None:
        if self.__train_dataset is None and self.__test_dataset is None:
            self.__target = val
        else:
            raise Exception('Target cannot be set since the dataset was train-test-split!')

    @property
    def train(self) -> Dataset:
        if self.__train_dataset is not None:
            return self.__train_dataset
        else:
            raise Exception('The dataset as not train-test-split!')

    @property
    def test(self) -> Dataset:
        if self.__test_dataset is not None:
            return self.__test_dataset
        else:
            raise Exception('The dataset as not train-test-split!')

    @property
    def was_split(self) -> bool:
        return self.__train_dataset is not None and self.__test_dataset is not None

    def train_test_split(self, test_size: float = 0.2, random_state: Optional[int] = None) -> None:
        if self.was_split:
            raise Exception('The dataset has already been train-test-split!')
        X_train, X_test, y_train, y_test = train_test_split(deepcopy(self.__data), deepcopy(self.__target),
                                                            test_size=test_size,
                                                            random_state=random_state, stratify=self.__target)
        self.__train_dataset = Dataset(self.name + '_train', X_train, y_train)
        self.__test_dataset = Dataset(self.name + '_test', X_test, y_test)

        self.__data = None
        self.__target = None

        if self.verbose:
            print_unbuffered(f'Dataset {self.__repr__()} was train-test-split')

    def custom_train_test_split(self, train: Dataset, test: Dataset) -> None:
        self.__train_dataset = train
        self.__test_dataset = test

        self.__data = None
        self.__target = None

    def check_binary_classification(self) -> bool:
        if self.target is not None:
            unique = np.unique(self.target)
            if len(unique) > 2:
                return False
            return True
        else:
            return False

    def generate_report(self, output_path: Optional[str] = None, show_jupyter: bool = False,
                        minimal: bool = False) -> None:
        if self.data is None and self.target is None:
            raise Exception('Both data and target are None!')

        if self.target is None:
            data = self.data
        elif self.data is None:
            data = self.target
        else:
            data = self.data.assign(target=self.target)

        profile = ProfileReport(data, title='Pandas Profiling Report for ' + self.name, minimal=minimal,
                                progress_bar=False)

        if output_path is not None:
            profile.to_file(output_file=output_path)
        if show_jupyter:
            profile.to_notebook_iframe()

    @property
    def imbalance_ratio(self) -> float:
        if self.target is None:
            return float(0)
        names, counts = np.unique(self.target, return_counts=True)
        if len(names) == 1:
            return float(0)
        elif len(names) > 2:
            raise Exception('Target has too many classes for binary classification!')
        else:
            return float(max(counts) / min(counts))

    def __eq__(self, other) -> bool:
        if not isinstance(other, Dataset):
            return False
        if self.name != other.name:
            return False
        if self.data is not None and not self.data.equals(other.data):
            return False
        if self.target is not None and not self.target.equals(other.target):
            return False
        return True

    def remove_nans(self, col_thresh=0.9) -> None:
        if self.data is not None:
            nans = self.data.isnull().sum() / self.data.shape[0]
            nans = list(nans[nans > col_thresh].index)
            self.data.drop(nans, axis=1, inplace=True)

            nans = self.data.isna().any(axis=1)
            nans = list(nans[nans].index)
            self.data.drop(nans, axis=0, inplace=True)
            if self.target is not None:
                self.target.drop(nans, axis=0, inplace=True)

            nans = self.target.isna()
            nans = list(nans[nans].index)
            self.data.drop(nans, axis=0, inplace=True)
            if self.target is not None:
                self.target.drop(nans, axis=0, inplace=True)

    def remove_outliers(self, n_std: Union[float, int] = 3) -> None:
        if self.data is not None and self.target is not None:
            categorical_columns = list(self.data.select_dtypes(include=['category', 'object', 'int']))
            numerical_columns = list(set(self.data.columns).difference(categorical_columns))

            for col in numerical_columns:
                mean = self.data[col].mean()
                std = self.data[col].std()
                index = np.logical_and(self.data[col] <= mean + (n_std * std), (self.data[col] >= mean - (n_std * std)))
                self.data = self.data[index]
                self.target = self.target[index]

    def head(self, n: int = 10):
        new_data = self.data.head(n)
        new_target = self.target.head(n)
        return Dataset(
            name=self.name + '_head',
            dataframe=new_data,
            target=new_target,
            verbose=self.verbose
        )

    def __str__(self) -> str:
        out = f"Name: {self.name}"
        if self.data is not None:
            out += f"Dataset: \n{self.data.head()}"
        if self.target is not None:
            out += f"Target: \n{self.target.head()}"
        if self.check_binary_classification():
            out += f"Imbalance ratio: \n{self.imbalance_ratio}"
        return out

    def __repr__(self) -> str:
        return f"<Dataset {self.name}>"


class DatasetFromCSV(Dataset):
    def __init__(self, path: str, target: Optional[str] = None, name: str = 'dataset',
                 verbose: bool = False, *args, **kwargs) -> None:
        X = pd.read_csv(path, *args, **kwargs)

        if verbose:
            print_unbuffered(f'Data from {path} file is loaded')

        if target is None:
            y = X.iloc[:, -1]
            target = X.columns[-1]
        else:
            y = X[target]
        y = pd.Series(y, name='target')
        X = X.drop([target], axis=1)
        super().__init__(name=name, dataframe=X, target=y, verbose=verbose)


class DatasetFromOpenML(Dataset):
    """
    Before using this class, run 'openml configure apikey <KEY>' and replace <KEY> with your API OpenML key
    and create file '~/.openml/config' with content: ‘apikey=KEY’; in a new line add 'cache_dir = ‘DIR’' to cache data
    Or give API key as an argument apikey.

    In parameters give either task_id
    """

    def __init__(self, task_id: Optional[int] = None, apikey: Optional[str] = None, verbose: bool = False) -> None:
        if openml.config.apikey == '':
            if apikey is None:
                raise Exception('API key is not available!')
            else:
                openml.config.apikey = apikey

        data = openml.datasets.get_dataset(task_id)
        self.__openml_name = data.name if 'name' in data.__dict__.keys() else ''
        self.__openml_description = data.description if 'description' in data.__dict__.keys() else ''

        X, y, categorical_indicator, attribute_names = data.get_data(
            dataset_format='dataframe', target=data.default_target_attribute
        )

        if verbose:
            print_unbuffered(f'Dataset from OpenML with id {str(id)} was downloaded')

        X = pd.DataFrame(X, columns=attribute_names)
        y = pd.Series(y, name='target')

        for i in range(len(X.columns)):
            if categorical_indicator[i]:
                col = X.columns[i]
                col_type = X[col].dtype
                if col_type not in ['category', 'object', 'int']:
                    X[col] = np.array([col]).astype('category')

        super().__init__(name=data.name, dataframe=X, target=y, verbose=verbose)

    def openml_description(self) -> Optional[str]:
        return "Name: " + self.name + '\n' + 'Description: ' + '\n' + self.__openml_description
