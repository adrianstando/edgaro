import pandas as pd
import numpy as np
import openml
from pandas_profiling import ProfileReport
from typing import Optional


# TODO:
# Error rate - comparing two datasets - "Stop Oversampling for Class Imbalance Learning: A
# Critical Review"

class Dataset:
    def __init__(self, name: str, dataframe: Optional[pd.DataFrame], target: Optional[pd.Series]):
        if dataframe is not None and target is not None:
            if dataframe.shape[0] != target.shape[0]:
                raise Exception('Dataframe and target have different number of rows!')

        self.name = name
        self.data = dataframe
        self.target = target

    def check_binary_classification(self):
        if self.target is not None:
            unique = np.unique(self.target)
            if len(unique) > 2:
                return False
            return True
        else:
            return False

    def generate_report(self, output_path: Optional[str] = None, show_jupyter: bool = False, minimal: bool = False):
        if self.data is None and self.target is None:
            raise Exception('Both data and target are None!')

        if self.target is None:
            data = self.data
        elif self.data is None:
            data = self.target
        else:
            data = self.data.assign(target=self.target)

        profile = ProfileReport(data, title='Pandas Profiling Report for ' + self.name, minimal=minimal, progress_bar=False)

        if output_path is not None:
            profile.to_file(output_file=output_path)
        if show_jupyter:
            profile.to_notebook_iframe()

    def imbalance_ratio(self):
        if self.target is None:
            return 0
        names, counts = np.unique(self.target, return_counts=True)
        if len(names) == 1:
            return 0
        elif len(names) > 2:
            raise Exception('Target has too many classes for binary classification!')
        else:
            return max(counts) / min(counts)

    def __eq__(self, other):
        if not isinstance(other, Dataset):
            return False
        if self.name != other.name:
            return False
        if not self.data.equals(other.data):
            return False
        if not self.target.equals(other.target):
            return False
        return True

    def remove_nans(self, col_thresh=0.9):
        nans = self.data.isnull().sum() / self.data.shape[0]
        nans = list(nans[nans > col_thresh].index)
        self.data.drop(nans, axis=1, inplace=True)

        nans = self.data.isna().any(axis=1)
        nans = list(nans[nans == True].index)
        self.data.drop(nans, axis=0, inplace=True)
        self.target.drop(nans, axis=0, inplace=True)

        nans = self.target.isna()
        nans = list(nans[nans == True].index)
        self.data.drop(nans, axis=0, inplace=True)
        self.target.drop(nans, axis=0, inplace=True)

    def __str__(self):
        out = f"Name: {self.name}"
        if self.data is not None:
            out += f"Dataset: \n{self.data.head()}"
        if self.target is not None:
            out += f"Target: \n{self.data.head()}"
        if self.check_binary_classification():
            out += f"Imbalance ratio: \n{self.imbalance_ratio()}"
        return out

    def __repr__(self):
        return f"<Dataset {self.name}>"


class DatasetFromCSV(Dataset):
    def __init__(self, path: str, target: str, name: str = 'dataset', *args, **kwargs):
        X = pd.read_csv(path, *args, **kwargs)
        y = X[target]
        y = pd.Series(y, name='target')
        X = X.drop([target], axis=1)
        super().__init__(name=name, dataframe=X, target=y)


class DatasetFromOpenML(Dataset):
    """
    Before using this class, run 'openml configure apikey <KEY>' and replace <KEY> with your API OpenML key
    and create file '~/.openml/config' with content: ‘apikey=KEY’; in a new line add 'cache_dir = ‘DIR’' to cache data
    Or give API key as an argument apikey.

    In parameters give either task_id or openml_dataset
    """

    def __init__(self, task_id: Optional[int] = None,
                 openml_dataset: Optional[openml.datasets.dataset.OpenMLDataset] = None, apikey: Optional[str] = None):
        if openml.config.apikey == '':
            if apikey is None:
                raise Exception('API key is not available!')
            else:
                openml.config.apikey = apikey

        if task_id is not None and openml_dataset is not None:
            raise Exception('Provide only one argument of task_id and openml_dataset!')

        if task_id is None and openml_dataset is None:
            raise Exception('Provide needed arguments!')

        if openml_dataset is None:
            data = openml.datasets.get_dataset(task_id)
        else:
            data = openml_dataset

        self.__openml_name = data.name if 'name' in data.__dict__.keys() else ''
        self.__openml_description = data.description if 'description' in data.__dict__.keys() else ''

        X, y, categorical_indicator, attribute_names = data.get_data(
            dataset_format='dataframe', target=data.default_target_attribute
        )

        X = pd.DataFrame(X, columns=attribute_names)
        y = pd.Series(y, name='target')

        super().__init__(name=data.name, dataframe=X, target=y)

    def print_openml_description(self):
        print('Name: ')
        print(self.__openml_name)
        print('\n')

        print('Description: ')
        print(self.__openml_description)

    def openml_description(self):
        return self.__openml_description
