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
        if target is not None:
            unique = np.unique(target)
            if len(unique) > 2:
                raise Exception('This dataset is not correct for a binary classification task!')

        self.name = name
        self.data = dataframe
        self.target = target

    def generate_report(self, output_path: Optional[str] = None, show_jupyter: bool = False):
        if self.data is None and self.target is None:
            raise Exception('Both data and target are None!')

        if self.target is None:
            data = self.data
        elif self.data is None:
            data = self.target
        else:
            data = self.data.assign(target=self.target)

        profile = ProfileReport(data, title='Pandas Profiling Report for ' + self.name)

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




class DatasetFromCSV(Dataset):
    def __init__(self, path: str, target: str, name: str = 'dataset', *args, **kwargs):
        X = pd.read_csv(path, *args, **kwargs)
        y = X[target]
        y = pd.Series(y, name='target')
        X = X.drop([target], axis=1)
        super(Dataset, self).__init__(name=name, dataframe=X, target=y)


# TODO:
# Add printing description

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

        data = None
        if openml_dataset is not None:
            data = openml.datasets.get_dataset(task_id)
        else:
            data = openml_dataset

        X, y, categorical_indicator, attribute_names = data.get_data(
            dataset_format='dataframe', target=data.default_target_attribute
        )

        X = pd.DataFrame(X, columns=attribute_names)
        y = pd.Series(y, name='target')

        super(Dataset, self).__init__(name=data.name, dataframe=X, target=y)
