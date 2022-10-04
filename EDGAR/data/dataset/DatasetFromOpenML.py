import pandas as pd
import openml
from typing import Optional
from EDGAR.data.dataset.Dataset import Dataset


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
