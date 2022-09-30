import pandas as pd
import openml
import EDGAR.data.dataset.Dataset as Dataset


class DatasetFromOpenML(Dataset):
    """
    Before using this class, run 'openml configure apikey <MYKEY>' and replace <MYKEY> with your API OpenML key
    and create file '~/.openml/config' with content: ‘apikey=MYKEY’; in a new line add 'cachedir = ‘MYDIR’' to cache data
    Or give API key as an argument apikey.

    In parameters give either task_id or openml_dataset
    """
    def __init__(self, task_id: int = None, openml_dataset: openml.datasets.dataset.OpenMLDataset = None, apikey=None):
        if openml.config.apikey == '' and apikey is None:
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
