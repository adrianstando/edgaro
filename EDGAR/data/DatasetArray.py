from __future__ import annotations

from typing import List, Union, Optional
from EDGAR.data.Dataset import Dataset, DatasetFromOpenML
import openml


class DatasetArray:
    def __init__(self, datasets: List[Dataset], name: str = 'dataset_array'):  # umożliwić DatasetArray
        keys = [df.name for df in datasets]
        if len(set(keys)) == len(keys):
            self.datasets = datasets
            self.name = name
        else:
            raise Exception('Dataset names are not unique!')

    def __getitem__(self, key: Union[str, int]) -> Optional[Dataset]:
        if isinstance(key, str):
            for df in self.datasets:
                if df.name == key:
                    return df
        elif isinstance(key, int):
            if key <= len(self.datasets):
                return self.datasets[key]
        return None

    def __len__(self):
        return len(self.datasets)

    def __eq__(self, other):
        if not self.name == other.name:
            return False
        if not len(self) == len(other):
            return False
        for m in self.datasets:
            if m not in other.datasets:
                return False
        return True

    def __iter__(self):
        self.current_i = 0
        return self

    def __next__(self):
        if self.current_i < len(self.datasets):
            out = self.datasets[self.current_i]
            self.current_i += 1
            return out
        else:
            raise StopIteration

    def remove_nans(self):
        for dataset in self.datasets:
            dataset.remove_nans()
        self.remove_empty_datasets()

    def remove_non_binary_target_datasets(self):
        self.datasets = [dataset for dataset in self.datasets if dataset.check_binary_classification()]

    def remove_empty_datasets(self):
        self.datasets = [dataset for dataset in self.datasets if len(dataset.target) != 0 and len(dataset.data) != 0]


class DatasetArrayFromOpenMLSuite(DatasetArray):
    def __init__(self, suite_name: str = 'OpenML100', apikey: Optional[str] = None, name: str = 'dataset_array'):
        if openml.config.apikey == '':
            if apikey is None:
                raise Exception('API key is not available!')
            else:
                openml.config.apikey = apikey

        benchmark_suite = openml.study.get_suite(suite_name)

        dataset_array = []
        for i in benchmark_suite.data:
            try:
                ds = DatasetFromOpenML(task_id=i, apikey=apikey)
                dataset_array.append(ds)
            except openml.exceptions.OpenMLServerException:
                print(f'The dataset numer {i} was not downloaded due to the server exception!')

        self.__openml_name = benchmark_suite.name if 'name' in benchmark_suite.__dict__.keys() else ''
        self.__openml_description = benchmark_suite.description if 'description' in benchmark_suite.__dict__.keys() else ''

        super().__init__(datasets=dataset_array, name=name)

    def print_openml_description(self):
        print('Name: ')
        print(self.__openml_name)
        print('Description: ')
        print(self.__openml_description)
