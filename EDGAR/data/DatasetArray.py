from __future__ import annotations

from typing import List, Union, Optional
from EDGAR.data.Dataset import Dataset, DatasetFromOpenML
import openml


class DatasetArray:
    def __init__(self, datasets: List[Union[Dataset, DatasetArray]], name: str = 'dataset_array'):
        keys = [df.name for df in datasets]
        if len(set(keys)) == len(keys):
            self.datasets = datasets
            self.name = name
        else:
            raise Exception('Dataset names are not unique!')

    def __getitem__(self, key: Union[Union[str, int], List[Union[str, int]]]):
        if isinstance(key, list):
            out = DatasetArray([self.__getitem__(k) for k in key])
            if len(out) == 0:
                return None
            else:
                return out
        elif isinstance(key, str):
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
        if not isinstance(other, DatasetArray):
            return False
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
        for dataset in self.datasets:
            if isinstance(dataset, DatasetArray):
                dataset.remove_non_binary_target_datasets()

        # remove empty DatasetArrays
        self.datasets = [dataset for dataset in self.datasets if
                         not isinstance(dataset, DatasetArray) or len(dataset) > 0]

        # remove non-binary classification tasks
        self.datasets = [dataset
                         for dataset in self.datasets
                         if isinstance(dataset, DatasetArray) or (
                                 isinstance(dataset, Dataset) and dataset.check_binary_classification())
                         ]

    def remove_empty_datasets(self):
        for dataset in self.datasets:
            if isinstance(dataset, DatasetArray):
                dataset.remove_empty_datasets()

        # remove empty DatasetArrays
        self.datasets = [dataset for dataset in self.datasets if
                         not isinstance(dataset, DatasetArray) or len(dataset) > 0]

        # remove empty Datasets
        self.datasets = [dataset
                         for dataset in self.datasets
                         if isinstance(dataset, DatasetArray) or (
                                 isinstance(dataset, Dataset) and len(dataset.target) != 0 and len(dataset.data) != 0)
                         ]

    def append(self, other: Union[Dataset, DatasetArray, List[Dataset, DatasetArray]]):
        if isinstance(other, list):
            self.datasets += other
        else:
            self.datasets.append(other)

    def __str__(self):
        return ''.join([str(ds) + '\n' for ds in self.datasets])

    def __repr__(self):
        return f"<DatasetArray {self.name} with {len(self.datasets)} datasets>"


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
