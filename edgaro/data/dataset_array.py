from __future__ import annotations

import openml
import os
import pandas as pd
import numpy as np

from typing import List, Union, Optional

from edgaro.data.dataset import Dataset, DatasetFromOpenML
from edgaro.base.utils import print_unbuffered


class DatasetArray:
    def __init__(self, datasets: List[Union[Dataset, DatasetArray]],
                 name: str = 'dataset_array', verbose: bool = False) -> None:
        keys = [df.name for df in datasets]
        if len(set(keys)) == len(keys):
            self.verbose = verbose
            self.datasets = datasets
            self.name = name

            if self.verbose:
                print_unbuffered(f'DatasetArray {self.__repr__()} created')
        else:
            raise Exception('Dataset names are not unique!')

    def __getitem__(self, key: Union[Union[str, int], List[Union[str, int]]]) -> Optional[Union[DatasetArray, Dataset]]:
        if isinstance(key, list):
            outs = [self.__getitem__(k) for k in key]
            outs = [o for o in outs if o is not None]
            out = DatasetArray(outs, self.name + "_subset")
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

    def __len__(self) -> int:
        return len(self.datasets)

    def __eq__(self, other) -> bool:
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

    def __iter__(self) -> DatasetArray:
        self.current_i = 0
        return self

    def __next__(self) -> Union[Dataset, DatasetArray]:
        if self.current_i < len(self.datasets):
            out = self.datasets[self.current_i]
            self.current_i += 1
            return out
        else:
            raise StopIteration

    def train_test_split(self, test_size: float = 0.2, random_state: Optional[int] = None) -> None:
        for ds in self.datasets:
            ds.train_test_split(test_size, random_state)

        if self.verbose:
            print_unbuffered(f'DatasetArray {self.__repr__()} was train-test-split')

    @property
    def train(self) -> DatasetArray:
        out = [ds.train for ds in self.datasets]
        return DatasetArray(out, self.name + '_train')

    @property
    def test(self) -> DatasetArray:
        out = [ds.test for ds in self.datasets]
        return DatasetArray(out, self.name + '_test')

    def remove_nans(self) -> None:
        for dataset in self.datasets:
            dataset.remove_nans()
        self.remove_empty_datasets()

    def remove_outliers(self, n_std: Union[float, int] = 3) -> None:
        for dataset in self.datasets:
            dataset.remove_outliers(n_std=n_std)

    def remove_non_binary_target_datasets(self) -> None:
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

        if self.verbose:
            print_unbuffered(f'Non binary datasets were removed from DatasetArray {self.__repr__()}')

    def remove_empty_datasets(self) -> None:
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
                                 isinstance(dataset, Dataset) and
                                 dataset.target is not None and len(dataset.target) != 0 and
                                 dataset.data is not None and len(dataset.data) != 0)
                         ]

        if self.verbose:
            print_unbuffered(f'Empty datasets were removed from DatasetArray {self.__repr__()}')

    def append(self, other: Union[Dataset, DatasetArray, List[Union[Dataset, DatasetArray]]]) -> None:
        if isinstance(other, list):
            self.datasets += other
        else:
            self.datasets.append(other)

    def head(self, n: int = 10):
        return DatasetArray(
            datasets=[d.head(n) for d in self.datasets],
            name=self.name + '_head',
            verbose=self.verbose
        )

    def __str__(self) -> str:
        return ''.join([str(ds) + '\n' for ds in self.datasets])

    def __repr__(self) -> str:
        return f"<DatasetArray {self.name} with {len(self.datasets)} datasets>"


class DatasetArrayFromOpenMLSuite(DatasetArray):
    def __init__(self, suite_name: str = 'OpenML100', apikey: Optional[str] = None,
                 name: str = 'dataset_array', verbose: bool = False) -> None:
        if openml.config.apikey == '':
            if apikey is None:
                raise Exception('API key is not available!')
            else:
                openml.config.apikey = apikey

        benchmark_suite = openml.study.get_suite(suite_name)

        if verbose:
            print_unbuffered(f'Benchmark suite data was downloaded for {suite_name}')

        dataset_array = []
        if benchmark_suite.data is None:
            raise Exception('No data was downloaded!')
        else:
            for i in benchmark_suite.data:
                try:
                    ds = DatasetFromOpenML(task_id=i, apikey=apikey)
                    dataset_array.append(ds)
                except openml.exceptions.OpenMLServerException:
                    print(f'The dataset numer {i} was not downloaded due to the server exception!')

            self.__openml_name = benchmark_suite.name if 'name' in benchmark_suite.__dict__.keys() else ''
            self.__openml_description = benchmark_suite.description if 'description' in benchmark_suite.__dict__.keys() else ''

            super().__init__(datasets=dataset_array, name=name, verbose=verbose)

            if self.verbose:
                print_unbuffered(f'DatasetArray from OpenML benchmark suite {suite_name} was created')

    def openml_description(self) -> str:
        return "Name: " + self.__openml_name + '\n' + 'Description: ' + '\n' + self.__openml_description


class DatasetArrayFromDirectory(DatasetArray):
    """
    Attention!
    The target class is assumed to be the last column!
    """

    def __init__(self, path: str, name: str = 'dataset_array', verbose: bool = False) -> None:
        if not os.path.exists(path):
            raise Exception('The path does not exist!')
        if not os.path.isdir(path):
            raise Exception('The path argument is not a directory!')

        if verbose:
            print_unbuffered(f'The files from {path} are being loaded')

        dataset_array = []
        for filename in os.listdir(path):
            f = os.path.join(path, filename)
            if os.path.isfile(f):
                if f.lower().endswith('.csv'):
                    d = pd.read_csv(f)
                elif f.lower().endswith('.npy'):
                    d = np.load(f)
                else:
                    continue

                y = d[d.columns[-1]]
                X = d.drop(d.columns[-1], axis=1)
                dataset_array.append(
                    Dataset(dataframe=X, target=y, name=os.path.splitext(filename)[0])
                )

        if verbose:
            print_unbuffered(f'The files from {path} were loaded')

        super().__init__(datasets=dataset_array, name=name, verbose=verbose)
