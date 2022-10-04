from __future__ import annotations

from typing import List, Union, Optional
from EDGAR.data.Dataset import Dataset


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


class DatasetArrayFromOpenMLSuite(DatasetArray):
    raise NotImplementedError
