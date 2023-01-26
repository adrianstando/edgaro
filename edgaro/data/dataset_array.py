from __future__ import annotations

import openml
import os
import pandas as pd
import numpy as np

from typing import List, Union, Optional
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import train_test_split

from edgaro.data.dataset import Dataset, DatasetFromOpenML
from edgaro.base.utils import print_unbuffered


class DatasetArray:
    """ Create a DatasetArray object

    This class creates a unified representation of an array of the Dataset objects, which can be further processed by
    other package classes.

    Parameters
    ----------
    datasets : list[Dataset, DatasetArray]
        The list of Dataset and DatasetArray objects.
    name : str, default='dataset_array'
        Name of the dataset array.
    verbose : bool, default=False
        Print messages during calculations.

    Attributes
    ----------
    name : str
        Name of the dataset array.
    datasets : list[Dataset, DatasetArray]
        The list of Dataset and DatasetArray objects.
    verbose : bool
        Print messages during calculations.
    """

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
        """
        Split each of the Dataset objects into train and test datasets.

        Parameters
        ----------
        test_size : float, default=0.2
            The size of a train dataset.
        random_state : int, optional, default=None
            Random state seed.
        """
        for ds in self.datasets:
            ds.train_test_split(test_size, random_state)

        if self.verbose:
            print_unbuffered(f'DatasetArray {self.__repr__()} was train-test-split')

    @property
    def train(self) -> DatasetArray:
        """
        DatasetArray : the DatasetArray of train part of Dataset objects if the Dataset objects were train-test-split
        """
        out = [ds.train for ds in self.datasets]
        return DatasetArray(out, self.name + '_train')

    @property
    def test(self) -> DatasetArray:
        """
        DatasetArray : the DatasetArray of train part of Dataset objects if the Dataset objects were train-test-split
        """
        out = [ds.test for ds in self.datasets]
        return DatasetArray(out, self.name + '_test')

    def remove_nans(self, col_thresh: float = 0.9) -> None:
        """
        Remove rows with NaN values and columns containing almost only NaN values.

        Parameters
        ----------
        col_thresh : float, default=0.9
            The threshold of NaN values in columns above which a column should be dropped
        """
        for dataset in self.datasets:
            dataset.remove_nans(col_thresh=col_thresh)
        self.remove_empty_datasets()

    def remove_outliers(self, n_std: Union[float, int] = 3) -> None:
        """
        Remove outliers with NaN values and columns containing almost only NaN values.

        It is only applicable for continuous variables (that means not `category`, 'object' and 'int' type).

        Parameters
        ----------
        n_std : float, int, default=3
            Number of standard deviations.
            The observations that lies outside the range `column_mean +/- n_std*column_std` will be removed.
        """
        for dataset in self.datasets:
            dataset.remove_outliers(n_std=n_std)

    def remove_non_binary_target_datasets(self) -> None:
        """
        Remove Dataset objects which do not represent binary classification task.
        """
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
        """
        Remove empty Dataset objects.
        """
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

    def remove_categorical_and_ordinal_variables(self):
        """
        Remove categorical and ordinal variables.
        """
        for ds in self.datasets:
            ds.remove_categorical_and_ordinal_variables()

    def append(self, other: Union[Dataset, DatasetArray, List[Union[Dataset, DatasetArray]]]) -> None:
        """
        Append new object to an DatasetArray.

        Parameters
        ----------
        other : Dataset, DatasetArray, list[Dataset, DatasetArray]
            The object to be appended to the array.
        """
        if isinstance(other, list):
            self.datasets += other
        else:
            self.datasets.append(other)

    def head(self, n: int = 10):
        """
        Get first `n` rows of each of the Dataset objects.

        Parameters
        ----------
        n : int, default=10
            Number of rows.

        Returns
        -------
        DatasetArray
            A DatasetArray object with Dataset objects with `n` first rows.
        """
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
    """
    Create a DatasetArray object from an `OpenML` suite.

    Before using this class, you should follow the procedure of configuring Authentication on the website
    `here <https://openml.github.io/openml-python/main/examples/20_basic/introduction_tutorial.html#sphx-glr-examples-20-basic-introduction-tutorial-py>`_.

    Otherwise, you should have your own API key for OpenML and pass it as a parameter.

    Parameters
    ----------
    suite_name : str, default = 'OpenML100'
        A task ID for a dataset in OpenML.
    apikey : str, optional, default=None
        An API key to OpenML (if you configured OpenML, you do not need to pass this parameter).
    name : str
        Name of the dataset array.
    verbose : bool, default=False
        Print messages during calculations.
    """

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
        for i in benchmark_suite.data:
            try:
                ds = DatasetFromOpenML(task_id=i, apikey=apikey)
                dataset_array.append(ds)
            except openml.exceptions.OpenMLServerException:
                print(f'The dataset numer {i} was not downloaded due to the server exception!')

        self.__openml_name = benchmark_suite.name if 'name' in benchmark_suite.__dict__.keys() else ''
        self.__openml_description = benchmark_suite.description if 'description' in benchmark_suite.__dict__.keys() else ''

        super().__init__(datasets=dataset_array, name=name, verbose=verbose)

        if verbose:
            print_unbuffered(f'DatasetArray from OpenML benchmark suite {suite_name} was created')

    def openml_description(self) -> str:
        """
        Description of the suite from OpenML.

        Returns
        -------
        str
            The suite description.
        """
        return "Name: " + self.__openml_name + '\n' + 'Description: ' + '\n' + self.__openml_description


class DatasetArrayFromDirectory(DatasetArray):
    """ Create a DatasetArray object by loading `*.csv` and `*.npy` files.

    The `*.npy` files are the files, which contain numpy arrays or pickled objects. They are loaded using this
    `function <https://numpy.org/doc/stable/reference/generated/numpy.load.html>`_.

    The class assumes that the last column in each file is a target column.

    Parameters
    ----------
    path : str
        A path of a directory to load files from.
    name : str, default='dataset_array'
        Name of the dataset array.
    verbose : bool, default=False
        Print messages during calculations.
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
                if f.lower().endswith('.csv') or f.lower().endswith('.npy'):
                    if f.lower().endswith('.csv'):
                        d = pd.read_csv(f)
                    elif f.lower().endswith('.npy'):
                        d = pd.DataFrame(np.load(f, allow_pickle=True))

                    y = d[d.columns[-1]]
                    X = d.drop(d.columns[-1], axis=1)
                    dataset_array.append(
                        Dataset(dataframe=X, target=y, name=os.path.splitext(filename)[0])
                    )

        if verbose:
            print_unbuffered(f'The files from {path} were loaded')

        super().__init__(datasets=dataset_array, name=name, verbose=verbose)


def load_benchmarking_set(apikey: Optional[str] = None, keep_categorical: bool = False, minimal_IR: float = 1.5,
                          minimal_n_rows: int = 1000, percent_categorical_to_remove: float = 0.75):
    """
    The function loads an example benchmarking set.

    Parameters
    ----------
    apikey : str, optional, default=None
        An API key to OpenML (if you configured OpenML, you do not need to pass this parameter).
        For details see `DatasetArrayFromOpenMLSuite` class documentation.
    keep_categorical : bool, default=False
        If True, the datasets will remain categorical variables.
    minimal_IR : float, default=1.5
        Minimal IR in the set.
    minimal_n_rows : int, default=1000
        Minimal number of rows in a Dataset.
    percent_categorical_to_remove : float, default=0.75
        Only applicable if keep_categorical=False; if categorical and nominal variables are above that number of
        all variables, the Dataset is removed.

    Returns
    -------
    DatasetArray
    """

    # Functions for changing some column types as `edgaro` assumes integers to be categorical

    def convert_to_float(ds_array, dataset_name, column_name):
        ds_array[dataset_name].data[column_name] = ds_array[dataset_name].data[column_name].astype('float64')

    def convert_columns_to_int(ds_array, dataset_name, cols):
        for col in cols:
            convert_to_float(ds_array, dataset_name, col)

    def convert_all_int_columns_to_float(ds_array, dataset_name):
        d = ds_array[dataset_name].data
        for col in list(d.select_dtypes(['uint8', 'int']).columns):
            convert_to_float(ds_array, dataset_name, col)

    exclude_global = ['nomao', 'Internet-Advertisements', 'isolet', 'sylva_agnostic', 'webpage', 'scene', 'protein_homo']

    # OpenML-CC18
    df_openml = DatasetArrayFromOpenMLSuite('OpenML-CC18', apikey=apikey)
    df_openml.remove_nans()
    df_openml.remove_non_binary_target_datasets()
    df_openml.remove_empty_datasets()
    stats = pd.DataFrame({'name': [], 'IR': [], 'nrow': [], 'ncol': []})

    for df in df_openml:
        tmp = pd.DataFrame({'name': [df.name], 'IR': [df.imbalance_ratio],
                            'nrow': [df.data.shape[0]], 'ncol': [df.data.shape[1]]})
        stats = pd.concat([stats, tmp])

    stats = stats[np.logical_and(stats.IR >= minimal_IR, stats.nrow >= minimal_n_rows)]
    stats = stats[np.logical_not(np.isin(np.array(stats.name), exclude_global))]
    df_openml = df_openml[list(stats.name)]

    if not keep_categorical:
        for i in range(len(df_openml)):
            df_openml[i].data = df_openml[i].data.convert_dtypes()

        n_cols = [ds.data.shape[1] for ds in df_openml]
        colnames = list(stats.name)
        df_openml.remove_categorical_and_ordinal_variables()
        n_cols_after = [ds.data.shape[1] for ds in df_openml]
        exclude = []

        for i in range(len(n_cols_after)):
            if n_cols_after[i] / n_cols[i] < 1 - percent_categorical_to_remove:
                exclude.append(colnames[i])

        for exc in exclude:
            colnames.remove(exc)

        df_openml = df_openml[colnames]
    else:
        convert_all_int_columns_to_float(df_openml, 'spambase')
        convert_all_int_columns_to_float(df_openml, 'qsar-biodeg')
        convert_all_int_columns_to_float(df_openml, 'credit-g')
        convert_all_int_columns_to_float(df_openml, 'adult')
        convert_all_int_columns_to_float(df_openml, 'kc1')
        convert_all_int_columns_to_float(df_openml, 'pc1')
        convert_all_int_columns_to_float(df_openml, 'pc3')
        convert_all_int_columns_to_float(df_openml, 'pc4')
        convert_all_int_columns_to_float(df_openml, 'bank-marketing')
        convert_columns_to_int(df_openml, 'churn', ['account_length', 'number_vmail_messages', 'total_day_calls',
                                                    'total_eve_calls', 'total_night_calls', 'total_intl_calls',
                                                    'number_customer_service_calls'])

    # OpenML-100
    df_openml2 = DatasetArrayFromOpenMLSuite('OpenML100', apikey=apikey)
    df_openml2.remove_nans()
    df_openml2.remove_non_binary_target_datasets()
    df_openml2.remove_empty_datasets()
    stats2 = pd.DataFrame({'name': [], 'IR': [], 'nrow': [], 'ncol': []})

    for df in df_openml2:
        tmp = pd.DataFrame({'name': [df.name], 'IR': [df.imbalance_ratio],
                            'nrow': [df.data.shape[0]], 'ncol': [df.data.shape[1]]})
        stats2 = pd.concat([stats2, tmp])

    stats2 = stats2[np.logical_and(stats2.IR >= minimal_IR, stats2.nrow >= minimal_n_rows)]
    stats2 = stats2[np.logical_not(stats2.name.isin(stats.name))]  # so as not to repeat datasets
    stats2 = stats2[np.logical_not(np.isin(np.array(stats2.name), exclude_global))]
    df_openml2 = df_openml2[list(stats2.name)]

    df_openml2['SpeedDating'].data = df_openml2['SpeedDating'].data[[col
                                                                     for col in df_openml2['SpeedDating'].data.columns
                                                                     if col == 'd_age' or not col.startswith('d_')]]

    if not keep_categorical:
        for i in range(len(df_openml2)):
            df_openml2[i].data = df_openml2[i].data.convert_dtypes()

        n_cols = [ds.data.shape[1] for ds in df_openml2]
        colnames = list(stats2.name)
        df_openml2.remove_categorical_and_ordinal_variables()
        n_cols_after = [ds.data.shape[1] for ds in df_openml2]
        exclude = []

        for i in range(len(n_cols_after)):
            if n_cols_after[i] / n_cols[i] < 1 - percent_categorical_to_remove:
                exclude.append(colnames[i])

        for exc in exclude:
            colnames.remove(exc)

        df_openml2 = df_openml2[colnames]
    else:
        convert_all_int_columns_to_float(df_openml2, 'steel-plates-fault')

    # imblearn
    datasets = fetch_datasets()
    stats3 = pd.DataFrame({'name': [], 'IR': [], 'nrow': [], 'ncol': []})
    df_tab = []

    for key in datasets.keys():
        if key != 'ozone_level':  # this dataset is already included in previous set
            ds = datasets[key]
            df = Dataset(dataframe=pd.DataFrame(ds.data), target=pd.Series(ds.target), name=ds.DESCR)
            df_tab.append(df)
            tmp = pd.DataFrame({'name': [df.name], 'IR': [df.imbalance_ratio],
                                'nrow': [df.data.shape[0]], 'ncol': [df.data.shape[1]]})
            stats3 = pd.concat([stats3, tmp])

    stats3 = stats3[np.logical_and(stats3.IR >= minimal_IR, stats3.nrow >= minimal_n_rows)]
    stats3 = stats3[np.logical_not(stats3.name.isin(stats.name))]  # so as not to repeat datasets
    stats3 = stats3[np.logical_not(stats3.name.isin(stats2.name))]  # so as not to repeat datasets
    exclude = ['optical_digits', 'satimage', 'pen_digits',
               'letter_img']  # excluding datasets made from images since they are not real tabular data.
    stats3 = stats3[np.logical_not(np.isin(np.array(stats3.name), exclude))]
    stats3 = stats3[np.logical_not(np.isin(np.array(stats3.name), exclude_global))]
    df_tab = [d for d in df_tab if d.name in list(stats3.name)]

    # `protein_homo` dataset is very big and after balancing, for example to `IR=1`, it would have gigantic sizes
    # therefore, only subset of this dataset will be included in research
    # for df in df_tab:
    #     if df.name == 'protein_homo':
    #         X, _, y, _ = train_test_split(df.data, df.target, train_size=30000, stratify=df.target, random_state=42)
    #         df.data = X
    #         df.target = y

    if not keep_categorical:
        for i in range(len(df_tab)):
            df_tab[i].data = df_tab[i].data.convert_dtypes()

        n_cols = [ds.data.shape[1] for ds in df_tab]
        colnames = list(stats3.name)

        for i in range(len(df_tab)):
            df_tab[i].remove_categorical_and_ordinal_variables()

        n_cols_after = [ds.data.shape[1] for ds in df_tab]
        exclude = []

        for i in range(len(n_cols_after)):
            if n_cols_after[i] / n_cols[i] < 1 - percent_categorical_to_remove:
                exclude.append(colnames[i])

        for exc in exclude:
            colnames.remove(exc)

        df_tab = [d for d in df_tab if d.name in colnames]

    out = DatasetArray(
        datasets=df_openml.datasets + df_openml2.datasets + df_tab,
        name='benchmarking_set',
        verbose=False
    )

    for d in out:
        d.data = d.data.astype(float)

    return out
