import pandas as pd
import pytest
import re

from copy import deepcopy

from EDGAR.data.dataset import Dataset, DatasetFromCSV, DatasetFromOpenML

from .resources.objects import *


class TestCreateObjects:
    def test_dataset(self):
        try:
            Dataset(name_1, df_1, target_1)
        except (Exception,):
            assert False

    def test_csv_1(self):
        try:
            ds = DatasetFromCSV(path=example_path)
        except (Exception,):
            assert False

        assert ds.target.equals(pd.Series(pd.read_csv(example_path).iloc[:, -1], name='target'))

    def test_csv_2(self):
        try:
            DatasetFromCSV(path=example_path, target=example_target)
        except (Exception,):
            assert False

    def test_openml(self):
        try:
            DatasetFromOpenML(task_id_1, apikey=APIKEY)
        except (Exception,):
            assert False


@pytest.mark.parametrize('name, df, target, expected_IR, target_fake', [
    (name_1, df_1, target_1, IR_1, target_1_fake),
    (name_2, df_2, target_2, IR_2, target_2_fake)
])
class TestDatasetBasicProperties:
    def test_properties_not_none(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, df, target)
        assert ds.data is not None
        assert ds.target is not None
        assert ds.name is not None

    def test_properties_value(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, df, target)
        assert ds.name == name
        assert ds.data.equals(df)
        assert ds.target.equals(target)

    def test_str_repr(self, name, df, target, expected_IR, target_fake):
        try:
            ds = Dataset(name, df, target)
            str(ds)
            repr(ds)
        except (Exception,):
            assert False

    def test_imbalance_ratio(self, name, df, target, expected_IR, target_fake):
        try:
            ds = Dataset(name, df, target)
            IR = ds.imbalance_ratio
        except (Exception,):
            assert False

        assert IR == expected_IR

    def test_generate_report(self, name, df, target, expected_IR, target_fake):
        try:
            ds = Dataset(name, df, target)
            ds.generate_report(output_path=os.path.join('resources', 'out.html'), minimal=True)
        except (Exception,):
            assert False

    def test_check_binary(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, df, target)
        assert ds.check_binary_classification()

    def test_check_no_binary(self, name, df, target, expected_IR, target_fake):
        tmp = Dataset(name, df, target_fake)
        assert not tmp.check_binary_classification()

    def test_equal(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, df, target)
        tmp = Dataset(name, df, target)
        assert ds == tmp

    def test_train_test_split_before(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, df, target)
        assert not ds.was_split

        try:
            ds.data = df
            ds.target = target
        except (Exception,):
            assert False

        assert isinstance(ds.data, pd.DataFrame)
        assert isinstance(ds.target, pd.Series)

        with pytest.raises(Exception):
            out = ds.train

        with pytest.raises(Exception):
            out = ds.test

    def test_train_test_split_after(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, pd.concat([df for _ in range(3)]), pd.concat([target for _ in range(3)]))

        try:
            ds.train_test_split(test_size=0.3, random_state=42)
            train = ds.train
            test = ds.test
        except (Exception,):
            assert False

        assert isinstance(train, Dataset)
        assert isinstance(test, Dataset)

        assert isinstance(train.data, pd.DataFrame)
        assert isinstance(train.target, pd.Series)

        assert isinstance(test.data, pd.DataFrame)
        assert isinstance(test.target, pd.Series)

        with pytest.raises(Exception):
            ds.data = df

        with pytest.raises(Exception):
            ds.target = target

        assert ds.was_split

    def test_train_test_split_double_raise_exception(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, pd.concat([df for _ in range(3)]), pd.concat([target for _ in range(3)]))

        try:
            ds.train_test_split(test_size=0.3, random_state=42)
            train = ds.train
            test = ds.test
        except (Exception,):
            assert False

        with pytest.raises(Exception):
            ds.train_test_split(test_size=0.3, random_state=42)


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_1, df_1, target_2)),
    (Dataset(name_1, df_1, target_1), Dataset(name_1, df_2, target_1)),
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_1)),
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_1)),
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_2))
])
def test_dataset_not_equal(ds1, ds2):
    assert not ds1 == ds2


@pytest.fixture(scope='module', params=[example_path])
def dataset_csv(request):
    return DatasetFromCSV(request.param)


class TestCSV:
    def test_calculate_imbalance_ratio(self, dataset_csv):
        try:
            IR = dataset_csv.imbalance_ratio
        except (Exception,):
            assert False

    def test_check_binary(self, dataset_csv):
        assert dataset_csv.check_binary_classification()


@pytest.fixture(scope='module', params=[task_id_1, task_id_2])
def dataset_openml(request):
    return DatasetFromOpenML(request.param, apikey=APIKEY)


class TestOpenML:
    def test_calculate_imbalance_ratio(self, dataset_openml):
        try:
            IR = dataset_openml.imbalance_ratio
        except (Exception,):
            assert False

    def test_check_binary(self, dataset_openml):
        assert dataset_openml.check_binary_classification()

    def test_description_openml(self, dataset_openml):
        out = dataset_openml.openml_description()
        pattern = '^Name: .* Description: .*$'
        assert re.match(pattern, out.replace('\n', ' '))


@pytest.mark.parametrize('name, df, target, col_thresh_nan_remove, data_shape_nan, target_shape_nan', [
    (name_4_nans, deepcopy(df_4_nans), deepcopy(target_4_nans), 0.9, (1, 2), (1,)),
    (name_4_nans, deepcopy(df_4_nans), deepcopy(target_2), 1, (2, 2), (2,))
])
def test_output_shape(name, df, target, col_thresh_nan_remove, data_shape_nan, target_shape_nan):
    # noinspection PyTypeChecker
    ds = Dataset(name, df, target)
    ds.remove_nans(col_thresh=col_thresh_nan_remove)

    assert ds.data.shape == data_shape_nan
    assert ds.target.shape == target_shape_nan


@pytest.mark.parametrize('name, df, target, n_rows', [
    (name_1, deepcopy(df_1), deepcopy(target_1), 3),
    (name_2, deepcopy(df_2), deepcopy(target_2), 3)
])
def test_remove_outliers(name, df, target, n_rows):
    ds = Dataset(name, df, target)
    ds.data = pd.concat([ds.data, pd.DataFrame({'a': [0, 0, 0, 1000], 'b': ['a', 'b', 'p', 'q']})])
    ds.target = pd.concat([ds.target, pd.Series([0, 0, 1, 1], name='target')])

    assert ds.data.shape[0] == n_rows + 4
    assert ds.target.shape[0] == n_rows + 4

    ds.remove_outliers(n_std=1)

    assert ds.data.shape[0] == n_rows + 3
    assert ds.target.shape[0] == n_rows + 3
