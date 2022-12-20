import numpy as np
import pandas as pd
import pytest
import re
import openml

from copy import deepcopy

from edgaro.data.dataset import Dataset, DatasetFromCSV, DatasetFromOpenML, load_mammography

from .resources.objects import *


class TestCreateObjects:
    def test_dataset(self):
        try:
            Dataset(name_1, df_1, target_1)
        except (Exception,):
            assert False

    def test_dataset_fail_df(self):
        with pytest.raises(Exception):
            Dataset(name_1, pd.concat([df_1, df_1]), target_1)

    def test_dataset_fail_target(self):
        with pytest.raises(Exception):
            Dataset(name_1, df_1, pd.concat([target_1, target_1]))

    def test_dataset_verbose(self, capsys):
        ds = Dataset(name_1, df_1, target_1, verbose=True)
        captured = capsys.readouterr()
        assert captured.out == f'Dataset {ds.__repr__()} created\n'

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

    def test_csv_verbose(self, capsys):
        ds = DatasetFromCSV(path=example_path, target=example_target, verbose=True)
        captured = capsys.readouterr()
        assert captured.out.startswith(f'Data from {example_path} file is loaded\n')

    def test_openml(self):
        try:
            DatasetFromOpenML(task_id_1, apikey=APIKEY)
        except (Exception,):
            assert False

    def test_mammography(self):
        try:
            df = load_mammography()
        except (Exception,):
            assert False

        assert isinstance(df, Dataset)


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

    def test_imbalance_ratio_three_class_target(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, df, target_fake)
        with pytest.raises(Exception):
            IR = ds.imbalance_ratio

    def test_imbalance_ratio_empty_target(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, df, None)
        assert ds.imbalance_ratio == 0

    def test_imbalance_ratio_one_class_target(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, df, pd.Series(np.ones(df.shape[0])))
        assert ds.imbalance_ratio == 0

    def test_generate_report_path(self, name, df, target, expected_IR, target_fake):
        try:
            ds = Dataset(name, df, target)
            ds.generate_report(output_path=os.path.join('resources', 'out.html'), minimal=True)
        except (Exception,):
            assert False

    def test_generate_report_fail_none(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, None, None)
        with pytest.raises(Exception):
            ds.generate_report(output_path=os.path.join('resources', 'out.html'), minimal=True)

    def test_generate_report_only_data(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, df, None)
        try:
            ds.generate_report(output_path=os.path.join('resources', 'out.html'), minimal=True)
        except (Exception,):
            assert False

    def test_generate_report_only_target(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, None, target)
        try:
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

    def test_train_test_split_verbose(self, name, df, target, expected_IR, target_fake, capsys):
        ds = Dataset(name, pd.concat([df for _ in range(3)]), pd.concat([target for _ in range(3)]), verbose=True)
        ds.train_test_split(test_size=0.3, random_state=42)

        captured = capsys.readouterr()
        assert captured.out.endswith(f'Dataset {ds.__repr__()} was train-test-split\n')

    def test_custom_train_test_split(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, None, None)
        train = Dataset(name, pd.concat([df for _ in range(3)]), pd.concat([target for _ in range(3)]))
        test = Dataset(name, pd.concat([df for _ in range(1)]), pd.concat([target for _ in range(1)]))

        ds.custom_train_test_split(
            train=train,
            test=test
        )

        assert ds.train == train
        assert ds.test == test

    def test_head(self, name, df, target, expected_IR, target_fake):
        ds = Dataset(name, df, target)
        ds_head = ds.head(2)

        assert ds_head.data.shape[0] == 2
        assert ds_head.target.shape[0] == 2


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


@pytest.mark.parametrize('task', [
    task_id_1
])
def test_dataset_openml_verbose(task, capsys):
    ds = DatasetFromOpenML(task, apikey=APIKEY, verbose=True)
    captured = capsys.readouterr()
    assert captured.out.startswith(f'Dataset from OpenML with id {str(task)} was downloaded\n')


@pytest.mark.parametrize('task', [
    task_id_1
])
def test_apikey_fail_none(task):
    tmp_code = openml.config.apikey
    openml.config.apikey = ''

    with pytest.raises(Exception):
        DatasetFromOpenML(task, apikey=None)

    openml.config.apikey = tmp_code


@pytest.mark.parametrize('task', [
    task_id_1
])
def test_apikey_with_apikey_empty_openml(task):
    tmp_code = openml.config.apikey
    openml.config.apikey = ''

    try:
        DatasetFromOpenML(task, apikey=APIKEY)
    except (Exception,):
        assert False
    finally:
        openml.config.apikey = tmp_code


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
