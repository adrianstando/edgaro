import pytest
from EDGAR.data.Dataset import Dataset, DatasetFromCSV, DatasetFromOpenML
import re
from .resources.objects import *


# TODO:
# check second method to instantiate with openml_dataset argument

def test_dataset_properties():
    ds = Dataset(name_1, df_1, target_1)
    assert ds.name == name_1
    assert ds.data.equals(df_1)
    assert ds.target.equals(target_1)


@pytest.mark.parametrize('task_id', [
    task_id_1,
    task_id_2
])
def test_create_dataset_openml(task_id):
    try:
        DatasetFromOpenML(task_id=task_id, apikey=APIKEY)
    except (Exception,):
        assert False


@pytest.mark.parametrize('path', [
    example_path
])
def test_create_dataset_csv(path):
    try:
        DatasetFromCSV(path, target=example_target)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1),
    Dataset(name_2, df_2, target_2),
    Dataset(name_3, df_3, target_3),
    DatasetFromCSV(path=example_path, target=example_target),
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY)
])
def test_dataset(ds):
    ds = Dataset(name_1, df_1, target_1)
    assert ds.data is not None
    assert ds.target is not None
    assert ds.name is not None


@pytest.mark.parametrize('ds,expected', [
    (Dataset(name_1, df_1, target_1), 2 / 1),
    (Dataset(name_2, df_2, target_2), 2 / 1),
    (DatasetFromCSV(example_path, example_target), 2 / 2)
])
def test_imbalance_ratio(ds, expected):
    assert ds.imbalance_ratio() == expected


@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1),
    Dataset(name_2, df_2, target_2),
    Dataset(name_3, df_3, target_3),
    DatasetFromCSV(example_path, example_target),
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY)
])
def test_exception_imbalance_ratio(ds):
    try:
        ds.imbalance_ratio()
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1),
    Dataset(name_2, df_2, target_2),
    Dataset(name_3, df_3, target_3),
    DatasetFromCSV(path=example_path, target=example_target),
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY)
])
def test_profiling(ds):
    try:
        ds.generate_report()
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1),
    Dataset(name_2, df_2, target_2),
    Dataset(name_3, df_3, target_3),
    DatasetFromCSV(path=example_path, target=example_target),
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY)
])
def test_csv_check_binary(ds):
    assert ds.check_binary_classification()


@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1_fake),
    Dataset(name_2, df_2, target_1_fake)
])
def test_csv_check_no_binary(ds):
    assert not ds.check_binary_classification()


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_1, df_1, target_1)),
    (DatasetFromCSV(path=example_path, target=example_target), DatasetFromCSV(path=example_path, target=example_target)),
    (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY))
])
def test_csv_equal(ds1, ds2):
    assert ds1 == ds2


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_1, df_1, target_2)),
    (Dataset(name_1, df_1, target_1), Dataset(name_1, df_2, target_1)),
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_1)),
    (DatasetFromCSV(path=example_path, target=example_target, name='x'), DatasetFromCSV(path=example_path, target=example_target, name='xx')),
    (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY))
])
def test_csv_not_equal(ds1, ds2):
    assert not ds1 == ds2


def test_remove_nans():
    ds = Dataset(name_4_nans, df_4_nans, target_4_nans)
    ds.remove_nans()

    assert ds.data.shape == (1, 2)
    assert ds.target.shape == (1,)


@pytest.mark.parametrize('ds', [
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)
])
def test_print_description_openml(ds, capsys):
    ds.print_openml_description()
    captured = capsys.readouterr()
    pattern = '^Name: .* Description: .*$'
    assert re.match(pattern, captured.out.replace('\n', ' '))

