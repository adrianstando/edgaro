import pytest
from EDGAR.data.DatasetArray import DatasetArray
from EDGAR.data.Dataset import Dataset, DatasetFromOpenML
from copy import deepcopy
from .resources.objects import *


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)),
    (DatasetFromOpenML(task_id=3), DatasetFromOpenML(task_id=36))
])
def test_dataset_array(ds1, ds2):
    tab = DatasetArray([deepcopy(ds1), deepcopy(ds2)])
    assert len(tab) == 2
    assert tab[0] == ds1
    assert tab[1] == ds2
    assert tab[ds1.name] == ds1
    assert tab[ds2.name] == ds2


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_1, df_2, target_2)),
    (Dataset(name_2, df_1, target_1), Dataset(name_2, df_2, target_2))
])
def test_no_unique_names(ds1, ds2):
    try:
        DatasetArray([ds1, ds2])
        assert False
    except (Exception,):
        pass


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)),
    (DatasetFromOpenML(task_id=3), DatasetFromOpenML(task_id=36))
])
def test_equal(ds1, ds2):
    tab1 = DatasetArray([deepcopy(ds1), deepcopy(ds2)])
    tab2 = DatasetArray([deepcopy(ds1), deepcopy(ds2)])
    assert tab1 == tab2


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)),
    (DatasetFromOpenML(task_id=3), DatasetFromOpenML(task_id=36))
])
def test_equal_reverse(ds1, ds2):
    tab1 = DatasetArray([deepcopy(ds1), deepcopy(ds2)])
    tab2 = DatasetArray([deepcopy(ds2), deepcopy(ds1)])
    assert tab1 == tab2


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_1, df_1, target_2)),
    (Dataset(name_1, df_1, target_1), Dataset(name_1, df_2, target_1)),
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_1)),
])
def test_not_equal(ds1, ds2):
    tab1 = DatasetArray([deepcopy(ds1)])
    tab2 = DatasetArray([deepcopy(ds2)])
    assert not tab1 == tab2


@pytest.mark.parametrize('da1,da2', [
    (
        [Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)],
        [Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_1)]
    ),
    (
        [Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)],
        [Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_2)]
    ),
    (
        [Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)],
        [Dataset(name_1, df_1, target_1), Dataset(name_1 + '_extra', df_2, target_2)],
    ),
    # reverse order
    (
        [Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)],
        [Dataset(name_1, df_2, target_2), Dataset(name_1 + '_extra', df_1, target_1)],
    ),
    (
        [Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)],
        [Dataset(name_2, df_1, target_2), Dataset(name_1, df_1, target_1)]
    ),
    (
        [Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2), Dataset(name_1 + '_extra', df_1, target_1)],
        [Dataset(name_2, df_1, target_2), Dataset(name_1, df_1, target_1)]
    ),
])
def test_not_equal_2(da1, da2):
    tab1 = DatasetArray(da1)
    tab2 = DatasetArray(da2)
    assert not tab1 == tab2
