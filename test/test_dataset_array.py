import pytest
from EDGAR.data.DatasetArray import DatasetArray, DatasetArrayFromOpenMLSuite
from EDGAR.data.Dataset import Dataset, DatasetFromOpenML
from copy import deepcopy
import re
from .resources.objects import *


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)),
    (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)),
    (Dataset(name_3, df_3, target_3), Dataset(name_1, df_1, target_1))
])
def test_dataset_array(ds1, ds2):
    tab = DatasetArray([deepcopy(ds1), deepcopy(ds2)])
    assert len(tab) == 2
    assert tab[0] == ds1
    assert tab[1] == ds2
    assert tab[ds1.name] == ds1
    assert tab[ds2.name] == ds2

    assert len(tab[[ds1.name, ds2.name]]) == 2
    assert tab[[ds1.name, ds2.name]] == tab


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)),
    (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)),
    (Dataset(name_3, df_3, target_3), Dataset(name_1, df_1, target_1))
])
def test_dataset_array_with_dataset_arrays(ds1, ds2):
    tab = DatasetArray([deepcopy(ds1), DatasetArray([deepcopy(ds2), deepcopy(ds1)], name='example')])
    assert len(tab) == 2
    assert tab[0] == ds1
    assert tab[1][0] == ds2
    assert tab[1][1] == ds1
    assert tab[ds1.name] == ds1
    assert tab['example'][0] == ds2
    assert tab['example'][1] == ds1


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_1, df_2, target_2)),
    (Dataset(name_2, df_1, target_1), Dataset(name_2, df_2, target_2)),
    (Dataset(name_2, df_1, target_1), DatasetArray([Dataset(name_2, df_2, target_2), Dataset(name_1, df_2, target_2)], name=name_2))
])
def test_no_unique_names(ds1, ds2):
    try:
        DatasetArray([ds1, ds2])
        assert False
    except (Exception,):
        pass


def test_no_unique_names_array():
    try:
        DatasetArray([Dataset(name_2, df_1, target_1), DatasetArray([Dataset(name_1, df_2, target_2), Dataset(name_1, df_2, target_2)], name=name_1)])
        assert False
    except (Exception,):
        pass


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)),
    (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)),
    (Dataset(name_3, df_3, target_3), Dataset(name_2, df_3, target_3)),
    (Dataset(name_3, df_3, target_3), DatasetArray([Dataset(name_2, df_3, target_3), Dataset(name_1, df_3, target_3)]))
])
def test_equal(ds1, ds2):
    tab1 = DatasetArray([deepcopy(ds1), deepcopy(ds2)])
    tab2 = DatasetArray([deepcopy(ds1), deepcopy(ds2)])
    assert tab1 == tab2


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)),
    (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)),
    (DatasetArray([Dataset(name_2, df_3, target_3), Dataset(name_1, df_3, target_3)]), Dataset(name_3, df_3, target_3)),
    (DatasetArray([Dataset(name_1, df_3, target_3), Dataset(name_2, df_3, target_3)]), Dataset(name_3, df_3, target_3))
])
def test_equal_reverse(ds1, ds2):
    tab1 = DatasetArray([deepcopy(ds1), deepcopy(ds2)])
    tab2 = DatasetArray([deepcopy(ds2), deepcopy(ds1)])
    assert tab1 == tab2


@pytest.mark.parametrize('ds1,ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_1, df_1, target_2)),
    (Dataset(name_1, df_1, target_1), Dataset(name_1, df_2, target_1)),
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_1)),
    (
            DatasetArray([
                DatasetArray([Dataset(name_1, df_2, target_1), Dataset(name_2, df_3, target_3)]),
                Dataset(name_2, df_3, target_3)
            ]),
            DatasetArray([
                DatasetArray([Dataset(name_1, df_2, target_2), Dataset(name_2, df_3, target_3)]),
                Dataset(name_2, df_3, target_3)
            ])
    )
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
    (
        [Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2), DatasetArray([Dataset(name_2, df_3, target_3), Dataset(name_1, df_3, target_3)], name='xx')],
        [Dataset(name_2, df_1, target_2), Dataset(name_1, df_1, target_1), DatasetArray([Dataset(name_2, df_3, target_3), Dataset(name_1, df_3, target_3)], name='xxyy')]
    )
])
def test_not_equal_2(da1, da2):
    tab1 = DatasetArray(da1)
    tab2 = DatasetArray(da2)
    assert not tab1 == tab2


def test_remove_nans():
    da = DatasetArray([
        Dataset(name_4_nans + '_0', df_4_nans, target_4_nans),
        Dataset(name_4_nans + '_1', df_4_nans, target_4_nans)
    ])
    da = deepcopy(da)
    da.remove_nans()

    assert da[0].data.shape == (1, 2)
    assert da[0].target.shape == (1,)
    assert da[1].data.shape == (1, 2)
    assert da[1].target.shape == (1,)


def test_remove_nans_2():
    da = DatasetArray([
        Dataset(name_4_nans + '_0', df_4_nans, target_4_nans),
        Dataset(name_4_nans + '_1', df_4_nans, target_4_nans),
        DatasetArray([
            Dataset(name_4_nans + '_0', df_4_nans, target_4_nans),
            Dataset(name_4_nans + '_1', df_4_nans, target_4_nans),
        ])
    ])
    da = deepcopy(da)
    da.remove_nans()

    assert da[0].data.shape == (1, 2)
    assert da[0].target.shape == (1,)
    assert da[1].data.shape == (1, 2)
    assert da[1].target.shape == (1,)

    assert len(da[2]) == 2
    assert da[2][0].data.shape == (1, 2)
    assert da[2][0].target.shape == (1,)
    assert da[2][1].data.shape == (1, 2)
    assert da[2][1].target.shape == (1,)


@pytest.mark.parametrize('suite_name', [
    suite_name_1,
    # suite_name_2 # commented since it's a time-consuming test
])
class TestOpenMLSuite:
    openml_suite = None

    def test_dataset_array_from_openml_suite(self, suite_name):
        try:
            if self.openml_suite is None:
                self.openml_suite = DatasetArrayFromOpenMLSuite(suite_name=suite_name, apikey=APIKEY)
        except (Exception,):
            assert False

    def test_print_description_from_openml_suite(self, suite_name, capsys):
        if self.openml_suite is None:
            self.test_dataset_array_from_openml_suite(suite_name)

        if self.openml_suite is None:
            assert False

        # clear stdout
        capsys.readouterr()

        ds = deepcopy(self.openml_suite)
        ds.print_openml_description()
        captured = capsys.readouterr()
        pattern = '^Name: .* Description: .*$'
        assert re.match(pattern, captured.out.replace('\n', ' '))

    def test_remove_non_binary_target_datasets(self, suite_name):
        if self.openml_suite is None:
            self.test_dataset_array_from_openml_suite(suite_name)

        if self.openml_suite is None:
            assert False

        # These OpenML suites contain also non-binary target datasets
        da = deepcopy(self.openml_suite)
        length_1 = len(da)

        da.remove_non_binary_target_datasets()
        length_2 = len(da)

        assert length_2 < length_1


@pytest.mark.parametrize('ds,expected_binary_n', [
    ([Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_1_fake), Dataset(name_3, df_1, target_1_fake)], 1),
    ([Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_1), Dataset(name_3, df_1, target_1_fake)], 2),
    ([Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_1), Dataset(name_3, df_1, target_1)], 3),
    ([Dataset(name_1, df_1, target_1_fake), Dataset(name_2, df_1, target_1_fake), Dataset(name_3, df_1, target_1_fake)], 0)
])
def test_remove_non_binary_target_datasets_2(ds, expected_binary_n):
    # These OpenML suites contain also non-binary target datasets
    da = DatasetArray(ds)
    da = deepcopy(da)
    length_1 = len(da)

    da.remove_non_binary_target_datasets()
    length_2 = len(da)

    assert length_2 <= length_1
    assert length_2 == expected_binary_n


def test_remove_non_binary_target_datasets_3_1():
    da = DatasetArray([
        Dataset(name_4_nans + '_0', df_4_nans, target_1_fake),
        Dataset(name_4_nans + '_1', df_4_nans, target_4_nans),
        DatasetArray([
            Dataset(name_4_nans + '_0', df_4_nans, target_4_nans),
            Dataset(name_4_nans + '_1', df_4_nans, target_4_nans),
        ])
    ])

    assert len(da) == 3

    da = deepcopy(da)
    da.remove_non_binary_target_datasets()

    assert len(da) == 0


def test_remove_non_binary_target_datasets_3_2():
    da = DatasetArray([
        Dataset(name_4_nans + '_0', df_4_nans, target_1),
        Dataset(name_4_nans + '_1', df_4_nans, target_4_nans),
        DatasetArray([
            Dataset(name_4_nans + '_0', df_4_nans, target_1),
            Dataset(name_4_nans + '_1', df_4_nans, target_1),
        ])
    ])

    assert len(da) == 3

    da = deepcopy(da)
    da.remove_non_binary_target_datasets()

    assert len(da) == 2
    assert len(da[1]) == 2


def test_remove_non_binary_target_datasets_3_3():
    da = DatasetArray([
        Dataset(name_4_nans + '_0', df_4_nans, target_1),
        Dataset(name_4_nans + '_1', df_4_nans, target_4_nans),
        DatasetArray([
            Dataset(name_4_nans + '_0', df_4_nans, target_1),
            Dataset(name_4_nans + '_1', df_4_nans, target_4_nans),
        ])
    ])

    assert len(da) == 3

    da = deepcopy(da)
    da.remove_non_binary_target_datasets()

    assert len(da) == 2
    assert len(da[1]) == 1


def test_remove_non_binary_target_datasets_4():
    da = DatasetArray([
        Dataset(name_1 + '_1', df_1, target_1),
        Dataset(name_1 + '_2', df_1, target_1),
        DatasetArray([
            Dataset(name_4_nans + '_0', df_4_nans, target_1_fake),
            Dataset(name_1 + '_1', df_1, target_1)
        ])
    ])

    assert len(da) == 3
    assert len(da[2]) == 2

    da = deepcopy(da)
    da.remove_non_binary_target_datasets()

    assert len(da) == 3
    assert len(da[2]) == 1


def test_remove_empty_datasets():
    da = DatasetArray([
        Dataset(name_1 + '_1', None, None),
        Dataset(name_1 + '_2', df_1, target_1),
        DatasetArray([
            Dataset(name_4_nans + '_0', df_4_nans, target_1_fake),
            Dataset(name_1 + '_1', df_1, target_1)
        ])
    ])

    assert len(da) == 3
    assert len(da[2]) == 2

    da = deepcopy(da)
    da.remove_non_binary_target_datasets()

    assert len(da) == 2
    assert len(da[1]) == 1


def test_remove_empty_datasets_2():
    da = DatasetArray([
        Dataset(name_1 + '_1', df_1, target_1),
        Dataset(name_1 + '_2', df_1, target_1),
        DatasetArray([
            Dataset(name_4_nans + '_0', None, None),
            Dataset(name_1 + '_1', df_1, target_1)
        ])
    ])

    assert len(da) == 3
    assert len(da[2]) == 2

    da = deepcopy(da)
    da.remove_non_binary_target_datasets()

    assert len(da) == 3
    assert len(da[2]) == 1
