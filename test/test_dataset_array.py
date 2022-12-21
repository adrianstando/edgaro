import pytest
import re
import openml

from copy import deepcopy

from edgaro.data.dataset_array import DatasetArray, DatasetArrayFromOpenMLSuite, DatasetArrayFromDirectory
from edgaro.data.dataset_array import load_benchmarking_set
from edgaro.data.dataset import Dataset, DatasetFromOpenML

from .resources.objects import *


def test_dataset_array_creation():
    try:
        ds1, ds2 = Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)
        DatasetArray([ds1, ds2])
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds1, ds2', [
    (Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)),
    (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)),
    (Dataset(name_3, df_3, target_3), Dataset(name_1, df_1, target_1))
])
class TestDatasetArrayBasicProperties:
    def test_content_get_item(self, ds1, ds2):
        tab = DatasetArray([deepcopy(ds1), deepcopy(ds2)])
        assert len(tab) == 2
        assert tab[0] == ds1
        assert tab[1] == ds2
        assert tab[ds1.name] == ds1
        assert tab[ds2.name] == ds2

        assert len(tab[[ds1.name, ds2.name]]) == 2

        tmp = deepcopy(tab)
        tmp.name += '_subset'
        assert tab[[ds1.name, ds2.name]] == tmp

    def test_str_repr(self, ds1, ds2):
        tab = DatasetArray([deepcopy(ds1), deepcopy(ds2)])
        try:
            str(tab)
            repr(tab)
        except (Exception,):
            assert False

    def test_verbose(self, ds1, ds2, capsys):
        ds = DatasetArray([deepcopy(ds1), deepcopy(ds2)], verbose=True)
        captured = capsys.readouterr()
        assert captured.out.startswith(f'DatasetArray {ds.__repr__()} created\n')

    def test_dataset_array_with_dataset_arrays(self, ds1, ds2):
        tab = DatasetArray([
            deepcopy(ds1),
            DatasetArray([deepcopy(ds2), deepcopy(ds1)], name='example')
        ])

        assert len(tab) == 2
        assert tab[0] == ds1
        assert tab[1][0] == ds2
        assert tab[1][1] == ds1
        assert tab[ds1.name] == ds1
        assert tab['example'][0] == ds2
        assert tab['example'][1] == ds1

    def test_datasetarray_get_out_index(self, ds1, ds2):
        tab = DatasetArray([
            deepcopy(ds1),
            DatasetArray([deepcopy(ds2), deepcopy(ds1)], name='example')
        ])

        assert tab[['non_exist_1', 'non_exist_2']] is None

    def test_datasetarray_get_wrong_key(self, ds1, ds2):
        tab = DatasetArray([
            deepcopy(ds1),
            DatasetArray([deepcopy(ds2), deepcopy(ds1)], name='example')
        ])

        assert tab[[100, 101]] is None


def test_benchmarking_set():
    try:
        df = load_benchmarking_set(apikey=APIKEY)
    except (Exception,):
        assert False

    assert isinstance(df, DatasetArray)


class TestUniqueNames:
    @pytest.mark.parametrize('ds1, ds2', [
        (Dataset(name_1, df_1, target_1), Dataset(name_1, df_2, target_2)),
        (Dataset(name_2, df_1, target_1), Dataset(name_2, df_2, target_2)),
        (Dataset(name_2, df_1, target_1), DatasetArray([Dataset(name_2, df_2, target_2), Dataset(name_1, df_2, target_2)], name=name_2))
    ])
    def test_no_unique_names_1(self, ds1, ds2):
        with pytest.raises(Exception):
            DatasetArray([ds1, ds2])

    def test_no_unique_names_2(self):
        with pytest.raises(Exception):
            (Dataset(name_2, df_1, target_1), DatasetArray([Dataset(name_1, df_2, target_2), Dataset(name_1, df_2, target_2)], name=name_1))


class TestEqual:
    @pytest.mark.parametrize('ds1, ds2', [
        (Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)),
        (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)),
        (Dataset(name_3, df_3, target_3), Dataset(name_2, df_3, target_3)),
        (Dataset(name_3, df_3, target_3), DatasetArray([Dataset(name_2, df_3, target_3), Dataset(name_1, df_3, target_3)]))
    ])
    def test_equal(self, ds1, ds2):
        tab1 = DatasetArray([deepcopy(ds1), deepcopy(ds2)])
        tab2 = DatasetArray([deepcopy(ds1), deepcopy(ds2)])
        assert tab1 == tab2

    @pytest.mark.parametrize('ds1,ds2', [
        (Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)),
        (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)),
        (DatasetArray([Dataset(name_2, df_3, target_3), Dataset(name_1, df_3, target_3)]), Dataset(name_3, df_3, target_3)),
        (DatasetArray([Dataset(name_1, df_3, target_3), Dataset(name_2, df_3, target_3)]), Dataset(name_3, df_3, target_3))
    ])
    def test_equal_reverse(self, ds1, ds2):
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
    def test_not_equal(self, ds1, ds2):
        tab1 = DatasetArray([deepcopy(ds1)])
        tab2 = DatasetArray([deepcopy(ds2)])
        assert not tab1 == tab2

    @pytest.mark.parametrize('ds1,ds2', [
        (Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)),
        (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY))
    ])
    def test_not_equal_name(self, ds1, ds2):
        tab1 = DatasetArray([deepcopy(ds1), deepcopy(ds2)], name=ds1.name)
        tab2 = DatasetArray([deepcopy(ds2), deepcopy(ds1)], name=ds2.name)
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
    def test_not_equal_2(self, da1, da2):
        tab1 = DatasetArray(da1)
        tab2 = DatasetArray(da2)
        assert not tab1 == tab2


class TestOtherFunctions:
    @pytest.mark.parametrize('ds', [
        Dataset(name_1, df_1, target_1),
        DatasetArray([Dataset(name_2, df_2, target_2), Dataset(name_1, df_2, target_2)], name=name_2),
        DatasetArray([
            Dataset(name_2, df_2, target_2),
            Dataset(name_1, df_2, target_2),
            DatasetArray([Dataset(name_2, df_2, target_2), Dataset(name_1, df_2, target_2)], name=name_2 + 'x')
        ], name=name_2),
        Dataset(name_1, None, target_1),
        Dataset(name_1, df_1, None),
        Dataset(name_1, None, None)
    ])
    def test_str_repr(self, ds):
        try:
            str(ds)
            repr(ds)
        except (Exception,):
            assert False

    def test_remove_nans(self):
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

    def test_remove_nans_2(self):
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

    @pytest.mark.parametrize('ds1, ds2', [
        (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY))
    ])
    def test_train_test_split(self, ds1, ds2):
        tab = DatasetArray([deepcopy(ds1), deepcopy(ds2)])

        try:
            tab.train_test_split()
        except (Exception,):
            assert False

        for ds in tab:
            assert ds.was_split

    @pytest.mark.parametrize('ds1, ds2', [
        (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY))
    ])
    def test_train_test_split_verbose(self, ds1, ds2, capsys):
        tab = DatasetArray([deepcopy(ds1), deepcopy(ds2)], verbose=True)
        tab.train_test_split()

        captured = capsys.readouterr()
        assert captured.out.endswith(f'DatasetArray {tab.__repr__()} was train-test-split\n')

    @pytest.mark.parametrize('ds1, ds2', [
        (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY))
    ])
    def test_train_test_split_train_property(self, ds1, ds2):
        tab = DatasetArray([deepcopy(ds1), deepcopy(ds2)], verbose=True)
        tab.train_test_split()

        assert tab.train.name.endswith('_train')

    @pytest.mark.parametrize('ds1, ds2', [
        (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY))
    ])
    def test_train_test_split_test_property(self, ds1, ds2):
        tab = DatasetArray([deepcopy(ds1), deepcopy(ds2)], verbose=True)
        tab.train_test_split()

        assert tab.test.name.endswith('_test')

    @pytest.mark.parametrize('ds1, ds2', [
        (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY))
    ])
    def test_remove_outliers(self, ds1, ds2):
        tab = DatasetArray([deepcopy(ds1), deepcopy(ds2)], verbose=True)

        try:
            tab.remove_outliers()
        except (Exception,):
            assert False

    @pytest.mark.parametrize('ds1, ds2', [
        (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY))
    ])
    def test_head(self, ds1, ds2):
        ds = DatasetArray([ds1, ds2])
        ds_head = ds.head(2)

        for d in ds_head:
            assert d.data.shape[0] == 2
            assert d.target.shape[0] == 2


@pytest.fixture(
    scope='module',
    params=[
        suite_name_1,
        # suite_name_2 # commented because of waiting time
    ]
)
def dataset_openml(request):
    return DatasetArrayFromOpenMLSuite(request.param, apikey=APIKEY)


@pytest.mark.parametrize('task', [
    suite_name_1
])
def test_apikey_fail_none(task):
    tmp_code = openml.config.apikey
    openml.config.apikey = ''

    with pytest.raises(Exception):
        DatasetArrayFromOpenMLSuite(task, apikey=None)

    openml.config.apikey = tmp_code


@pytest.mark.parametrize('task', [
    suite_name_1
])
def test_apikey_with_apikey_empty_openml(task):
    tmp_code = openml.config.apikey
    openml.config.apikey = ''

    try:
        DatasetArrayFromOpenMLSuite(task, apikey=APIKEY)
    except (Exception,):
        assert False
    finally:
        openml.config.apikey = tmp_code


@pytest.mark.parametrize('task', [
    suite_name_1
])
def test_apikey_verbose(task, capsys):
    DatasetArrayFromOpenMLSuite(task, apikey=APIKEY, verbose=True)

    captured = capsys.readouterr()
    assert f'Benchmark suite data was downloaded for {task}\n' in captured.out
    assert f'DatasetArray from OpenML benchmark suite {task} was created' in captured.out


class TestOpenMLSuite:
    def test_print_description_from_openml_suite(self, dataset_openml):
        ds = deepcopy(dataset_openml)
        out = ds.openml_description()
        pattern = '^Name: .* Description: .*$'
        assert re.match(pattern, out.replace('\n', ' '))

    def test_remove_non_binary_target_datasets(self, dataset_openml):
        # These OpenML suites contain also non-binary target datasets
        da = deepcopy(dataset_openml)
        length_1 = len(da)

        da.remove_non_binary_target_datasets()
        length_2 = len(da)

        assert length_2 < length_1

    def test_remove_non_binary_target_verbose(self, dataset_openml, capsys):
        # These OpenML suites contain also non-binary target datasets
        da = deepcopy(dataset_openml)
        da.verbose = True
        da.remove_non_binary_target_datasets()

        captured = capsys.readouterr()
        assert captured.out.startswith(f'Non binary datasets were removed from DatasetArray {da.__repr__()}\n')


class TestRemoveNonBinary:
    @pytest.mark.parametrize('ds,expected_binary_n', [
        ([Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_1_fake), Dataset(name_3, df_1, target_1_fake)], 1),
        ([Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_1), Dataset(name_3, df_1, target_1_fake)], 2),
        ([Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_1), Dataset(name_3, df_1, target_1)], 3),
        ([Dataset(name_1, df_1, target_1_fake), Dataset(name_2, df_1, target_1_fake), Dataset(name_3, df_1, target_1_fake)], 0)
    ])
    def test_remove_non_binary_target_datasets(self, ds, expected_binary_n):
        da = DatasetArray(ds)
        da = deepcopy(da)
        length_1 = len(da)

        da.remove_non_binary_target_datasets()
        length_2 = len(da)

        assert length_2 <= length_1
        assert length_2 == expected_binary_n

    def test_remove_non_binary_target_datasets_nested_to_zero(self):
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

    def test_remove_non_binary_target_datasets_non_nested(self):
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

    def test_remove_non_binary_target_datasets_not_nested_and_nested(self):
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

    def test_remove_non_binary_target_datasets_only_nested(self):
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


class TestRemoveEmpty:
    def test_remove_empty_datasets(self):
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

    def test_remove_empty_datasets_only_nested(self):
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

    def test_remove_empty_datasets_verbose(self, capsys):
        da = DatasetArray([
            Dataset(name_1 + '_1', df_1, target_1),
            Dataset(name_1 + '_2', df_1, target_1),
            DatasetArray([
                Dataset(name_4_nans + '_0', None, None),
                Dataset(name_1 + '_1', df_1, target_1)
            ])
        ])

        da = deepcopy(da)
        da.verbose = True
        da.remove_empty_datasets()

        captured = capsys.readouterr()
        assert captured.out.startswith(f'Empty datasets were removed from DatasetArray {da.__repr__()}\n')


@pytest.mark.parametrize('ds1,ds2', [
    (DatasetArray([Dataset(name_1, df_1, target_1)]), Dataset(name_2, df_2, target_2)),
    (DatasetArray([Dataset(name_1, df_1, target_1)]), DatasetArray([Dataset(name_2, df_3, target_3), Dataset(name_1, df_3, target_3)]))
])
def test_append(ds1, ds2):
    ds1 = deepcopy(ds1)
    ds2 = deepcopy(ds2)

    len_1 = len(ds1)
    ds1.append(ds2)
    assert len(ds1) == len_1 + 1


@pytest.mark.parametrize('ds1,ds2', [
    (DatasetArray([Dataset(name_1, df_1, target_1)]), Dataset(name_2, df_2, target_2)),
    (DatasetArray([Dataset(name_1, df_1, target_1)]), DatasetArray([Dataset(name_2, df_3, target_3), Dataset(name_1, df_3, target_3)]))
])
def test_append_list(ds1, ds2):
    ds1 = deepcopy(ds1)
    ds2 = deepcopy(ds2)
    ds3 = deepcopy(ds2)
    ds3.name += 'xx'

    len_1 = len(ds1)
    ds1.append([ds2, ds3])
    assert len(ds1) == len_1 + 2


class TestDatasetArrayFromDirectory:
    def test_create_array(self):
        try:
            DatasetArrayFromDirectory('./resources')
        except (Exception,):
            assert False

    def test_create_array_verbose(self, capsys):
        path = './resources'
        DatasetArrayFromDirectory(path, verbose=True)
        captured = capsys.readouterr()
        assert f'The files from {path} were loaded\n' in captured.out

    def test_non_existing_path(self):
        with pytest.raises(Exception):
            DatasetArrayFromDirectory('./non-existing')

    def test_not_dir(self):
        with pytest.raises(Exception):
            DatasetArrayFromDirectory('./resources/data_1.csv')
