import pandas as pd
import pytest

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from copy import deepcopy

from edgaro.balancing.transformer import TransformerFromIMBLEARN, RandomUnderSampler as RandomUnderSampler_EDAGR, \
    RandomOverSampler as RandomOverSampler_EDGAR, Transformer
from edgaro.base.base_transformer import BaseTransformer
from edgaro.data.dataset import Dataset, DatasetFromOpenML
from edgaro.data.dataset_array import DatasetArray
from edgaro.balancing.transformer_array import TransformerArray

from .resources.objects import *

ds_openml_1 = DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)
ds_openml_2 = deepcopy(ds_openml_1)
ds_openml_2.name += '__xx'


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1),
    Dataset(name_2, df_2, target_2),
    DatasetArray([Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)]),
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2),
                  DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)])]),
    DatasetArray([DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)], name='x'),
                  DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)])])
])
def test_transformer_array(imblearn_sampler, ds):
    try:
        transformer = TransformerFromIMBLEARN(imblearn_sampler)
        array = TransformerArray(transformer)
        array.fit(ds)
        array.transform(ds)
        str(transformer)
        repr(transformer)
        str(array)
        repr(array)
    except (Exception,):
        assert False


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1),
    Dataset(name_2, df_2, target_2),
    DatasetArray([Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)]),
])
def test_transform_keep_original(imblearn_sampler, ds):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer, keep_original_dataset=True)
    array.fit(ds)
    transform_true = array.transform(ds)

    array.keep_original_dataset = False
    transform_false = array.transform(ds)

    assert len(transform_true) - 1 == len(transform_false)


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1),
    Dataset(name_2, df_2, target_2),
])
@pytest.mark.parametrize('sufix', [
    '_transformed',
    '_example_sufix'
])
def test_transformer_sufix(imblearn_sampler, ds, sufix):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer, dataset_suffixes=sufix)
    array.fit(ds)
    out = array.transform(ds)
    for dataset in out.datasets:
        assert dataset.name == ds.name + sufix


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)])
])
@pytest.mark.parametrize('sufix', [
    '_transformed',
    '_example_sufix'
])
def test_transformer_sufix_2_datasetarray(imblearn_sampler, ds, sufix):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer, dataset_suffixes=sufix)
    array.fit(ds)
    out = array.transform(ds)

    s = sufix if isinstance(sufix, str) else '_transformed_array'
    assert out.name == ds.name + s

    expected_names = [dataset.name + sufix for dataset in ds.datasets]
    assert len(expected_names) == len(out)
    for k in range(len(out)):
        assert out[k].name == expected_names[k]


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_2, df_2, target_2)])
])
@pytest.mark.parametrize('sufix', [
    ['_transformed'],
    ['_example_sufix']
])
def test_transformer_sufix_tab_datasetarray(imblearn_sampler, ds, sufix):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer, dataset_suffixes=sufix)
    array.fit(ds)
    out = array.transform(ds)

    expected_names = [ds.datasets[i].name + sufix[i] for i in range(len(ds.datasets))]
    assert len(expected_names) == len(out)

    for k in range(len(out)):
        assert out[k].name == expected_names[k]


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)]),
    DatasetArray([Dataset(name_1, df_1, target_1), deepcopy(ds_openml_2)])
])
@pytest.mark.parametrize('sufix', [
    ['_transformed_0', '_transformed_1'],
    ['_example_sufix_0', '_example_sufix_1']
])
def test_transformer_sufix_tab_2_datasetarray(imblearn_sampler, ds, sufix):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer, dataset_suffixes=sufix)
    array.fit(ds)
    out = array.transform(ds)

    expected_names = [ds.datasets[i].name + sufix[i] for i in range(len(ds.datasets))]
    assert len(expected_names) == len(out)

    for k in range(len(out)):
        assert out[k].name == expected_names[k]


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray(
        [Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_1), Dataset(name_1 + '3', df_1, target_1)])
])
@pytest.mark.parametrize('sufix', [
    ['_transformed_0', '_transformed_1'],
    ['_example_sufix_0', '_example_sufix_1']
])
def test_transformer_sufix_tab_datasetarray_wrong_length(imblearn_sampler, ds, sufix):
    with pytest.raises(Exception):
        transformer = TransformerFromIMBLEARN(imblearn_sampler)
        array = TransformerArray(transformer, dataset_suffixes=sufix)
        array.fit(ds)
        array.transform(ds)


@pytest.mark.parametrize('imblearn_sampler,ratio', [
    (RandomUnderSampler(sampling_strategy=1, random_state=42), 1),
    (RandomOverSampler(sampling_strategy=1, random_state=42), 1)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)])
])
def test_imbalance_ratio(imblearn_sampler, ratio, ds):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer)
    array.fit(ds)
    out = array.transform(ds)

    for ds_array in out:
        assert ds_array.imbalance_ratio == round(ratio, 2)


@pytest.mark.parametrize('imblearn_sampler,ratio', [
    (RandomUnderSampler(sampling_strategy=1, random_state=42), 1),
    (RandomUnderSampler(sampling_strategy=0.95, random_state=42), 0.95),
    (RandomOverSampler(sampling_strategy=1, random_state=42), 1),
    (RandomOverSampler(sampling_strategy=0.95, random_state=42), 0.95)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([deepcopy(ds_openml_2)]),
    DatasetArray([deepcopy(ds_openml_1), deepcopy(ds_openml_2)])
])
def test_imbalance_ratio_2(imblearn_sampler, ratio, ds):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer)
    array.fit(ds)
    out = array.transform(ds)

    for ds_array in out:
        assert round(1 / ds_array.imbalance_ratio, 2) == round(ratio, 2)


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomUnderSampler(sampling_strategy=0.95, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_1, df_1, target_1), deepcopy(ds_openml_2)])
])
@pytest.mark.parametrize('param', [
    {
        'sampling_strategy': [1, 0.98]
    },
    {
        'random_state': [1, 2],
        'sampling_strategy': [1, 0.98]
    },
    {
        'random_state': [1, 2],
        'imbalance_ratio': [1 / 1, 1 / 0.98]
    },
    {
        'random_state': [1, 2],
        'IR': [1 / 1, 1 / 0.98]
    }
])
def test_set_get_params(imblearn_sampler, ds, param):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer)
    array.fit(ds)
    array.set_params(**param)
    array.transform(ds)

    tmp = []
    for i in range(len(param[list(param.keys())[0]])):
        tmp_dict = {}
        for key in param:
            tmp_dict[key] = param[key][i]
        tmp.append(tmp_dict)

    assert array.get_params() == tmp

    for i in range(len(array.transformers)):
        tmp = {}
        for key in param:
            if key == 'imbalance_ratio' or key == 'IR':
                tmp['sampling_strategy'] = 1 / param[key][i]
            else:
                tmp[key] = param[key][i]

        expected_params = list(tmp.items())
        existing_params = list(array.transformers[i].get_params().items())
        assert np.alltrue([p in existing_params for p in expected_params])


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomUnderSampler(sampling_strategy=0.95, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_1, df_1, target_1), deepcopy(ds_openml_2)])
])
@pytest.mark.parametrize('param', [
    {
        'sampling_strategy': [[1, 0.98], [1, 0.98]]
    },
    {
        'random_state': [[1, 2], [1, 2]],
        'sampling_strategy': [[1, 0.98], [1, 0.98]]
    }
])
def test_set_get_params_2(imblearn_sampler, ds, param):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer)
    array.set_params(**param)
    array.fit(ds)
    array.transform(ds)

    def transform_dict_to_lst(dct):
        lengths = [len(val) for key_, val in dct.items()]
        tmp_ = []
        for i_ in range(lengths[0]):
            tmp_dict_ = {}
            for key_ in dct:
                tmp_dict_[key_] = dct[key_][i_]
            tmp_.append(tmp_dict_)
        return tmp_

    out1 = transform_dict_to_lst(param)
    assert array.get_params() == out1

    for i in range(len(array.transformers)):
        expected_params = transform_dict_to_lst(out1[i])
        existing_params = array.transformers[i].get_params()
        assert np.alltrue([expected_params[k] in existing_params for k in range(len(expected_params))])
        assert np.alltrue([existing_params[k] in expected_params for k in range(len(existing_params))])


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomUnderSampler(sampling_strategy=0.95, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_1, df_1, target_1), deepcopy(ds_openml_2)])
])
@pytest.mark.parametrize('param', [
    [
        [
            {
                'sampling_strategy': 0.98
            },
            {
                'sampling_strategy': 1
            }
        ] for _ in range(2)
    ],
    [
        [
            {
                'sampling_strategy': 0.98,
                'random_state': 1
            },
            {
                'sampling_strategy': 1,
                'random_state': 2
            }
        ] for _ in range(2)
    ]
])
def test_params_in_arguments(imblearn_sampler, ds, param):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer, parameters=param)
    array.fit(ds)
    array.transform(ds)

    assert array.get_params() == param

    for i in range(len(array.transformers)):
        for j in range(len(array.transformers[i])):
            expected_params = list(param[i][j].items())
            existing_params = list(array.transformers[i][j].get_params().items())
            assert np.alltrue([p in existing_params for p in expected_params])


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomUnderSampler(sampling_strategy=0.95, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_1, df_1, target_1), deepcopy(ds_openml_2)])
])
@pytest.mark.parametrize('param', [
    [
        [
            {
                'sampling_strategy': 0.98
            },
            {
                'sampling_strategy': 1
            }
        ] for _ in range(2)
    ],
    [
        [
            {
                'sampling_strategy': 0.98,
                'random_state': 1
            },
            {
                'sampling_strategy': 1,
                'random_state': 2
            }
        ] for _ in range(2)
    ]
])
@pytest.mark.parametrize('sufix', [
    ['_transformed_0x', '_transformed_1x'],
    ['_example_sufix_0', '_example_sufix_1']
])
def test_params_in_arguments_and_sufix(imblearn_sampler, ds, param, sufix):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer, parameters=param, dataset_suffixes=sufix)
    array.fit(ds)
    out = array.transform(ds)

    # params
    assert array.get_params() == param

    for i in range(len(array.transformers)):
        for j in range(len(array.transformers[i])):
            expected_params = list(param[i][j].items())
            existing_params = list(array.transformers[i][j].get_params().items())
            assert np.alltrue([p in existing_params for p in expected_params])

    # sufix
    assert len(param) == len(out)
    assert len(sufix) == len(out)

    for j in range(len(out)):
        for k in range(len(out[j])):
            out_tmp = out[j][k]
            assert out_tmp.name == ds[j].name + sufix[j] + '_' + str(k)


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomUnderSampler(sampling_strategy=0.95, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_1, df_1, target_1), deepcopy(ds_openml_2)])
])
@pytest.mark.parametrize('param', [
    [
        [
            {
                'sampling_strategy': 0.98
            },
            {
                'sampling_strategy': 1
            }
        ] for _ in range(2)
    ],
    [
        [
            {
                'sampling_strategy': 0.98,
                'random_state': 1
            },
            {
                'sampling_strategy': 1,
                'random_state': 2
            }
        ] for _ in range(2)
    ]
])
@pytest.mark.parametrize('sufix', [
    ['_transformed_0'],
    ['_example_sufix_0']
])
def test_params_in_arguments_and_sufix_2(imblearn_sampler, ds, param, sufix):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer, parameters=param, dataset_suffixes=sufix)
    array.fit(ds)
    out = array.transform(ds)

    # params
    assert array.get_params() == param

    for i in range(len(array.transformers)):
        for j in range(len(array.transformers[i])):
            expected_params = list(param[i][j].items())
            existing_params = list(array.transformers[i][j].get_params().items())
            assert np.alltrue([p in existing_params for p in expected_params])

    # sufix
    assert len(param) == len(out)

    for j in range(len(out)):
        for k in range(len(out[j])):
            out_tmp = out[j][k]
            assert out_tmp.name == ds[j].name + sufix[0] + '_' + str(j) + '_' + str(k)


def test_over_under_sampler():
    ds = Dataset(name_2, df_1, target_1)

    try:
        transformer_1 = RandomOverSampler_EDGAR()
        transformer_1.fit(ds)
        transformer_1.transform(ds)

        transformer_2 = RandomUnderSampler_EDAGR()
        transformer_2.fit(ds)
        transformer_2.transform(ds)
    except (Exception,):
        assert False


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)])
])
def test_was_fitted(imblearn_sampler, ds):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer)
    assert not array.was_fitted
    array.fit(ds)
    assert array.was_fitted


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)])
])
def test_output_verbose(imblearn_sampler, ds, capsys):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer, verbose=True)
    array.fit(ds)
    array.transform(ds)

    captured = capsys.readouterr()
    assert f'TransformerArray {array.__repr__()} was fitted with {ds.name}' in captured.out
    assert f'TransformerArray {array.__repr__()} transformed with {ds.name}' in captured.out


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, df_1, target_1)])
])
def test_base_transformer_get_set(imblearn_sampler, ds):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer)

    assert isinstance(array.base_transformer, Transformer)

    try:
        ar = deepcopy(array)
        ar.base_transformer = deepcopy(imblearn_sampler)
        ar.fit(ds)
    except (Exception,):
        assert False

    with pytest.raises(Exception):
        ar = deepcopy(array)
        tmp = deepcopy(imblearn_sampler)
        tmp.__class__ = BaseTransformer
        ar.base_transformer = tmp
        x = ar.base_transformer
