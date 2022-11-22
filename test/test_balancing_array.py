import pandas as pd
import pytest
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from copy import deepcopy
from EDGAR.balancing.transformer import TransformerFromIMBLEARN, RandomUnderSampler as RandomUnderSampler_EDAGR, RandomOverSampler as RandomOverSampler_EDGAR
from EDGAR.data.dataset import Dataset, DatasetFromOpenML
from EDGAR.data.dataset_array import DatasetArray
from EDGAR.balancing.transformer_array import TransformerArray, AutomaticTransformerArray
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
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2), DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)])]),
    DatasetArray([DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)], name='x'), DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)])])
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
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_1, target_1), Dataset(name_1 + '3', df_1, target_1)])
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
        'sampling_strategy': [1, 0.98]
    },
    {
        'random_state': [1, 2],
        'sampling_strategy': [1, 0.98]
    }
])
def test_set_get_params_2(imblearn_sampler, ds, param):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer)
    array.set_params(**param)
    array.fit(ds)
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
            tmp[key] = param[key][i]

        expected_params = tmp
        existing_params = array.transformers[i].get_params()
        assert expected_params in existing_params


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
        {
            'sampling_strategy': 0.98
        },
        {
            'sampling_strategy': 1
        }
    ],
    [
        {
            'sampling_strategy': 0.98,
            'random_state': 1
        },
        {
            'sampling_strategy': 1,
            'random_state': 2
        }
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
            expected_params = list(param[j].items())
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
        {
            'sampling_strategy': 0.98
        },
        {
            'sampling_strategy': 1
        }
    ],
    [
        {
            'sampling_strategy': 0.98,
            'random_state': 1
        },
        {
            'sampling_strategy': 1,
            'random_state': 2
        }
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
            expected_params = list(param[j].items())
            existing_params = list(array.transformers[i][j].get_params().items())
            assert np.alltrue([p in existing_params for p in expected_params])

    # sufix
    assert len(param) == len(out)
    assert len(sufix) == len(out)

    for j in range(len(out)):
        ds_array = out[j]
        expected_names = [ds.datasets[j].name + sufix[i] for i in range(len(ds.datasets))]
        assert np.alltrue([d.name in expected_names for d in ds_array])


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
        {
            'sampling_strategy': 0.98
        },
        {
            'sampling_strategy': 1
        }
    ],
    [
        {
            'sampling_strategy': 0.98,
            'random_state': 1
        },
        {
            'sampling_strategy': 1,
            'random_state': 2
        }
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
            expected_params = list(param[j].items())
            existing_params = list(array.transformers[i][j].get_params().items())
            assert np.alltrue([p in existing_params for p in expected_params])

    # sufix
    assert len(param) == len(out)

    for j in range(len(out)):
        ds_array = out[j]
        expected_names = [ds.datasets[j].name + sufix[0] + '_' + str(i) for i in range(len(ds.datasets))]
        assert np.alltrue([d.name in expected_names for d in ds_array])


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


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))
])
@pytest.mark.parametrize('n', [
    2, 3
])
def test_automatic_transformer_array(ds, n):
    try:
        array = AutomaticTransformerArray(n_per_method=n)
        array.fit(ds)
        array.transform(ds)
        str(array)
        repr(array)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))
])
@pytest.mark.parametrize('n', [
    2, 3
])
@pytest.mark.parametrize('suffix', [
    '_suffix_1',
    '_example_sufix'
])
def test_automatic_transformer_array_suffix(ds, n, suffix):
    array = AutomaticTransformerArray(n_per_method=n, result_array_sufix=suffix)
    array.fit(ds)
    out = array.transform(ds)
    assert out.name == ds.name + suffix
    assert out[0].name.endswith('_RANDOM_UNDERSAMPLING')
    assert out[1].name.endswith('_RANDOM_OVERSAMPLING')
    assert out[2].name.endswith('_SMOTE')


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))
])
@pytest.mark.parametrize('n', [
    2, 3
])
def test_automatic_transformer_array_shape(ds, n):
    array = AutomaticTransformerArray(n_per_method=n)
    array.fit(ds)
    out = array.transform(ds)

    assert len(out) == 3
    for d in out:
        assert len(d) == n


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))
])
@pytest.mark.parametrize('min_samples_to_modify, n, expected_n', [
    (1, 5, 5),
    (2, 5, 5),
    (3, 5, 4),
    (4, 5, 3),
    (6, 5, 3),
    (7, 5, 2),
    (10, 5, 2),
    (20, 5, 1)
])
def test_automatic_transformer_array_shape_with_min_samples_to_modify(ds, min_samples_to_modify, n, expected_n):
    array = AutomaticTransformerArray(n_per_method=n, min_samples_to_modify=min_samples_to_modify)
    array.fit(ds)
    out = array.transform(ds)

    assert len(out) == 3
    for d in out:
        assert len(d) == expected_n


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))
])
@pytest.mark.parametrize('n', [
    2, 3
])
def test_automatic_transformer_array_was_fitted(ds, n):
    array = AutomaticTransformerArray(n_per_method=n)
    assert not array.was_fitted
    array.fit(ds)
    assert array.was_fitted


@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))]),
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)])),
        Dataset(name_2, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))]),
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)])),
        Dataset(name_2, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)])),
        DatasetArray([
            Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)])),
            Dataset(name_2, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))])])
])
@pytest.mark.parametrize('n', [
    2, 3
])
def test_array_of_automatic_transformer_array(ds, n):
    try:
        transformer = AutomaticTransformerArray(n_per_method=n)
        array = TransformerArray(transformer)
        array.fit(ds)
        array.transform(ds)
        str(array)
        repr(array)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))]),
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)])),
        Dataset(name_2, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))])
])
@pytest.mark.parametrize('n', [
    2, 3
])
@pytest.mark.parametrize('suffix', [
    '_suffix_1',
    '_example_sufix'
])
def test_array_of_automatic_transformer_array_suffix(ds, n, suffix):
    transformer = AutomaticTransformerArray(n_per_method=n)
    array = TransformerArray(transformer, result_array_sufix=suffix, allow_dataset_array_sufix_change=False)
    array.fit(ds)
    out = array.transform(ds)

    assert out.name == ds.name + suffix

    for d in out:
        assert d[0].name.endswith('_RANDOM_UNDERSAMPLING')
        assert d[1].name.endswith('_RANDOM_OVERSAMPLING')
        assert d[2].name.endswith('_SMOTE')


@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))]),
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)])),
        Dataset(name_2, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))])
])
@pytest.mark.parametrize('n', [
    2, 3
])
def test_array_of_automatic_transformer_array_was_fitted(ds, n):
    transformer = AutomaticTransformerArray(n_per_method=n)
    array = TransformerArray(transformer)
    assert not array.was_fitted
    array.fit(ds)
    assert array.was_fitted


@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))]),
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)])),
        Dataset(name_2, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))])
])
@pytest.mark.parametrize('n', [
    2, 3
])
def test_array_of_automatic_transformer_array_shape(ds, n):
    transformer = AutomaticTransformerArray(n_per_method=n)
    array = TransformerArray(transformer)
    array.fit(ds)
    out = array.transform(ds)

    assert len(out) == len(ds)
    for trans_for_ds in out:
        assert len(trans_for_ds) == 3
        for trans_for_ir in trans_for_ds:
            assert len(trans_for_ir) == n
