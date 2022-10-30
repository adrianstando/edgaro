import pytest
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from EDGAR.balancing.Transformer import TransformerFromIMBLEARN, RandomUnderSampler as RandomUnderSampler_EDAGR, RandomOverSampler as RandomOverSampler_EDGAR
from EDGAR.data.Dataset import Dataset, DatasetFromOpenML
from EDGAR.data.DatasetArray import DatasetArray
from EDGAR.balancing.TransformerArray import TransformerArray
from .resources.objects import *


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
    for ds_array in out:
        assert ds_array.name in expected_names


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

    for ds_array in out:
        assert ds_array.name in expected_names


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)]),
    DatasetArray([Dataset(name_1, df_1, target_1), DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY)])
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

    for ds_array in out:
        assert ds_array.name in expected_names


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, df_1, target_1)]),
    DatasetArray([Dataset(name_2, df_2, target_2)])
])
@pytest.mark.parametrize('sufix', [
    ['_transformed_0', '_transformed_1'],
    ['_example_sufix_0', '_example_sufix_1']
])
def test_transformer_sufix_tab_datasetarray_wrong_length(imblearn_sampler, ds, sufix):
    try:
        transformer = TransformerFromIMBLEARN(imblearn_sampler)
        array = TransformerArray(transformer, dataset_suffixes=sufix)
        array.fit(ds)
        array.transform(ds)
    except (Exception,):
        pass


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
    DatasetArray([DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY)]),
    DatasetArray([DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)])
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
    DatasetArray([DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)])
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

    for i in range(len(array.get_transformers())):
        tmp = {}
        for key in param:
            tmp[key] = param[key][i]

        expected_params = list(tmp.items())
        existing_params = list(array.get_transformers()[i].get_params().items())
        assert np.alltrue([p in existing_params for p in expected_params])


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomUnderSampler(sampling_strategy=0.95, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)]),
    DatasetArray([DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)])
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

    for i in range(len(array.get_transformers())):
        tmp = {}
        for key in param:
            tmp[key] = param[key][i]

        expected_params = tmp
        existing_params = array.get_transformers()[i].get_params()
        assert expected_params in existing_params


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomUnderSampler(sampling_strategy=0.95, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)]),
    DatasetArray([DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)])
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

    for i in range(len(array.get_transformers())):
        for j in range(len(array.get_transformers()[i])):
            expected_params = list(param[j].items())
            existing_params = list(array.get_transformers()[i][j].get_params().items())
            assert np.alltrue([p in existing_params for p in expected_params])


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomUnderSampler(sampling_strategy=0.95, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)]),
    DatasetArray([DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)])
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
    ['_transformed_0', '_transformed_1'],
    ['_example_sufix_0', '_example_sufix_1']
])
def test_params_in_arguments_and_sufix(imblearn_sampler, ds, param, sufix):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer, parameters=param, dataset_suffixes=sufix)
    array.fit(ds)
    out = array.transform(ds)

    # params
    assert array.get_params() == param

    for i in range(len(array.get_transformers())):
        for j in range(len(array.get_transformers()[i])):
            expected_params = list(param[j].items())
            existing_params = list(array.get_transformers()[i][j].get_params().items())
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
    DatasetArray([DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)])
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

    for i in range(len(array.get_transformers())):
        for j in range(len(array.get_transformers()[i])):
            expected_params = list(param[j].items())
            existing_params = list(array.get_transformers()[i][j].get_params().items())
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
