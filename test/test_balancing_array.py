import pytest
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from EDGAR.balancing.Transformer import TransformerFromIMBLEARN
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
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)])
])
def test_transformer_array(imblearn_sampler, ds):
    try:
        transformer = TransformerFromIMBLEARN(imblearn_sampler)
        array = TransformerArray(transformer)
        array.fit(ds)
        array.transform(ds)
    except (Exception,):
        assert False


# TODO:
# connect two tests below into one
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
    array = TransformerArray(transformer, name_sufix=sufix)
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
    array = TransformerArray(transformer, name_sufix=sufix)
    array.fit(ds)
    out = array.transform(ds)

    expected_names = [dataset.name + sufix for dataset in ds.datasets]
    assert len(expected_names) == len(out)

    for ds_array in out:
        assert ds_array.name == ds.name + '_array'

        for i in range(len(ds_array.datasets)):
            assert ds_array.datasets[i].name in expected_names


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
    array = TransformerArray(transformer, name_sufix=sufix)
    array.fit(ds)
    out = array.transform(ds)

    expected_names = [ds.datasets[i].name + sufix[i] for i in range(len(ds.datasets))]
    assert len(expected_names) == len(out)

    for ds_array in out:
        assert ds_array.name == ds.name + '_array'

        for i in range(len(ds_array.datasets)):
            assert ds_array.datasets[i].name in expected_names


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)]),
    DatasetArray([Dataset(name_1, df_1, target_1), DatasetFromOpenML(task_id=task_id_1)])
])
@pytest.mark.parametrize('sufix', [
    ['_transformed_0', '_transformed_1'],
    ['_example_sufix_0', '_example_sufix_1']
])
def test_transformer_sufix_tab_2_datasetarray(imblearn_sampler, ds, sufix):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer, name_sufix=sufix)
    array.fit(ds)
    out = array.transform(ds)

    expected_names = [ds.datasets[i].name + sufix[i] for i in range(len(ds.datasets))]
    assert len(expected_names) == len(out)

    for ds_array in out:
        assert ds_array.name == ds.name + '_array'

        for i in range(len(ds_array.datasets)):
            assert ds_array.datasets[i].name in expected_names


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
        array = TransformerArray(transformer, name_sufix=sufix)
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
        for i in range(len(ds_array.datasets)):
            assert ds_array.datasets[i].imbalance_ratio() == round(ratio, 2)


@pytest.mark.parametrize('imblearn_sampler,ratio', [
    (RandomUnderSampler(sampling_strategy=1, random_state=42), 1),
    (RandomUnderSampler(sampling_strategy=0.95, random_state=42), 0.95),
    (RandomOverSampler(sampling_strategy=1, random_state=42), 1),
    (RandomOverSampler(sampling_strategy=0.95, random_state=42), 0.95)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([DatasetFromOpenML(task_id=task_id_1)]),
    DatasetArray([DatasetFromOpenML(task_id=task_id_1), DatasetFromOpenML(task_id=task_id_2)])
])
def test_imbalance_ratio_2(imblearn_sampler, ratio, ds):
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    array = TransformerArray(transformer)
    array.fit(ds)
    out = array.transform(ds)

    for ds_array in out:
        for i in range(len(ds_array.datasets)):
            assert round(1 / ds_array.datasets[i].imbalance_ratio(), 2) == round(ratio, 2)


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomUnderSampler(sampling_strategy=0.95, random_state=42)
])
@pytest.mark.parametrize('ds', [
    DatasetArray([Dataset(name_2, df_1, target_1), Dataset(name_1, df_1, target_1)]),
    DatasetArray([DatasetFromOpenML(task_id=task_id_1), DatasetFromOpenML(task_id=task_id_2)])
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
        for j in range(len(array.get_transformers()[i])):
            tmp = {}
            for key in param:
                tmp[key] = param[key][j]

            expected_params = list(tmp.items())
            existing_params = list(array.get_transformers()[i][j].get_params().items())
            assert np.alltrue([p in existing_params for p in expected_params])


# set params
# parameters in arguments
# sufix and params
