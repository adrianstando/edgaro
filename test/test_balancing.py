import pytest

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from copy import deepcopy

from edgaro.balancing.transformer import TransformerFromIMBLEARN, SMOTE
from edgaro.data.dataset import Dataset

from .resources.objects import *


@pytest.fixture(scope='module', params=[
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
def sampler(request):
    return request.param


@pytest.fixture(scope='module', params=[
    (name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
    (name_2, pd.concat([df_2 for _ in range(5)]), pd.concat([target_2 for _ in range(5)]))])
def dataset(request):
    return Dataset(request.param[0], request.param[1], request.param[2])


def test_transformer(sampler, dataset):
    try:
        transformer = TransformerFromIMBLEARN(sampler)
        transformer.fit(dataset)
        transformer.transform(dataset)
        str(transformer)
        repr(transformer)
    except (Exception,):
        assert False


def test_transformer_train_test_split(sampler, dataset):
    ds = deepcopy(dataset)
    ds.train_test_split()

    try:
        transformer = TransformerFromIMBLEARN(sampler)
        transformer.fit(ds)
        transformer.transform(ds)
        str(transformer)
        repr(transformer)
    except (Exception,):
        assert False


def test_sufix(sampler, dataset):
    transformer = TransformerFromIMBLEARN(sampler)
    transformer.fit(dataset)
    out = transformer.transform(dataset)

    assert out.name == dataset.name + '_transformed'


@pytest.mark.parametrize('sufix', [
    'sufix_1',
    'sufix_2',
    'example_sufix'
])
def test_sufix_2(sampler, dataset, sufix):
    transformer = TransformerFromIMBLEARN(sampler, sufix)
    transformer.fit(dataset)
    out = transformer.transform(dataset)

    assert out.name == dataset.name + sufix


@pytest.mark.parametrize('imblearn_sampler,expected_imbalance_ratio', [
    (RandomUnderSampler(sampling_strategy=1, random_state=42), 1),
    (RandomOverSampler(sampling_strategy=1, random_state=42), 1)
])
def test_imbalance_ratio(imblearn_sampler, expected_imbalance_ratio):
    ds = Dataset(name_1, df_1, target_1)
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    transformer.fit(ds)
    out = transformer.transform(ds)

    assert out.imbalance_ratio == expected_imbalance_ratio


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42)
])
def test_was_fitted(imblearn_sampler):
    ds = Dataset(name_1, df_1, target_1)
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    assert not transformer.was_fitted
    transformer.fit(ds)
    assert transformer.was_fitted


def test_output_verbose(sampler, dataset, capsys):
    transformer = TransformerFromIMBLEARN(sampler, verbose=True)
    transformer.fit(dataset)
    transformer.transform(dataset)

    captured = capsys.readouterr()
    assert f'Transformer {transformer.__repr__()} was fitted with {dataset.name}' in captured.out
    assert f'Transformer {transformer.__repr__()} transformed with {dataset.name}' in captured.out


def test_empty_data_fit(sampler, dataset):
    ds = deepcopy(dataset)
    ds.data = None

    transformer = TransformerFromIMBLEARN(sampler)
    with pytest.raises(Exception):
        transformer.fit(ds)


def test_empty_data_target(sampler, dataset):
    ds = deepcopy(dataset)
    ds.target = None

    transformer = TransformerFromIMBLEARN(sampler)
    with pytest.raises(Exception):
        transformer.fit(ds)


@pytest.mark.parametrize('ds', [
    Dataset(name_2, pd.concat([df_2 for _ in range(10)]), pd.concat([target_2 for _ in range(10)])),  # all continuous
    Dataset(name_1, pd.concat([df_1 for _ in range(10)]), pd.concat([target_1 for _ in range(10)])),  # mixed types
    Dataset(name_1, pd.concat([df_1_categorical for _ in range(10)]), pd.concat([target_1 for _ in range(10)])),  # all categorical
])
def test_smote(ds):
    try:
        transformer = SMOTE()
        transformer.fit(ds)
        transformer.transform(ds)
        str(transformer)
        repr(transformer)
    except (Exception,):
        assert False
