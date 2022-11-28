import pytest

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from edgaro.balancing.transformer import TransformerFromIMBLEARN
from edgaro.data.dataset import Dataset

from .resources.objects import *


@pytest.fixture(scope='module', params=[
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
def sampler(request):
    return request.param


@pytest.fixture(scope='module', params=[(name_1, df_1, target_1), (name_2, df_2, target_2)])
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
