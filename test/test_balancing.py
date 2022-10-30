import pytest
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from EDGAR.balancing.Transformer import TransformerFromIMBLEARN
from EDGAR.data.Dataset import Dataset
from .resources.objects import *


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1),
    Dataset(name_2, df_2, target_2)
])
def test_transformer(imblearn_sampler, ds):
    try:
        transformer = TransformerFromIMBLEARN(imblearn_sampler)
        transformer.fit(ds)
        transformer.transform(ds)
        str(transformer)
        repr(transformer)
    except (Exception,):
        assert False


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
def test_sufix(imblearn_sampler):
    ds = Dataset(name_1, df_1, target_1)
    transformer = TransformerFromIMBLEARN(imblearn_sampler)
    transformer.fit(ds)
    out = transformer.transform(ds)

    assert out.name == name_1 + '_transformed'


@pytest.mark.parametrize('imblearn_sampler', [
    RandomUnderSampler(sampling_strategy=1, random_state=42),
    RandomOverSampler(sampling_strategy=1, random_state=42)
])
@pytest.mark.parametrize('sufix', [
    'sufix_1',
    'sufix_2',
    'example_sufix'
])
def test_sufix_2(imblearn_sampler, sufix):
    ds = Dataset(name_1, df_1, target_1)
    transformer = TransformerFromIMBLEARN(imblearn_sampler, sufix)
    transformer.fit(ds)
    out = transformer.transform(ds)

    assert out.name == name_1 + sufix


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
