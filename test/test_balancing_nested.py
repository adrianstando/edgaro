import pandas as pd
import pytest

from edgaro.data.dataset import Dataset
from edgaro.data.dataset_array import DatasetArray
from edgaro.balancing.transformer_array import TransformerArray
from edgaro.balancing.nested_transformer import BasicAutomaticTransformer

from .resources.objects import *


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))
])
@pytest.mark.parametrize('n', [
    2, 3
])
def test_automatic_transformer_array(ds, n):
    try:
        array = BasicAutomaticTransformer(n_per_method=n)
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
    array = BasicAutomaticTransformer(n_per_method=n, result_array_sufix=suffix)
    array.fit(ds)
    out = array.transform(ds)
    assert out.name == ds.name + suffix

    assert out[0].name.endswith('UNDERSAMPLING__Random')
    assert out[1].name.endswith('OVERSAMPLING__Random')
    assert out[2].name.endswith('OVERSAMPLING__SMOTE')


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))
])
@pytest.mark.parametrize('n', [
    2, 3
])
def test_automatic_transformer_array_shape(ds, n):
    array = BasicAutomaticTransformer(n_per_method=n)
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
    array = BasicAutomaticTransformer(n_per_method=n, min_samples_to_modify=min_samples_to_modify)
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
    array = BasicAutomaticTransformer(n_per_method=n)
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
        transformer = BasicAutomaticTransformer(n_per_method=n)
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
    transformer = BasicAutomaticTransformer(n_per_method=n)
    array = TransformerArray(transformer, result_array_sufix=suffix, allow_dataset_array_sufix_change=False)
    array.fit(ds)
    out = array.transform(ds)

    assert out.name == ds.name + suffix

    for d in out:
        assert d[0].name.endswith('UNDERSAMPLING__Random')
        assert d[1].name.endswith('OVERSAMPLING__Random')
        assert d[2].name.endswith('OVERSAMPLING__SMOTE')


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
    transformer = BasicAutomaticTransformer(n_per_method=n)
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
    transformer = BasicAutomaticTransformer(n_per_method=n)
    array = TransformerArray(transformer)
    array.fit(ds)
    out = array.transform(ds)

    assert len(out) == len(ds)
    for trans_for_ds in out:
        assert len(trans_for_ds) == 3
        for trans_for_ir in trans_for_ds:
            assert len(trans_for_ir) == n
