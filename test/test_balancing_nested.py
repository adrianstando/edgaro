import pandas as pd
import pytest

from copy import deepcopy

from edgaro.balancing.transformer import SMOTE
from edgaro.base.base_transformer import BaseTransformer
from edgaro.data.dataset import Dataset
from edgaro.data.dataset_array import DatasetArray
from edgaro.balancing.transformer_array import TransformerArray
from edgaro.balancing.nested_transformer import BasicAutomaticTransformer, ExtensionAutomaticTransformer, \
    AutomaticTransformer, NestedAutomaticTransformer

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
        str(super(NestedAutomaticTransformer, array))
        repr(super(NestedAutomaticTransformer, array))
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

    assert len(array.get_dataset_suffixes()) == 0

    array.fit(ds)
    out = array.transform(ds)
    assert out.name == ds.name + suffix

    assert out[0].name.endswith('UNDERSAMPLING__Random')
    assert out[1].name.endswith('OVERSAMPLING__Random')
    assert out[2].name.endswith('OVERSAMPLING__SMOTE')

    try:
        out = array.get_dataset_suffixes()
    except (Exception,):
        assert False

    assert len(out) > 0


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


@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1),
    Dataset(name_2, df_2, target_2)
])
def test_transform_empty(ds):
    array = BasicAutomaticTransformer(n_per_method=2)
    assert array.transform(ds) == DatasetArray([])


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))
])
def test_transform_keep_original(ds):
    array = BasicAutomaticTransformer(n_per_method=2, keep_original_dataset=True)
    array.fit(ds)
    transform_true = array.transform(ds)

    array.keep_original_dataset = False
    transform_false = array.transform(ds)

    assert len(transform_true) - 1 == len(transform_false)


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))
])
def test_output_verbose(ds, capsys):
    array = BasicAutomaticTransformer(n_per_method=2, verbose=True)
    array.fit(ds)
    array.transform(ds)

    captured = capsys.readouterr()
    assert f'AutomaticTransformerArray {array.__repr__()} was fitted with {ds.name}' in captured.out
    assert f'AutomaticTransformerArray {array.__repr__()} transformed {ds.name}' in captured.out


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))
])
def test_get_set_transformers(ds):
    array = BasicAutomaticTransformer(n_per_method=2)
    assert array.transformers == []
    assert len(array) == 0
    array.fit(ds)
    array.transform(ds)
    assert len(array.transformers) > 0
    assert len(array) > 0

    with pytest.raises(Exception):
        array.transformers = deepcopy(array.transformers)

    array1 = BasicAutomaticTransformer(n_per_method=2)
    try:
        array1.transformers = deepcopy(array.transformers)
    except (Exception,):
        assert False

    try:
        array1.transform(ds)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1)
])
def test_base_transformer_get_set(ds):
    array = BasicAutomaticTransformer(n_per_method=2)

    assert isinstance(array.base_transformer, list)

    try:
        ar = deepcopy(array)
        ar.base_transformer = TransformerArray(SMOTE())
        ar.fit(ds)
    except (Exception,):
        assert False

    with pytest.raises(Exception):
        ar = deepcopy(array)
        tmp = TransformerArray(SMOTE())
        tmp.__class__ = BaseTransformer
        ar.base_transformer = tmp
        x = ar.base_transformer


@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1)
])
def test_base_transformer_get_set_none(ds):
    array = BasicAutomaticTransformer(n_per_method=2)
    ar = deepcopy(array)
    ar.base_transformer = None
    assert ar.base_transformer == []


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_2 for _ in range(5)]), pd.concat([target_2 for _ in range(5)]))
])
def test_base_transformer_get_set_fail_fitted(ds):
    array = BasicAutomaticTransformer(n_per_method=2)
    ar = deepcopy(array)
    ar.fit(ds)
    with pytest.raises(Exception):
        ar.base_transformer = None


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)]))
])
def test_get_item(ds):
    array = BasicAutomaticTransformer(n_per_method=2)
    assert array[0] is None
    array.fit(ds)

    try:
        x = array[0]
        x = array[1]
        x = array[[1, 2]]
    except (Exception,):
        assert False

    assert array[123444] is None
    assert array[1.23] is None


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_2 for _ in range(5)]), pd.concat([target_2 for _ in range(5)]))
])
@pytest.mark.parametrize('n', [
    2, 3
])
def test_extension_automatic_transformer_array(ds, n):
    try:
        array = ExtensionAutomaticTransformer(n_per_method=n)
        str(array)
        repr(array)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_2 for _ in range(5)]), pd.concat([target_2 for _ in range(5)]))
])
@pytest.mark.parametrize('n', [
    2, 3
])
def test_extension_automatic_transformer_array(ds, n):
    try:
        array = AutomaticTransformer(n_per_method=n)
        str(array)
        repr(array)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    Dataset(name_1, pd.concat([df_2 for _ in range(5)]), pd.concat([target_2 for _ in range(5)]))
])
def test_get_set_params(ds):
    array = BasicAutomaticTransformer(n_per_method=2)
    array.fit(ds)

    try:
        param = array.get_params()
    except (Exception,):
        assert False

    assert isinstance(param, list)

    try:
        array.set_params(IR=[[1.5, 1] for _ in range(3)])
    except (Exception,):
        assert False

