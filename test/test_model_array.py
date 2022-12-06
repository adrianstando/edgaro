import pandas as pd
import pytest

from copy import deepcopy

from edgaro.balancing.transformer import Transformer
from edgaro.data.dataset_array import DatasetArray
from edgaro.data.dataset import Dataset
from edgaro.model.model import RandomForest, Model
from edgaro.model.model_array import ModelArray

from .resources.objects import *


@pytest.mark.parametrize('ds', [
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
        Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
    ]),
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
        DatasetArray([
            Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
            Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
        ])
    ])
])
def test_model_array(ds):
    try:
        ds.remove_nans()

        model = ModelArray(RandomForest(max_depth=1, n_estimators=1, random_state=42))
        model.fit(ds)
        model.predict(ds)
        model.predict_proba(ds)
        model.evaluate()
        str(model)
        repr(model)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
        Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
    ])
])
def test_was_fitted(ds):
    ds.remove_nans()

    model = ModelArray(RandomForest(max_depth=1, n_estimators=1, random_state=42))
    assert not model.was_fitted
    model.fit(ds)
    assert model.was_fitted


@pytest.mark.parametrize('ds', [
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
        Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
    ])
])
def test_model_array_output(ds):
    ds.remove_nans()

    model = ModelArray(RandomForest(max_depth=1, n_estimators=1, random_state=42))
    model.fit(ds)
    y = model.predict(ds)

    assert isinstance(y, DatasetArray)
    assert len(y) == len(ds)
    for i in range(len(ds)):
        assert isinstance(y[i], Dataset)

    y = model.predict_proba(ds)
    assert isinstance(y, DatasetArray)
    assert len(y) == len(ds)
    for i in range(len(ds)):
        assert isinstance(y[i], Dataset)

    out = model.evaluate()
    assert isinstance(out, pd.DataFrame)


@pytest.mark.parametrize('ds', [
    DatasetArray([
        DatasetArray([
            Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
            Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
            Dataset(name_2 + 'xx', pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
            DatasetArray([
                Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
                Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
                Dataset(name_2 + 'xx', pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
                DatasetArray([
                    Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
                    Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
                    Dataset(name_2 + 'xx', pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
                ])
            ])
        ]),
        Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
    ])
])
def test_evaluate_many_nested(ds):
    ds.remove_nans()

    model = ModelArray(RandomForest(max_depth=1, n_estimators=1, random_state=42))
    model.fit(ds)

    try:
        out = model.evaluate()
    except (Exception,):
        assert False

    assert isinstance(out, pd.DataFrame)


@pytest.mark.parametrize('ds', [
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
        DatasetArray([
            Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
            Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
        ])
    ])
])
def test_model_array_nested_output(ds):
    ds.remove_nans()

    model = ModelArray(RandomForest(max_depth=1, n_estimators=1, random_state=42))
    model.fit(ds)
    y = model.predict(ds)

    assert isinstance(y, DatasetArray)
    assert len(y) == len(ds)
    assert isinstance(y[0], Dataset)
    assert isinstance(y[1], DatasetArray)

    y = model.predict_proba(ds)
    assert isinstance(y, DatasetArray)
    assert len(y) == len(ds)
    assert isinstance(y[0], Dataset)
    assert isinstance(y[1], DatasetArray)

    out = model.evaluate()
    assert isinstance(out, pd.DataFrame)


@pytest.mark.parametrize('ds', [
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
        Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
    ])
])
def test_output_verbose(ds, capsys):
    ds.remove_nans()

    model = ModelArray(RandomForest(max_depth=1, n_estimators=1, random_state=42), verbose=True)
    model.fit(ds)
    y = model.predict(ds)
    y = model.predict_proba(ds)

    model_repr_pre = model.__repr__()
    for ele in model_repr_pre:
        if ele.isdigit():
            model_repr_pre = model_repr_pre.replace(ele, '0')

    captured = capsys.readouterr()
    assert f'ModelArray {model_repr_pre} is being fitted with {ds.name}' in captured.out
    assert f'ModelArray {model.__repr__()} was fitted with {ds.name}' in captured.out
    assert f'ModelArray {model.__repr__()} predicted on {ds.name}' in captured.out
    assert f'ModelArray {model.__repr__()} predicted probabilities on {ds.name}' in captured.out

    model.evaluate()
    captured = capsys.readouterr()
    assert f'ModelArray {model.__repr__()} is being evaluated' in captured.out
    assert f'ModelArray {model.__repr__()} was evaluated' in captured.out


@pytest.mark.parametrize('ds', [
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
        Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
    ])
])
@pytest.mark.parametrize('model', [
    RandomForest(max_depth=1, n_estimators=1, random_state=42)
])
def test_base_transformer_get_set(model, ds):
    array = ModelArray(model)

    assert isinstance(array.base_transformer, Model)

    try:
        ar = deepcopy(array)
        ar.base_transformer = deepcopy(model)
        ar.fit(ds)
    except (Exception,):
        assert False

    with pytest.raises(Exception):
        ar = deepcopy(array)
        tmp = deepcopy(model)
        tmp.__class__ = ModelArray
        ar.base_transformer = tmp
        x = ar.base_transformer
