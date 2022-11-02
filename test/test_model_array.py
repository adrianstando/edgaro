import pandas as pd
import pytest
from EDGAR.data.DatasetArray import DatasetArray
from EDGAR.data.Dataset import Dataset, DatasetFromOpenML
from EDGAR.model.Model import RandomForest
from EDGAR.model.ModelArray import ModelArray
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
