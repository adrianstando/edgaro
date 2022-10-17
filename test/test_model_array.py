import pandas as pd
import pytest
from EDGAR.data.DatasetArray import DatasetArray
from EDGAR.data.Dataset import Dataset, DatasetFromOpenML
from EDGAR.model.Model import RandomForest
from EDGAR.model.ModelArray import ModelArray
from .resources.objects import *


@pytest.mark.parametrize('ds', [
    DatasetArray([DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)]),
    DatasetArray([Dataset(name_3, df_3, target_3), Dataset(name_3 + 'x', df_3, target_3)]),
    DatasetArray([
        DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
        DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
        DatasetArray([Dataset(name_3, df_3, target_3), Dataset(name_3 + 'x', df_3, target_3)])
    ])
])
def test_model_array(ds):
    try:
        ds.remove_nans()

        model = ModelArray(RandomForest(random_state=42))
        model.fit(ds)
        model.predict(ds)
        model.predict_proba(ds)
        model.evaluate()
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    DatasetArray([DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)]),
    DatasetArray([Dataset(name_3, df_3, target_3), Dataset(name_3 + 'x', df_3, target_3)])
])
def test_model_array_output(ds):
    ds.remove_nans()

    model = ModelArray(RandomForest(random_state=42))
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
        DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
        DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
        DatasetArray([Dataset(name_3, df_3, target_3), Dataset(name_3 + 'x', df_3, target_3)])
    ])
])
def test_model_array_with_arrays_of_arrays_output(ds):
    ds.remove_nans()

    model = ModelArray(RandomForest(random_state=42))
    model.fit(ds)
    y = model.predict(ds)

    assert isinstance(y, DatasetArray)
    assert len(y) == len(ds)
    assert isinstance(y[0], Dataset)
    assert isinstance(y[1], Dataset)
    assert isinstance(y[2], DatasetArray)

    y = model.predict_proba(ds)
    assert isinstance(y, DatasetArray)
    assert len(y) == len(ds)
    assert isinstance(y[0], Dataset)
    assert isinstance(y[1], Dataset)
    assert isinstance(y[2], DatasetArray)

    out = model.evaluate()
    assert isinstance(out, pd.DataFrame)
