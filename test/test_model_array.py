import pytest
from EDGAR.data.DatasetArray import DatasetArray
from EDGAR.data.Dataset import Dataset, DatasetFromOpenML
from EDGAR.model.Model import RandomForest
from EDGAR.model.ModelArray import ModelArray
from .resources.objects import *


@pytest.mark.parametrize('ds', [
    DatasetArray([DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)]),
    DatasetArray([Dataset(name_3, df_3, target_3), Dataset(name_1, df_1, target_1)])
])
def test_model_array(ds):
    try:
        ds.remove_nans()

        model = ModelArray(RandomForest())
        model.fit(ds)
        model.predict(ds)
        model.predict_proba(ds)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    DatasetArray([DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)]),
    DatasetArray([Dataset(name_3, df_3, target_3), Dataset(name_1, df_1, target_1)])
])
def test_model_array_output(ds):
    ds.remove_nans()

    model = ModelArray(RandomForest())
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
