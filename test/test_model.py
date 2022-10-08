import pytest
from sklearn.ensemble import RandomForestClassifier
from EDGAR.data.Dataset import Dataset, DatasetFromOpenML
from EDGAR.model.Model import RandomForest, ModelFromSKLEARN
from .resources.objects import *


@pytest.mark.parametrize('ds', [
    DatasetFromOpenML(task_id=task_id_1),
    DatasetFromOpenML(task_id=task_id_2),
])
def test_model(ds):
    try:
        ds.remove_nans()

        model = RandomForest()
        model.fit(ds)
        model.predict(ds)
        model.predict_proba(ds)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    DatasetFromOpenML(task_id=task_id_1),
    DatasetFromOpenML(task_id=task_id_2),
])
def test_model_2(ds):
    try:
        ds.remove_nans()

        model = ModelFromSKLEARN(RandomForestClassifier())
        model.fit(ds)
        model.predict(ds)
        model.predict_proba(ds)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    DatasetFromOpenML(task_id=task_id_1),
    DatasetFromOpenML(task_id=task_id_2),
])
def test_model_output(ds):
    ds.remove_nans()

    model = RandomForest()
    model.fit(ds)
    y = model.predict(ds)
    assert isinstance(y, Dataset)
    assert y.check_binary_classification()
    y = model.predict_proba(ds)
    assert isinstance(y, Dataset)
