import pytest
from sklearn.ensemble import RandomForestClassifier
from EDGAR.data.Dataset import Dataset, DatasetFromOpenML
from EDGAR.model.Model import RandomForest, ModelFromSKLEARN
from .resources.objects import *


@pytest.mark.parametrize('ds', [
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
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
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
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
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
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


@pytest.mark.parametrize('ds', [
    Dataset(name_1, df_1, target_1),
    Dataset(name_2, df_2, target_2)
])
def test_model_output_names(ds):
    ds.remove_nans()

    model = RandomForest()
    model.fit(ds)
    y = model.predict(ds)
    assert y.name == ds.name + '_predicted'

    y = model.predict_proba(ds)
    assert y.name == ds.name + '_predicted_probabilities'


@pytest.mark.parametrize('ds,model_name', [
    (Dataset(name_1, df_1, target_1), 'model_1'),
    (Dataset(name_2, df_2, target_2), 'model_2')
])
def test_model_output_names(ds, model_name):
    ds.remove_nans()

    model = RandomForest(name=model_name)
    model.fit(ds)
    y = model.predict(ds)
    assert y.name == ds.name + '_' + model_name + '_predicted'

    y = model.predict_proba(ds)
    assert y.name == ds.name + '_' + model_name + '_predicted_probabilities'

