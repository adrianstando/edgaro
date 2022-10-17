from copy import deepcopy
import pytest
from sklearn.ensemble import RandomForestClassifier
from EDGAR.data.Dataset import Dataset, DatasetFromOpenML
from EDGAR.model.Model import RandomForest, ModelFromSKLEARN, XGBoost, RandomSearchCV, GridSearchCV
from .resources.objects import *


@pytest.mark.parametrize('ds', [
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
])
def test_model(ds):
    try:
        ds = deepcopy(ds)
        ds.remove_nans()

        model = RandomForest(test_size=0.01)
        model.fit(ds)
        model.evaluate()
        model.predict(ds)
        model.predict_proba(ds)
        str(model)
        repr(model)

        model = XGBoost(test_size=0.01)
        model.fit(ds)
        model.evaluate()
        model.predict(ds)
        model.predict_proba(ds)
        str(model)
        repr(model)

        model = RandomSearchCV(base_model=RandomForest(), param_grid={'n_estimators': [10, 20, 50]}, test_size=0.01)
        model.fit(ds)
        model.evaluate()
        model.predict(ds)
        model.predict_proba(ds)
        str(model)
        repr(model)

        model = GridSearchCV(base_model=RandomForest(), param_grid={'n_estimators': [10, 20, 50]}, test_size=0.01)
        model.fit(ds)
        model.evaluate()
        model.predict(ds)
        model.predict_proba(ds)
        str(model)
        repr(model)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
])
def test_model_2(ds):
    try:
        ds = deepcopy(ds)
        ds.remove_nans()

        model = ModelFromSKLEARN(RandomForestClassifier(), test_size=0.01)
        model.fit(ds)
        model.predict(ds)
        model.predict_proba(ds)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
])
def test_transform_model_target(ds):
    try:
        ds = deepcopy(ds)
        ds.remove_nans()

        model = ModelFromSKLEARN(RandomForestClassifier(), test_size=0.01)
        model.fit(ds)

        X = model.transform_data(ds)
        Y = model.transform_target(ds)
    except (Exception,):
        assert False


@pytest.mark.parametrize('ds', [
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
])
@pytest.mark.parametrize('model', [
    RandomForest(test_size=0.01),
    XGBoost(test_size=0.01),
    RandomSearchCV(base_model=RandomForest(), param_grid={'n_estimators': [10, 20, 50]}, test_size=0.01),
    GridSearchCV(base_model=RandomForest(), param_grid={'n_estimators': [10, 20, 50]}, test_size=0.01)
])
def test_model_output(ds, model):
    ds = deepcopy(ds)
    model = deepcopy(model)

    ds.remove_nans()
    model.fit(ds)
    y = model.predict(ds)
    model.evaluate()
    assert isinstance(y, Dataset)
    assert y.check_binary_classification()
    y = model.predict_proba(ds)
    assert isinstance(y, Dataset)


@pytest.mark.parametrize('ds', [
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
])
@pytest.mark.parametrize('model', [
    RandomForest(test_size=0.01),
    XGBoost(test_size=0.01),
    RandomSearchCV(base_model=RandomForest(), param_grid={'n_estimators': [10, 20, 50]}, test_size=0.01),
    GridSearchCV(base_model=RandomForest(), param_grid={'n_estimators': [10, 20, 50]}, test_size=0.01)
])
def test_model_output_names(ds, model):
    ds = deepcopy(ds)
    model = deepcopy(model)
    ds.remove_nans()

    model.fit(ds)
    y = model.predict(ds)
    assert y.name == ds.name + '_predicted'

    y = model.predict_proba(ds)
    assert y.name == ds.name + '_predicted_probabilities'


@pytest.mark.parametrize('ds,model_name', [
    (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), 'model_1'),
    (DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY), 'model_2')
])
@pytest.mark.parametrize('model', [
    RandomForest(test_size=0.01),
    XGBoost(test_size=0.01),
    RandomSearchCV(base_model=RandomForest(), param_grid={'n_estimators': [10, 20, 50]}, test_size=0.01),
    GridSearchCV(base_model=RandomForest(), param_grid={'n_estimators': [10, 20, 50]}, test_size=0.01)
])
def test_model_output_names_2(ds, model_name, model):
    ds = deepcopy(ds)
    model = deepcopy(model)
    model.name = model_name

    ds.remove_nans()

    model.fit(ds)
    y = model.predict(ds)
    assert y.name == ds.name + '_' + model_name + '_predicted'

    y = model.predict_proba(ds)
    assert y.name == ds.name + '_' + model_name + '_predicted_probabilities'

