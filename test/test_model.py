import numpy as np
import random

import pandas as pd
from copy import deepcopy
import pytest
from sklearn.ensemble import RandomForestClassifier
from EDGAR.data.Dataset import Dataset, DatasetFromOpenML
from EDGAR.model.Model import RandomForest, ModelFromSKLEARN
from .resources.objects import *


np.random.seed(42)
random.seed(42)


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
def test_model_output(ds):
    ds = deepcopy(ds)
    ds.remove_nans()

    model = RandomForest(test_size=0.01)
    model.fit(ds)
    y = model.predict(ds)
    assert isinstance(y, Dataset)
    assert y.check_binary_classification()
    y = model.predict_proba(ds)
    assert isinstance(y, Dataset)


@pytest.mark.parametrize('ds', [
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
])
def test_model_output_names(ds):
    ds = deepcopy(ds)
    ds.remove_nans()

    model = RandomForest(test_size=0.01)
    model.fit(ds)
    y = model.predict(ds)
    assert y.name == ds.name + '_predicted'

    y = model.predict_proba(ds)
    assert y.name == ds.name + '_predicted_probabilities'


@pytest.mark.parametrize('ds,model_name', [
    (DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), 'model_1'),
    (DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY), 'model_2')
])
def test_model_output_names_2(ds, model_name):
    ds = deepcopy(ds)
    ds.remove_nans()

    model = RandomForest(name=model_name, test_size=0.01)
    model.fit(ds)
    y = model.predict(ds)
    assert y.name == ds.name + '_' + model_name + '_predicted'

    y = model.predict_proba(ds)
    assert y.name == ds.name + '_' + model_name + '_predicted_probabilities'

