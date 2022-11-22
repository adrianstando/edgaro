import random
import string
from copy import deepcopy
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from EDGAR.data.dataset import Dataset, DatasetFromOpenML
from EDGAR.model.model import RandomForest, ModelFromSKLEARN, XGBoost, RandomSearchCV, GridSearchCV
from .resources.objects import *


@pytest.fixture(scope='module', params=[
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
    Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
])
def ds_test_fitting(request):
    ds = request.param
    ds = deepcopy(ds)
    ds.remove_nans()
    return ds


@pytest.fixture(scope='module', params=[
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
    Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
])
def ds_test_fitting_with_split(request):
    ds = request.param
    ds = deepcopy(ds)
    ds.remove_nans()
    ds.train_test_split(test_size=0.3, random_state=42)
    return ds


class TestFitting:
    def test_random_forest(self, ds_test_fitting):
        try:
            model = RandomForest(test_size=None, max_depth=1, n_estimators=1, random_state=42)
            model.fit(ds_test_fitting)
            model.evaluate()
            model.predict(ds_test_fitting)
            model.predict_proba(ds_test_fitting)
            str(model)
            repr(model)
        except (Exception,):
            assert False

        with pytest.raises(Exception):
            model.predict(ds_test_fitting.test)
            model.predict_proba(ds_test_fitting.test)

    def test_xgboost(self, ds_test_fitting):
        try:
            model = XGBoost(test_size=None, max_depth=1, n_estimators=1, random_state=42)
            model.fit(ds_test_fitting)
            model.evaluate()
            model.predict(ds_test_fitting)
            model.predict_proba(ds_test_fitting)
            str(model)
            repr(model)
        except (Exception,):
            assert False

        with pytest.raises(Exception):
            model.predict(ds_test_fitting.test)
            model.predict_proba(ds_test_fitting.test)

    def test_grid_search(self, ds_test_fitting):
        try:
            model = GridSearchCV(test_size=None, base_model=RandomForest(),
                                 param_grid={'n_estimators': [1, 2], 'max_depth': [1]}, random_state=42)
            model.fit(ds_test_fitting)
            model.evaluate()
            model.predict(ds_test_fitting)
            model.predict_proba(ds_test_fitting)
            str(model)
            repr(model)
        except (Exception,):
            assert False

        with pytest.raises(Exception):
            model.predict(ds_test_fitting.test)
            model.predict_proba(ds_test_fitting.test)

    def test_random_search(self, ds_test_fitting):
        try:
            model = RandomSearchCV(test_size=None, base_model=RandomForest(),
                                   param_grid={'n_estimators': [1, 2], 'max_depth': [1]}, random_state=42)
            model.fit(ds_test_fitting)
            model.evaluate()
            model.predict(ds_test_fitting)
            model.predict_proba(ds_test_fitting)
            str(model)
            repr(model)
        except (Exception,):
            assert False

        with pytest.raises(Exception):
            model.predict(ds_test_fitting.test)
            model.predict_proba(ds_test_fitting.test)

    def test_was_fitted(self, ds_test_fitting):
        model = RandomForest(test_size=None, max_depth=1, n_estimators=1, random_state=42)
        assert not model.was_fitted
        model.fit(ds_test_fitting)
        assert model.was_fitted


class TestFittingWithSplit:
    def test_random_forest_with_test(self, ds_test_fitting):
        try:
            ds = deepcopy(ds_test_fitting)
            model = RandomForest(test_size=0.3, max_depth=1, n_estimators=1, random_state=42)
            model.fit(ds)
            model.evaluate()
            model.predict(ds)
            model.predict_proba(ds)
            str(model)
            repr(model)
            model.predict(ds.test)
            model.predict_proba(ds.test)
        except (Exception,):
            assert False

    def test_xgboost_with_test(self, ds_test_fitting):
        try:
            ds = deepcopy(ds_test_fitting)
            model = XGBoost(test_size=0.3, max_depth=1, n_estimators=1, random_state=42)
            model.fit(ds)
            model.evaluate()
            model.predict(ds)
            model.predict_proba(ds)
            str(model)
            repr(model)
            model.predict(ds.test)
            model.predict_proba(ds.test)
        except (Exception,):
            assert False

    def test_grid_search_with_test(self, ds_test_fitting):
        try:
            ds = deepcopy(ds_test_fitting)
            model = GridSearchCV(test_size=0.3, base_model=RandomForest(),
                                 param_grid={'n_estimators': [1, 2], 'max_depth': [1]}, random_state=42)
            model.fit(ds)
            model.evaluate()
            model.predict(ds)
            model.predict_proba(ds)
            str(model)
            repr(model)
            model.predict(ds.test)
            model.predict_proba(ds.test)
        except (Exception,):
            assert False

    def test_random_search_with_test(self, ds_test_fitting):
        try:
            ds = deepcopy(ds_test_fitting)
            model = RandomSearchCV(test_size=0.3, base_model=RandomForest(),
                                   param_grid={'n_estimators': [1, 2], 'max_depth': [1]}, random_state=42)
            model.fit(ds)
            model.evaluate()
            model.predict(ds)
            model.predict_proba(ds)
            str(model)
            repr(model)
            model.predict(ds.test)
            model.predict_proba(ds.test)
        except (Exception,):
            assert False


class TestFittingWithEarlierSplit:
    def test_random_forest(self, ds_test_fitting_with_split):
        try:
            model = RandomForest(test_size=None, max_depth=1, n_estimators=1, random_state=42)
            model.fit(ds_test_fitting_with_split)
            model.evaluate()
            model.predict(ds_test_fitting_with_split)
            model.predict_proba(ds_test_fitting_with_split)
            str(model)
            repr(model)
            model.predict(ds_test_fitting_with_split.test)
            model.predict_proba(ds_test_fitting_with_split.test)
        except (Exception,):
            assert False

    def test_xgboost(self, ds_test_fitting_with_split):
        try:
            model = XGBoost(test_size=None, max_depth=1, n_estimators=1, random_state=42)
            model.fit(ds_test_fitting_with_split)
            model.evaluate()
            model.predict(ds_test_fitting_with_split)
            model.predict_proba(ds_test_fitting_with_split)
            str(model)
            repr(model)
            model.predict(ds_test_fitting_with_split.test)
            model.predict_proba(ds_test_fitting_with_split.test)
        except (Exception,):
            assert False

    def test_grid_search(self, ds_test_fitting_with_split):
        try:
            model = GridSearchCV(test_size=None, base_model=RandomForest(),
                                 param_grid={'n_estimators': [1, 2], 'max_depth': [1]}, random_state=42)
            model.fit(ds_test_fitting_with_split)
            model.evaluate()
            model.predict(ds_test_fitting_with_split)
            model.predict_proba(ds_test_fitting_with_split)
            str(model)
            repr(model)
            model.predict(ds_test_fitting_with_split.test)
            model.predict_proba(ds_test_fitting_with_split.test)
        except (Exception,):
            assert False

    def test_random_search(self, ds_test_fitting_with_split):
        try:
            model = RandomSearchCV(test_size=None, base_model=RandomForest(),
                                   param_grid={'n_estimators': [1, 2], 'max_depth': [1]}, random_state=42)
            model.fit(ds_test_fitting_with_split)
            model.evaluate()
            model.predict(ds_test_fitting_with_split)
            model.predict_proba(ds_test_fitting_with_split)
            str(model)
            repr(model)
            model.predict(ds_test_fitting_with_split.test)
            model.predict_proba(ds_test_fitting_with_split.test)
        except (Exception,):
            assert False


def test_fitting_from_sklearn_model(ds_test_fitting_with_split):
    try:
        model = ModelFromSKLEARN(RandomForestClassifier(max_depth=1, n_estimators=1, random_state=42), test_size=0.3)
        model.fit(ds_test_fitting_with_split)
        model.predict(ds_test_fitting_with_split)
        model.predict_proba(ds_test_fitting_with_split)
        model.predict(ds_test_fitting_with_split.test)
        model.predict_proba(ds_test_fitting_with_split.test)
    except (Exception,):
        assert False


@pytest.mark.parametrize('model', [
    RandomForest(max_depth=1, n_estimators=1, random_state=42),
    XGBoost(max_depth=1, n_estimators=1, random_state=42)
])
class TestModel:
    def test_transform_data(self, model, ds_test_fitting):
        model = deepcopy(model)
        ds = deepcopy(ds_test_fitting)

        model.fit(ds)
        try:
            X = model.transform_data(ds)
        except (Exception,):
            assert False

    def test_transform_target(self, model, ds_test_fitting):
        model = deepcopy(model)
        ds = deepcopy(ds_test_fitting)

        model.fit(ds)
        try:
            Y = model.transform_target(ds)
        except (Exception,):
            assert False

    def test_output_types(self, model, ds_test_fitting):
        model = deepcopy(model)
        ds = deepcopy(ds_test_fitting)
        model.fit(ds)

        y = model.predict(ds)
        assert isinstance(y, Dataset)
        assert y.check_binary_classification()
        y = model.predict_proba(ds)
        assert isinstance(y, Dataset)

    def test_output_names(self, model, ds_test_fitting):
        model = deepcopy(model)
        ds = deepcopy(ds_test_fitting)
        model.fit(ds)

        y = model.predict(ds)
        assert y.name == ds.name + '_predicted'
        y = model.predict_proba(ds)
        assert y.name == ds.name + '_predicted_probabilities'

    def test_output_names_with_model_name(self, model, ds_test_fitting):
        model_name = ''.join(random.choice(string.ascii_uppercase) for _ in range(5))
        model = deepcopy(model)
        model.name = model_name
        ds = deepcopy(ds_test_fitting)
        model.fit(ds)

        y = model.predict(ds)
        assert y.name == ds.name + '_' + model_name + '_predicted'
        y = model.predict_proba(ds)
        assert y.name == ds.name + '_' + model_name + '_predicted_probabilities'
