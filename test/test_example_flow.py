import pytest
from EDGAR.data.Dataset import DatasetFromOpenML, Dataset
from EDGAR.balancing.Transformer import TransformerFromIMBLEARN
from EDGAR.balancing.TransformerArray import TransformerArray
from imblearn.under_sampling import RandomUnderSampler
from EDGAR.model.Model import RandomForest
from EDGAR.model.ModelArray import ModelArray
from EDGAR.explain.PDPCalculator import PDPCalculator
from EDGAR.explain.PDPCalculatorArray import PDPCalculatorArray
from EDGAR.data.DatasetArray import DatasetArray
from sklearn.metrics import accuracy_score
from copy import deepcopy
from .resources.objects import *


@pytest.mark.parametrize('df', [
    DatasetFromOpenML(task_id=task_id_1),
    DatasetFromOpenML(task_id=task_id_2)
])
def test_flow(df):
    try:
        df.remove_nans()
        df.imbalance_ratio()
        transformator = TransformerFromIMBLEARN(RandomUnderSampler(sampling_strategy=1, random_state=42))
        transformator.fit(df)
        df_new = transformator.transform(df)
        df_new.imbalance_ratio()

        rf = RandomForest()
        rf.fit(df_new)
        y = rf.predict(df_new)
        accuracy_score(y.target, rf.get_train_dataset().target)

        pdp = PDPCalculator(rf)
        pdp.fit()

        col = df.data.columns[0]
        t = pdp.transform([col])
        t.plot(variable=col)

        cols = [df.data.columns[0], df.data.columns[1]]
        t1 = pdp.transform(cols)
        t1.plot(variable=cols[0])
        t1.plot(variable=cols[1])

    except (Exception,):
        assert False


@pytest.mark.parametrize('df', [
    DatasetArray([DatasetFromOpenML(task_id=task_id_1), DatasetFromOpenML(task_id=task_id_2)]),
    DatasetArray([Dataset(name_1, df_1, target_1), Dataset(name_2, df_2, target_2)])
])
def test_flow_array(df):
    try:
        df.remove_nans()
        df[0].imbalance_ratio()
        df[1].imbalance_ratio()

        transformator = TransformerArray(TransformerFromIMBLEARN(RandomUnderSampler(sampling_strategy=1, random_state=42)))
        transformator.fit(df)
        df_new = transformator.transform(df)

        df_new[0][0].imbalance_ratio()
        df_new[1][0].imbalance_ratio()

        # TODO:
        # this behaviour has to be changed
        df_new = DatasetArray([df_new[0][0], df_new[1][0]])

        rf = ModelArray(RandomForest())
        rf.fit(df_new)
        y = rf.predict(df_new)
        # TODO:
        # this behaviour has to be changed
        accuracy_score(y[0][0].target, rf.get_models()[0][0].get_train_dataset().target)

        pdp = PDPCalculatorArray(rf)
        pdp.fit()
        t = pdp.transform()
        t[0].plot(variable=df[0].data.columns[0])

        t1 = deepcopy(t)
        t1[0].results[df[0].data.columns[0]].y += 0.1
        t1[0].name += '_xx'

        t[0].plot(variable=df[0].data.columns[0], add_plot=[t1[0]])

    except (Exception,):
        assert False
