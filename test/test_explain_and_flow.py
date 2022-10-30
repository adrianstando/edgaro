import pytest
from EDGAR.data.Dataset import DatasetFromOpenML, Dataset
from EDGAR.balancing.Transformer import TransformerFromIMBLEARN, RandomUnderSampler as RUS
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
    DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
    DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)
])
def test_flow(df):
    try:
        df.remove_nans()
        IR = df.imbalance_ratio
        transformator = TransformerFromIMBLEARN(RandomUnderSampler(sampling_strategy=1, random_state=42))
        transformator.fit(df)
        df_new = transformator.transform(df)
        IR = df_new.imbalance_ratio

        rf = RandomForest(test_size=0.1)
        rf.fit(df_new)
        rf.evaluate()

        test = rf.get_test_dataset()
        y = rf.predict(test)
        accuracy_score(y.target, rf.transform_target(test).target)

        pdp = PDPCalculator(rf, N=200)
        pdp.fit()

        str(pdp)
        repr(pdp)

        col = df.data.columns[0]
        t = pdp.transform([col])

        str(t)
        repr(t)

        t.plot(variable=col)

        cols = [df.data.columns[0], df.data.columns[1]]
        t1 = pdp.transform(cols)
        t1.plot(variable=cols[0])
        t1.plot(variable=cols[1])

    except (Exception,):
        assert False


@pytest.mark.parametrize('df', [
    DatasetArray(
        [DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY), DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)]),
    DatasetArray([Dataset(name_1, df_3, target_3), Dataset(name_2, df_3, target_3)])
])
def test_flow_array(df):
    try:
        df.remove_nans()
        IR = df[0].imbalance_ratio
        IR = df[1].imbalance_ratio

        transformator = TransformerArray(
            TransformerFromIMBLEARN(RandomUnderSampler(sampling_strategy=1, random_state=42)))
        transformator.fit(df)
        df_new = transformator.transform(df)

        IR = df_new[0].imbalance_ratio
        IR = df_new[1].imbalance_ratio

        rf = ModelArray(RandomForest(test_size=0.1))
        rf.fit(df_new)

        for m in rf:
            m.evaluate()

        for i in range(len(rf)):
            m = rf.get_models()[i]
            test = m.get_test_dataset()
            y = m.predict(test)
            accuracy_score(y.target, m.transform_target(test).target)

        pdp = PDPCalculatorArray(rf, N=200)
        pdp.fit()

        str(pdp)
        repr(pdp)

        t = pdp.transform()

        str(t[0])
        repr(t[0])

        t[0].plot(variable=df[0].data.columns[0])

        t1 = deepcopy(t)
        t1[0].results[df[0].data.columns[0]].y += 0.1
        t1[0].name += '_xx'

        t[0].plot(variable=df[0].data.columns[0], add_plot=[t1[0]])

        t[0].compare(t1[0])
        t[0].compare(t1[0], variable=df[0].data.columns[0])
        t[0].compare(t1[0], variable=[df[0].data.columns[0], df[0].data.columns[1]])

        t_2 = pdp.transform(variables=[df[0].data.columns[0], df[0].data.columns[1]])
        t_2[0].plot(variable=df[0].data.columns[0])
        t_2[0].plot(variable=df[0].data.columns[1])

        try:
            t_2[0].plot(variable=df[0].data.columns[2])
            assert False
        except (Exception,):
            pass

    except (Exception,):
        assert False


@pytest.mark.parametrize('df', [
    DatasetArray([
        DatasetFromOpenML(task_id=task_id_1, apikey=APIKEY),
        DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
        DatasetArray([Dataset(name_1, df_3, target_3), Dataset(name_2, df_3, target_3)])
    ])
])
def test_flow_array_of_arrays(df):
    try:
        df.remove_nans()
        IR = df[0].imbalance_ratio
        IR = df[1].imbalance_ratio
        IR = df[2][0].imbalance_ratio
        IR = df[2][1].imbalance_ratio

        transformator = TransformerArray(RUS(imbalance_ratio=1, random_state=42))
        transformator.fit(df)
        df_new = transformator.transform(df)

        IR = df_new[0].imbalance_ratio
        IR = df_new[1].imbalance_ratio
        IR = df_new[2][0].imbalance_ratio
        IR = df_new[2][1].imbalance_ratio

        rf = ModelArray(RandomForest(test_size=0.1))
        rf.fit(df_new)

        def evaluate(model):
            if isinstance(model, ModelArray):
                for mod in model:
                    evaluate(mod)
            else:
                model.evaluate()

        evaluate(rf)

        pdp = PDPCalculatorArray(rf, N=200)
        pdp.fit()

        str(pdp)
        repr(pdp)

        t = pdp.transform()

        str(t[0])
        repr(t[0])

        t[0].plot(variable=df[0].data.columns[0])

        t1 = deepcopy(t)
        t1[0].results[df[0].data.columns[0]].y += 0.1
        t1[0].name += '_xx'

        t[0].plot(variable=df[0].data.columns[0], add_plot=[t1[0]])

        t[0].compare(t1[0])
        t[0].compare(t1[0], variable=df[0].data.columns[0])
        t[0].compare(t1[0], variable=[df[0].data.columns[0], df[0].data.columns[1]])

        # there are also datasets, which don't contain given columns
        t_2 = pdp.transform(variables=[df[0].data.columns[0], df[0].data.columns[1]])
        t_2[0].plot(variable=df[0].data.columns[0])
        t_2[0].plot(variable=df[0].data.columns[1])
        assert len(t_2[1].results.items()) == 0

        t[2][0].plot(variable=df[2][0].data.columns[0])

        try:
            t_2[0].plot(variable=df[0].data.columns[2])
            assert False
        except (Exception,):
            pass

    except (Exception,):
        assert False
