import pytest

from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from copy import deepcopy

from EDGAR.data.dataset import DatasetFromOpenML, Dataset
from EDGAR.balancing.transformer import TransformerFromIMBLEARN, RandomUnderSampler as RUS
from EDGAR.balancing.transformer_array import TransformerArray
from EDGAR.model.model import RandomForest
from EDGAR.model.model_array import ModelArray
from EDGAR.explain.explainer import Explainer
from EDGAR.explain.explainer_array import ExplainerArray
from EDGAR.data.dataset_array import DatasetArray

from .resources.objects import *


@pytest.mark.parametrize('df', [
    Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
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

        rf = RandomForest(test_size=0.3, max_depth=1, n_estimators=1, random_state=42)
        rf.fit(df_new)
        rf.evaluate()

        test = rf.get_test_dataset()
        y = rf.predict(test)
        accuracy_score(y.target, rf.transform_target(test).target)

        pdp = Explainer(rf, N=10)
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
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
        DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY)
    ])
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

        rf = ModelArray(RandomForest(test_size=0.3, max_depth=1, n_estimators=1, random_state=42))
        rf.fit(df_new)

        for m in rf:
            m.evaluate()

        for i in range(len(rf)):
            m = rf.get_models()[i]
            test = m.get_test_dataset()
            y = m.predict(test)
            accuracy_score(y.target, m.transform_target(test).target)

        pdp = ExplainerArray(rf, N=10)
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

        t[0].compare([t1[0], t1[0]])
        t[0].compare([t1[0], t1[0]], variable=df[0].data.columns[0])
        t[0].compare([t1[0], t1[0]], variable=[df[0].data.columns[0], df[0].data.columns[1]])

        t_2 = pdp.transform(variables=[df[0].data.columns[0], df[0].data.columns[1]])
        t_2[0].plot(variable=df[0].data.columns[0])
        t_2[0].plot(variable=df[0].data.columns[1])

        with pytest.raises(Exception):
            t_2[0].plot(variable=df[0].data.columns[2])

        ale = ExplainerArray(rf, N=10, curve_type='ALE')
        ale.fit()

        str(ale)
        repr(ale)

        p = ale.transform()

        str(p[0])
        repr(p[0])

        p[0].plot(variable=df[0].data.columns[0])

    except (Exception,):
        assert False


@pytest.mark.parametrize('df', [
    DatasetArray([
        Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
        DatasetFromOpenML(task_id=task_id_2, apikey=APIKEY),
        DatasetArray([
            Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
            Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
        ])
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

        rf = ModelArray(RandomForest(test_size=0.3, max_depth=1, n_estimators=1, random_state=42))
        rf.fit(df_new)

        def evaluate(model):
            if isinstance(model, ModelArray):
                for mod in model:
                    evaluate(mod)
            else:
                model.evaluate()

        evaluate(rf)

        pdp = ExplainerArray(rf, N=10)
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

        t[0].compare([t1[0], t1[0]])
        t[0].compare([t1[0], t1[0]], variable=df[0].data.columns[0])
        t[0].compare([t1[0], t1[0]], variable=[df[0].data.columns[0], df[0].data.columns[1]])

        # there are also datasets, which don't contain given columns
        t_2 = pdp.transform(variables=[df[0].data.columns[0], df[0].data.columns[1]])
        t_2[0].plot(variable=df[0].data.columns[0])
        t_2[0].plot(variable=df[0].data.columns[1])
        assert len(t_2[1].results.items()) == 0

        t[2][0].plot(variable=df[2][0].data.columns[0])

        with pytest.raises(Exception):
            t_2[0].plot(variable=df[0].data.columns[2])

        ale = ExplainerArray(rf, N=10, curve_type='ALE')
        ale.fit()

        str(ale)
        repr(ale)

        p = ale.transform()

        str(p[0])
        repr(p[0])

        p[0].plot(variable=df[0].data.columns[0])

    except (Exception,):
        assert False
