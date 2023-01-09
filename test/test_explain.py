import pytest

from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
from copy import deepcopy

from edgaro.data.dataset import DatasetFromOpenML, Dataset
from edgaro.balancing.transformer import TransformerFromIMBLEARN, RandomUnderSampler as RUS
from edgaro.balancing.transformer_array import TransformerArray
from edgaro.explain.explainer_result import ModelProfileExplanation
from edgaro.explain.explainer_result_array import ModelProfileExplanationArray
from edgaro.model.model import RandomForest
from edgaro.model.model_array import ModelArray
from edgaro.explain.explainer import Explainer
from edgaro.explain.explainer_array import ExplainerArray
from edgaro.data.dataset_array import DatasetArray

from .resources.objects import *


@pytest.fixture(
    scope='module',
    params=[
        Dataset(name_1, pd.concat([df_1 for _ in range(30)]), pd.concat([target_1 for _ in range(30)]))
    ]
)
def model(request):
    df = request.param
    df.remove_nans()
    transformator = TransformerFromIMBLEARN(RandomUnderSampler(sampling_strategy=1, random_state=42))
    transformator.fit(df)
    df_new = transformator.transform(df)

    rf = RandomForest(test_size=0.3, max_depth=1, n_estimators=1, random_state=42)
    rf.fit(df_new)
    return rf


def test_explainer(model, capsys):
    rf = deepcopy(model)

    try:
        pdp = Explainer(rf, N=10, verbose=True)
        pdp.fit()
        str(pdp)
        repr(pdp)
        pdp.transform()
    except (Exception,):
        assert False

    captured = capsys.readouterr()
    assert f'dalex explainer inside {pdp.__repr__()} was created with {name_1}' in captured.out
    assert f'{pdp.explanation_type} is being calculated in {pdp.__repr__()} for ' \
           f'{pdp.model.get_test_dataset().name}' in captured.out
    assert f'{pdp.explanation_type} was calculated calculated in {pdp.__repr__()} for ' \
           f'{pdp.model.get_test_dataset().name}' in captured.out


def test_transform_no_explainer(model, capsys):
    rf = deepcopy(model)
    pdp = Explainer(rf, N=10)

    with pytest.raises(Exception, ):
        pdp.transform()


def test_transform_ale(model, capsys):
    rf = deepcopy(model)
    pdp = Explainer(rf, N=10, explanation_type='ALE')
    pdp.fit()

    try:
        pdp.transform()
    except (Exception,):
        assert False


def test_transform_wrong_type(model, capsys):
    rf = deepcopy(model)
    pdp = Explainer(rf, N=10, explanation_type='XXX')

    with pytest.raises(Exception):
        pdp.transform()


@pytest.fixture(
    scope='module',
    params=[
        DatasetArray([
            Dataset(name_1, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)])),
            Dataset(name_2, pd.concat([df_1 for _ in range(20)]), pd.concat([target_1 for _ in range(20)])),
        ])
    ]
)
def model_array(request):
    df = request.param
    df.remove_nans()
    transformator = TransformerArray(
        TransformerFromIMBLEARN(RandomUnderSampler(sampling_strategy=1, random_state=42)))
    transformator.fit(df)
    df_new = transformator.transform(df)

    rf = ModelArray(RandomForest(test_size=0.3, max_depth=1, n_estimators=1, random_state=42))
    rf.fit(df_new)
    return rf


def test_explainer_array(model_array, capsys):
    rf = deepcopy(model_array)

    try:
        pdp = ExplainerArray(rf, N=10, verbose=True)
        pdp.fit()
        str(pdp)
        repr(pdp)
        pdp.transform()
    except (Exception,):
        assert False

    captured = capsys.readouterr()
    assert f'dalex explainers inside {pdp.__repr__()} were created' in captured.out
    assert f'{pdp.explanation_type}s are being calculated in {pdp.__repr__()}' in captured.out
    assert f'{pdp.explanation_type}s were calculated in {pdp.__repr__()}' in captured.out


def test_explainer_array_iterate(model_array):
    rf = deepcopy(model_array)
    pdp = ExplainerArray(rf, N=10)
    pdp.fit()
    try:
        for p in pdp:
            pass
    except (Exception,):
        assert False


def test_explainer_array_transform_not_fitted(model_array):
    rf = deepcopy(model_array)
    pdp = ExplainerArray(rf, N=10)

    with pytest.raises(Exception):
        pdp.transform()


@pytest.fixture(
    scope='module',
    params=[
        (Dataset(name_1, pd.concat([df_1 for _ in range(30)]), pd.concat([target_1 for _ in range(30)])), 'PDP'),
        (Dataset(name_1, pd.concat([df_1 for _ in range(30)]), pd.concat([target_1 for _ in range(30)])), 'ALE')
    ]
)
def pdp_transformed(request):
    df = request.param[0]
    df.remove_nans()
    transformator = TransformerFromIMBLEARN(RandomUnderSampler(sampling_strategy=1, random_state=42))
    transformator.fit(df)
    df_new = transformator.transform(df)

    rf = ModelArray(RandomForest(test_size=0.3, max_depth=1, n_estimators=1, random_state=42))
    rf.fit(df_new)

    pdp = ExplainerArray(rf, N=10, explanation_type=request.param[1])
    pdp.fit()
    t = pdp.transform()
    return t


def test_curve(pdp_transformed):
    try:
        str(pdp_transformed.results[list(pdp_transformed.results.keys())[0]])
        repr(pdp_transformed.results[list(pdp_transformed.results.keys())[0]])
    except (Exception,):
        assert False


def test_wrong_get_item(pdp_transformed):
    assert pdp_transformed['non-existing'] is None


def test_explainer_result_and_explainer_result_array(pdp_transformed):
    t = pdp_transformed

    try:
        str(t)
        repr(t)
        str(t[0])
        repr(t[0])
    except (Exception,):
        assert False


def test_plot_result_array(pdp_transformed):
    t = pdp_transformed
    column = list(pdp_transformed.results.keys())[0]

    try:
        t.plot(variable=column)
    except (Exception,):
        assert False


def test_plot_result_array_add_plot(pdp_transformed):
    t = pdp_transformed
    column = list(pdp_transformed.results.keys())[0]

    try:
        t1 = deepcopy(t)
        t1.results[column].y += 0.1
        t1.name += '_xx'
        t.plot(variable=column, add_plot=[t1])
    except (Exception,):
        assert False


def test_compare_result_array(pdp_transformed):
    t = pdp_transformed
    column1 = list(pdp_transformed.results.keys())[0]
    column2 = list(pdp_transformed.results.keys())[1]

    try:
        t1 = deepcopy(t)
        t1.results[column1].y += 0.1
        t1.name += '_xx'

        t.compare([t1, t1])
        t.compare([t1, t1], variable=column1)
        t.compare([t1, t1], variable=[column1, column2])
        x = t.compare([t1, t1], return_raw_per_variable=True)
    except (Exception,):
        assert False

    assert len(x) > 1


def test_compare_result_array_2(model_array):
    cols = list(model_array.get_models()[0].get_test_dataset().data.columns)
    column1 = cols[0]
    column2 = cols[1]

    pdp = ExplainerArray(model_array, N=10)
    pdp.fit()
    t = pdp.transform(variables=[column1])

    with pytest.raises(Exception):
        t[0].plot(variable=column2)


@pytest.fixture(
    scope='module',
    params=[
        (DatasetArray([
            Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
            Dataset(name_2, pd.concat([df_1_x for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
            DatasetArray([
                Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
                Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
            ])
        ]), 'ALE'),
        (DatasetArray([
            Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
            Dataset(name_2, pd.concat([df_1_x for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
            DatasetArray([
                Dataset(name_1, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)])),
                Dataset(name_2, pd.concat([df_1 for _ in range(5)]), pd.concat([target_1 for _ in range(5)]))
            ])
        ]), 'PDP')
    ]
)
def model_array2(request):
    df = request.param[0]
    df.remove_nans()
    transformator = TransformerArray(
        TransformerFromIMBLEARN(RandomUnderSampler(sampling_strategy=1, random_state=42)))
    transformator.fit(df)
    df_new = transformator.transform(df)

    rf = ModelArray(RandomForest(test_size=0.3, max_depth=1, n_estimators=1, random_state=42))
    rf.fit(df_new)

    pdp = ExplainerArray(rf, N=10, explanation_type=request.param[1])
    pdp.fit()
    t = pdp.transform()

    return rf, t


def test_array_of_arrays(model_array2):
    rf, t = model_array2
    try:
        str(t[0])
        repr(t[0])
    except (Exception,):
        assert False


def test_array_of_arrays_plot(model_array2):
    rf, t = model_array2
    column = rf.get_models()[0].get_train_dataset().data.columns[0]
    try:
        t[0].plot(variable=column)

        t1 = deepcopy(t)
        t1[0].results[column].y += 0.1
        t1[0].name += '_xx'

        t[0].plot(variable=column, add_plot=[t1[0]])
    except (Exception,):
        assert False


def test_explainer_result_plot(model_array2):
    rf, t = model_array2
    column = rf.get_models()[0].get_train_dataset().data.columns[0]
    try:
        t[2].plot(variables=column)
    except (Exception,):
        assert False


def test_array_of_arrays_plot_regex(model_array2):
    rf, t = model_array2
    column = rf.get_models()[0].get_train_dataset().data.columns[0]
    try:
        t[2].plot(variables=column, model_filter=f"^{name_1}")
    except (Exception,):
        assert False


def test_array_of_arrays_plot_summary_regex(model_array2):
    rf, t = model_array2
    column = rf.get_models()[0].get_train_dataset().data.columns[0]
    try:
        t[2].plot_summary(variables=column, model_filters=[f"^{name_1}", f"^{name_2}"])
    except (Exception,):
        assert False


def test_array_of_arrays_compare(model_array2):
    rf, t = model_array2
    column = rf.get_models()[0].get_train_dataset().data.columns[0]
    column1 = rf.get_models()[0].get_train_dataset().data.columns[1]
    try:
        t1 = deepcopy(t)
        t1[0].results[column].y += 0.1
        t1[0].name += '_xx'

        t[0].compare([t1[0], t1[0]])
        t[0].compare([t1[0], t1[0]], variable=column)
        t[0].compare([t1[0], t1[0]], variable=[column, column1])
    except (Exception,):
        assert False


def test_array_of_arrays_transform_columns_not_in_each(model_array2):
    rf, t = model_array2
    column = rf.get_models()[0].get_train_dataset().data.columns[0]
    column1 = rf.get_models()[0].get_train_dataset().data.columns[1]
    try:
        t1 = deepcopy(t)
        t1[0].results[column].y += 0.1
        t1[0].name += '_xx'

        # there are also datasets, which don't contain given columns
        pdp = ExplainerArray(rf, N=10, explanation_type=t.explanation_type)
        pdp.fit()

        pdp.transform(variables=[column, column1])
    except (Exception,):
        assert False


def test_array_of_arrays_transform_columns_not_in_each_plot(model_array2):
    rf, t = model_array2
    column = rf.get_models()[0].get_train_dataset().data.columns[0]
    column1 = rf.get_models()[1].get_train_dataset().data.columns[0]
    try:
        t1 = deepcopy(t)
        t1[0].results[column].y += 0.1
        t1[0].name += '_xx'

        # there are also datasets, which don't contain given columns
        pdp = ExplainerArray(rf, N=10, explanation_type=t.explanation_type)
        pdp.fit()

        t_2 = pdp.transform(variables=[column, column1])

        t_2[0].plot(variable=column)
        t_2[1].plot(variable=column1)
        assert len(t_2[1].results.items()) == 1
        assert len(t_2[0].results.items()) == 1

        t[2][0].plot(variable=column)

        with pytest.raises(Exception):
            t_2[0].plot(variable=column1)

    except (Exception,):
        assert False


def test_explainer_result_array(model_array2):
    _, t = model_array2

    try:
        str(t)
        repr(t)
    except (Exception,):
        assert False


def test_explainer_result_array_plot_fail(model_array2):
    _, t = model_array2

    # there are different columns
    with pytest.raises(Exception):
        t.plot()


def test_explainer_result_array_plot(model_array2):
    _, t = model_array2

    try:
        t[2].plot()
    except (Exception,):
        assert False


def test_explainer_result_array_get(model_array2):
    rf, t = model_array2

    try:
        x = t[[0, 1]]
        y = t[list(t.results)[0].name]
        z = t[0]
    except (Exception,):
        assert False

    assert len(x) == 2
    assert isinstance(x, ModelProfileExplanationArray)
    assert isinstance(y, ModelProfileExplanation)
    assert isinstance(z, ModelProfileExplanation)


