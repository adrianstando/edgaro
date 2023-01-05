=============
User manual
=============

This manual covers the most important use cases of the package. For
details and additional input parameters, see the full documentation.

Defining a dataset
------------------

To start working with the package, you can use an example dataset. You
can load it using the code below.

::

   from edgaro.data.dataset import load_mammography

   df = load_mammography()

It is also possible to create a ``Dataset`` object from a
``pandas.DataFrame`` object in your Python session:

::

   # X - your dataframe with features, pd.DataFrame
   # y - your target variable, pd.Series

   from edgaro.data.dataset import Dataset

   df = Dataset(X, y)

Apart from that, you can create a ``Dataset`` object from a *\*.csv*
file or from a dataset defined in OpenML:

::

   from edgaro.data.dataset import DatasetFromCSV, DatasetFromOpenML

   df = DatasetFromCSV('your_path')
   df = DatasetFromOpenML(task_id=1)

The ``Dataset`` object offers many handy functionalities, for example
splitting into train and test datasets (inside the object), removing
``None`` values and calculating the Imbalance Ratio.

::

   df.remove_nans()
   df.train_test_split(test_size=0.2)
   df.imbalance_ratio

To represent an array of the ``Dataset`` objects, ``DatasetArray`` is
used. It is composed of a list of ``Dataset``\ s. The object offers
vectorized functions described above, like splitting into train and test
datasets (inside the object) and removing ``None`` values.

::

   from edgaro.data.dataset_array import DatasetArray

   df_array = DatasetArray([df1, df2, df3])
   df_array.remove_nans()
   df_array.train_test_split(test_size=0.2)

It is also possible to load an example ``DatasetArray``, which can be
used for benchmarking.

::

   from edgaro.data.dataset_array import load_benchmarking_set

   df_array = load_benchmarking_set()

It is also possible to load a benchmark suite from OpenML to
``DatasetArray`` object.

::

   from edgaro.data.dataset_array import DatasetArrayFromOpenMLSuite

   df_array = DatasetArrayFromOpenMLSuite(suite_name='OpenML100')

Balancing datasets
------------------

To balance a dataset, a ``Transformer`` abstract class is defined. It is
possible to create custom balancing methods (by extending this class) or
you can use methods implemented in ``imblearn`` library. Firstly, a
transformer has to be fitted with a dataset, and then by calling
*transform* method it can be balanced.

::

   from edgaro.balancing.transformer import TransformerFromIMBLEARN
   from imblearn.under_sampling import RandomUnderSampler

   transformer = TransformerFromIMBLEARN(
       RandomUnderSampler(sampling_strategy=1, random_state=42)
   )
   transformer.fit(dataset)
   dataset_transformed = transformer.transform(dataset)

You can also define a custom suffix that will be added to balanced
``Dataset``\ ’s name.

::

   from edgaro.balancing.transformer import TransformerFromIMBLEARN
   from imblearn.under_sampling import RandomUnderSampler

   transformer = TransformerFromIMBLEARN(
       RandomUnderSampler(sampling_strategy=1, random_state=42),
       name_sufix='_new_sufix123'
   )

There is one extension worth mentioning - it is
``NestedAutomaticTransformer``. It behaves like a ``Transformer``
object, but it wraps inside a few balancing methods. Moreover, that
object, based on ``n_per_method`` argument, automatically set an
intermediate Imbalance Ratio (to investigate how the future models
change with changing IR). For example, let an original dataset have
:math:`IR=10` and :math:`n\_per\_method=3`. Then, the IR will be set to
:math:`[7, 4, 1]`. If you set *keep_original_dataset* to ``True``, the
original will be preserved in the result ``DatasetArray`` object.

The ``BasicAutomaticTransformer`` class contains three most popular
balancing techniques (Random UnderSampling, Random OverSampling, SMOTE).

::

   from edgaro.balancing.nested_transformer import BasicAutomaticTransformer

   transformer = BasicAutomaticTransformer()
   transformer.fit(dataset)
   dataset_transformed = transformer.transform(dataset)

The ``balancing`` submodule also offers the interface of an array. If
your input is a ``DatasetArray`` object or you want to balance input
with different parameters (in that case both ``DatasetArray`` and
``Dataset`` is correct) you should use this class.

To apply the same balancing technique to a ``DatasetArray`` you have to
use ``TransformerArray`` class:

::

   from edgaro.balancing.transformer_array import TransformerArray
   from edgaro.balancing.transformer import TransformerFromIMBLEARN
   from imblearn.under_sampling import RandomUnderSampler

   transformer = TransformerArray(TransformerFromIMBLEARN(
       RandomUnderSampler(sampling_strategy=1, random_state=42)
   ))
   transformer.fit(dataset_array)
   dataset_array_transformed = transformer.transform(dataset_array)

You can set, as in the ``Transformer`` class, the suffixes:

::

   from edgaro.balancing.transformer_array import TransformerArray
   from edgaro.balancing.transformer import TransformerFromIMBLEARN
   from imblearn.under_sampling import RandomUnderSampler

   transformer = TransformerArray(TransformerFromIMBLEARN(
       RandomUnderSampler(sampling_strategy=1, random_state=42),
       dataset_suffixes=['_suffix1', '_suffix2']
   ))

You can also set parameters - their nested structure should match the
``DatasetArray`` structure.

::

   from edgaro.balancing.transformer_array import TransformerArray
   from edgaro.balancing.transformer import TransformerFromIMBLEARN
   from imblearn.under_sampling import RandomUnderSampler
   from edgaro.data.dataset_array import DatasetArray

   dataset_array = DatasetArray([dataset1, dataset2])

   transformer = TransformerArray(TransformerFromIMBLEARN(
       RandomUnderSampler(),
       parameters=[
           [
               {'sampling_strategy': 0.98},
               {'sampling_strategy': 1},
               {'sampling_strategy': 0.9, 'random_state': 42}
           ] for _ in range(2)
       ]
   ))
   transformer.fit(dataset_array)
   dataset_array_transformed = transformer.transform(dataset_array)

Note: if a ``Dataset`` object was train-test-split, the balancing
methods are applied only on the training datasets and the test datasets
remain untouched.

Training a model
----------------

The classes in ``model`` module have a similar interface to those in
``balancing`` module. There is a *Model* class which is an abstract
class and can be extended with any ML model implementation. One possible
solution is to use *scikit-learn* models, which can be used by using
``ModelFromSKLEARN`` class.

In this module, there is also a ``ModelArray`` class, which behaves very
similarly to the ``TransformerArray`` class. However, instead of
transforming a ``Dataset``, the class predictions are made or
probabilities are returned. The returned objects are also ``Dataset``
objects.

::

   from edgaro.model.model import ModelFromSKLEARN
   from sklearn.ensemble import RandomForestClassifier

   model = ModelFromSKLEARN(RandomForestClassifier())
   model.fit(dataset)
   predictions = model.predict(dataset)
   predictions_probability = model.predict_proba(dataset)

An example of using ``ModelArray`` - that means the situation when the
input is ``DatasetArray`` object.

::

   from edgaro.model.model import ModelFromSKLEARN
   from edgaro.model.model_array import ModelArray
   from sklearn.ensemble import RandomForestClassifier

   model = ModelArray(ModelFromSKLEARN(RandomForestClassifier()))
   model.fit(dataset_array)
   predictions = model.predict(dataset_array)
   predictions_probability = model.predict_proba(dataset_array)

There is also a function to evaluate the model. If the input parameter
is not provided and the object was train-test-split, the evaluation is
made on the test dataset.

::

   from edgaro.model.model import ModelFromSKLEARN
   from edgaro.model.model_array import ModelArray
   from sklearn.ensemble import RandomForestClassifier

   model = ModelArray(ModelFromSKLEARN(RandomForestClassifier()))
   model.fit(dataset_array)
   model.evaluate()

   model = ModelFromSKLEARN(RandomForestClassifier())
   model.fit(dataset)
   model.evaluate()

Explaining and comparing explanations
-------------------------------------

To create explanations (PDP / ALE curves), the ``Explainer`` and
``ExplainerArray`` classes are provided. The first one should be used
when you only have one ``Model`` object and the latter when you have
``ModelArray``. The interface in ``explain`` module is similar to that
in ``model``:

::

   from edgaro.explain.explainer import Explainer

   exp = Explainer(model)
   exp.fit()
   explanation = exp.transform()

In case of input ``ModelArray``:

::

   from edgaro.explain.explainer import Explainer
   from edgaro.explain.explainer_array import ExplainerArray

   exp = ExplainerArray(model_array)
   exp.fit()
   explanation = exp.transform()

These functions return ``ModelProfileExplanation`` and ``ModelProfileExplanationArray``
objects that make it possible to compare explanations and visualise
them.

In case of a single ``Model``:

::

   from edgaro.explain.explainer import Explainer

   exp = Explainer(model)
   exp.fit()
   explanation = exp.transform()
   explanation.plot(variable='Col1')

In case of a ``ModelArray``:

::

   from edgaro.explain.explainer import Explainer
   from edgaro.explain.explainer_array import ExplainerArray

   exp = ExplainerArray(model_array)
   exp.fit()
   explanation = exp.transform()
   explanation.plot(variables=['Col1', 'Col2'])

In order to calculate the distance between the curves:

::

   from edgaro.explain.explainer import Explainer
   from edgaro.explain.explainer_array import ExplainerArray

   exp = ExplainerArray(model_array)
   exp.fit()
   explanation = exp.transform()
   explanation[0].compare(explanation[1], variable='Col1')

To create benchmarking summary plots, use for example the code:

::

   from edgaro.explain.explainer import Explainer
   from edgaro.explain.explainer_array import ExplainerArray

   exp = ExplainerArray(model_array)
   exp.fit()
   explanation = exp.transform()
   explanation.plot_aggregate(['SMOTE ', 'RandomOversample ', 'RandomUndersample ')

The elements of the list in the function *plot_aggregate()* are regular
expressions to match and group explanations matching the. In this case,
the output will be a boxplot with three boxes - one per method.

Note: if the input data objects were train-test-split, the explanations
are calculated on the test dataset.
