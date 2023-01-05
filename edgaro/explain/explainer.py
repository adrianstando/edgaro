import dalex as dx
import numpy as np
import pandas as pd
import warnings
import sys
import os
import multiprocessing

from typing import List, Optional, Literal

from edgaro.data.dataset import Dataset
from edgaro.model.model import Model
from edgaro.explain.explainer_result import ModelProfileExplanation, Curve
from edgaro.base.utils import print_unbuffered


def _predict_func(model, data):
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    data.columns = model.get_test_dataset().data.columns
    return np.array(model.predict_proba(Dataset('', data, None)).target)


class Explainer:
    """
    The class defines Explainer for a Model object - it allows to calculate PDP [1]_ or ALE [2]_ curves.

    Parameters
    ----------
    model : Model
        A Model object to calculate explanations on.
    N : int, optional, default=None
        Number of observations that will be sampled from the test Dataset before the calculation of profiles
        (PDP/ALE curves). None means all data.
    curve_type : {'PDP', 'ALE'}, default='PDP'
        A curve type to be calculated.
    verbose : bool, default=False
        Print messages during calculations.

    Attributes
    ----------
    model : Model
        A Model object to calculate explanations on.
    name: str
        A name of the Explainer, by default it is a Model name.
    N : int, optional, default=None
        Number of observations that will be sampled from the test Dataset before the calculation of profiles
        (PDP/ALE curves). None means all data.
    curve_type : {'PDP', 'ALE'}
        A curve type to be calculated.
    verbose : bool
        Print messages during calculations.
    explainer : dx.Explainer, optional
        An explainer object from `dalex` package.
    processes : int, default=1
        Number of processes for the calculation of explanations.
        If -1, it is replaced with the number of available CPU cores.

    References
    ----------
    .. [1] https://ema.drwhy.ai/partialDependenceProfiles.html
    .. [2] https://ema.drwhy.ai/accumulatedLocalProfiles.html

    """

    def __init__(self, model: Model, N: Optional[int] = None, curve_type: Literal['PDP', 'ALE'] = 'PDP',
                 verbose: bool = False, processes: int = 1) -> None:
        self.model = model
        self.explainer = None
        self.name = model.name
        self.N = N
        self.curve_type = curve_type
        self.verbose = verbose
        self.processes = processes

        if self.processes == -1:
            self.processes = multiprocessing.cpu_count()

    def fit(self) -> None:
        """
        Fit the Explainer object and create an explainer attribute.
        """

        dataset = self.model.get_test_dataset()

        if dataset is None:
            raise Exception('Error with dataset!')

        if dataset.target is None:
            raise Exception('Target data is not provided!')
        if dataset.data is None:
            raise Exception('Data in dataset is not provided!')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if not self.verbose:
                sys.stdout = open(os.devnull, 'w')

            self.explainer = dx.Explainer(self.model, dataset.data, dataset.target, label=dataset.name,
                                          verbose=self.verbose, predict_function=_predict_func)
            if not self.verbose:
                sys.stdout = sys.__stdout__

        if self.verbose:
            print_unbuffered(f'dalex explainer inside {self.__repr__()} was created with {dataset.name}')

    def transform(self, variables: Optional[List[str]] = None) -> ModelProfileExplanation:
        """
        Calculate the curve.

        Parameters
        ----------
        variables : list[str], optional
            List of variables for which the curves should be calculated.

        Returns
        -------
        ModelProfileExplanation
        """
        if self.explainer is None:
            raise Exception('Explainer was not fitted!')
        category_colnames_base = self.model.get_category_colnames()
        dict_output = {}

        if variables is None:
            variables = list(self.model.get_train_dataset().data.columns)

        if self.curve_type == 'PDP':
            curve_type = 'partial'
        elif self.curve_type == 'ALE':
            curve_type = 'accumulated'
        else:
            raise Exception('Wrong curve type!')

        if self.verbose:
            print_unbuffered(f'{self.curve_type} is being calculated in {self.__repr__()} for '
                             f'{self.model.get_test_dataset().name}')

        category_colnames = list(set(variables).intersection(set(category_colnames_base)))
        if len(category_colnames) > 0:
            for col in category_colnames:
                out_category = self.explainer.model_profile(verbose=False, variables=[col],
                                                            variable_type='categorical',
                                                            N=self.N, type=curve_type,
                                                            processes=self.processes)
                y = out_category.result['_yhat_']
                x = out_category.result['_x_']
                dict_output[str(col)] = Curve(x, y)

        other_colnames = list(set(variables).difference(set(category_colnames_base)))
        if len(other_colnames) > 0:
            out_others = self.explainer.model_profile(verbose=False, variables=other_colnames,
                                                      variable_type='numerical',
                                                      N=self.N, type=curve_type,
                                                      processes=self.processes)
            variable_names = out_others.result['_vname_'].unique()
            y = out_others.result['_yhat_']
            x = out_others.result['_x_']
            length = y.size / len(variable_names)
            for i in range(len(variable_names)):
                lower = int(i * length)
                higher = int((i + 1) * length)
                dict_output[str(variable_names[i])] = Curve(x[lower:higher], y[lower:higher])

        if self.verbose:
            print_unbuffered(f'{self.curve_type} was calculated calculated in {self.__repr__()} for '
                             f'{self.model.get_test_dataset().name}')

        return ModelProfileExplanation(dict_output, self.name, self.model.get_category_colnames())

    def __str__(self) -> str:
        return f"Explainer for model {self.name} with {self.curve_type} curve type"

    def __repr__(self) -> str:
        return f"<Explainer for model {self.name} with {self.curve_type} curve type>"
