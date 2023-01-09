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
from edgaro.explain.explainer_result import Explanation, ModelProfileExplanation, Curve, ModelPartsExplanation
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
    explanation_type : {'PDP', 'ALE', 'VI'}, default='PDP'
        An explanation type to be calculated (`PDP` - Partial Dependence Profile, `ALE` - Accumulated Local Effects,
        `VI` - Variable Importance)
    verbose : bool, default=False
        Print messages during calculations.
    processes : int, default=1
        Number of processes for the calculation of explanations.
        If -1, it is replaced with the number of available CPU cores.
    random_state : int, optional, default=None
        Random state seed.
    B : int, optional, default=10
        Number of permutation rounds to perform on each variable - applicable only if explanation_type='VI'.

    Attributes
    ----------
    model : Model
        A Model object to calculate explanations on.
    name: str
        A name of the Explainer, by default it is a Model name.
    N : int, optional, default=None
        Number of observations that will be sampled from the test Dataset before the calculation of profiles
        (PDP/ALE curves). None means all data.
    explanation_type : {'PDP', 'ALE', 'VI'}
        An explanation type to be calculated.
    verbose : bool
        Print messages during calculations.
    explainer : dx.Explainer, optional
        An explainer object from `dalex` package.
    processes : int
        Number of processes for the calculation of explanations.
        If -1, it is replaced with the number of available CPU cores.
    random_state : int, optional
        Random state seed
    B : int, optional
        Number of permutation rounds to perform on each variable - applicable only if explanation_type='VI'.

    References
    ----------
    .. [1] https://ema.drwhy.ai/partialDependenceProfiles.html
    .. [2] https://ema.drwhy.ai/accumulatedLocalProfiles.html

    """

    def __init__(self, model: Model, N: Optional[int] = None, explanation_type: Literal['PDP', 'ALE', 'VI'] = 'PDP',
                 verbose: bool = False, processes: int = 1, random_state: Optional[int] = None,
                 B: Optional[int] = 10) -> None:
        self.model = model
        self.explainer = None
        self.name = model.name
        self.N = N
        self.explanation_type = explanation_type
        self.verbose = verbose
        self.processes = processes
        self.random_state = random_state
        self.B = B

        if self.processes == -1:
            self.processes = multiprocessing.cpu_count()

    def fit(self) -> None:
        """
        Fit the Explainer object and create an explainer attribute.
        """

        dataset = self.model.get_test_dataset()
        dataset = self.model.transform_target(dataset) # ??? VI residual

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

    def transform(self, variables: Optional[List[str]] = None) -> Explanation:
        """
        Calculate the explanation.

        Parameters
        ----------
        variables : list[str], optional
            List of variables for which the explanation should be calculated.

        Returns
        -------
        Explanation
        """
        if self.explainer is None:
            raise Exception('Explainer was not fitted!')
        category_colnames_base = self.model.get_category_colnames()
        dict_output = {}

        if variables is None:
            variables = list(self.model.get_train_dataset().data.columns)

        if self.verbose:
            print_unbuffered(f'{self.explanation_type} is being calculated in {self.__repr__()} for '
                             f'{self.model.get_test_dataset().name}')

        if self.explanation_type == 'PDP' or self.explanation_type == 'ALE':
            explanation_type = 'partial' if self.explanation_type == 'PDP' else 'accumulated'

            self.__transform_curve_category(dict_output, variables, category_colnames_base, explanation_type)
            self.__transform_curve_other(dict_output, variables, category_colnames_base, explanation_type)

            if self.verbose:
                print_unbuffered(f'{self.explanation_type} was calculated calculated in {self.__repr__()} for '
                                 f'{self.model.get_test_dataset().name}')

            return ModelProfileExplanation(dict_output, self.name, self.model.get_category_colnames())
        elif self.explanation_type == 'VI':
            explanation_type = 'variable_importance'
            self.__transform_feature_importance(dict_output, variables, explanation_type)

            if self.verbose:
                print_unbuffered(f'{self.explanation_type} was calculated calculated in {self.__repr__()} for '
                                 f'{self.model.get_test_dataset().name}')

            return ModelPartsExplanation(dict_output, self.name)
        else:
            raise Exception('Wrong curve type!')

    def __transform_curve_category(self, dict_output, variables, category_colnames_base, explanation_type):
        category_colnames = list(set(variables).intersection(set(category_colnames_base)))
        if len(category_colnames) > 0:
            out_category = self.explainer.model_profile(verbose=False, variables=category_colnames,
                                                        variable_type='categorical',
                                                        N=self.N, type=explanation_type,
                                                        processes=self.processes, random_state=self.random_state)
            y = out_category.result['_yhat_']
            x = out_category.result['_x_']
            variable_names = out_category.result['_vname_'].unique()
            for i in range(len(variable_names)):
                dict_output[str(variable_names[i])] = Curve(x, y)

    def __transform_curve_other(self, dict_output, variables, category_colnames_base, explanation_type):
        other_colnames = list(set(variables).difference(set(category_colnames_base)))
        if len(other_colnames) > 0:
            out_others = self.explainer.model_profile(verbose=False, variables=other_colnames,
                                                      variable_type='numerical',
                                                      N=self.N, type=explanation_type,
                                                      processes=self.processes, random_state=self.random_state)
            variable_names = out_others.result['_vname_'].unique()
            y = out_others.result['_yhat_']
            x = out_others.result['_x_']
            length = y.size / len(variable_names)
            for i in range(len(variable_names)):
                lower = int(i * length)
                higher = int((i + 1) * length)
                dict_output[str(variable_names[i])] = Curve(x[lower:higher], y[lower:higher])

    def __transform_feature_importance(self, dict_output, variables, explanation_type):
        out = self.explainer.model_parts(verbose=False, variables=variables,
                                         N=self.N, B=self.B, type=explanation_type,
                                         processes=self.processes, random_state=self.random_state)
        out = out.result
        out = out[(out.variable != '_full_model_') & (out.variable != '_baseline_')]
        out = out[['variable', 'dropout_loss']]

        for index, row in out.iterrows():
            dict_output[row[0]] = row[1]

    def __str__(self) -> str:
        return f"Explainer for model {self.name} with {self.explanation_type}  explanation type"

    def __repr__(self) -> str:
        return f"<Explainer for model {self.name} with {self.explanation_type} explanation type>"
