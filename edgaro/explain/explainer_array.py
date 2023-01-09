from __future__ import annotations

import multiprocessing

from typing import Union, Optional, Literal

from edgaro.model.model import Model
from edgaro.model.model_array import ModelArray
from edgaro.explain.explainer import Explainer
from edgaro.explain.explainer_result_array import ExplanationArray, ModelProfileExplanationArray, \
    ModelPartsExplanationArray
from edgaro.base.utils import print_unbuffered


class ExplainerArray:
    """
    Create a class to calculate PDP [1]_, ALE [2]_ curves or Variable Importance for Model and ModelArray objects.

    Parameters
    ----------
    models : Model, ModelArray
        A Model/ModelArray object to calculate the curves on.
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
    models : Model, ModelArray
        A Model/ModelArray object to calculate the curves for.
    name: str
        A name of the ExplainerArray, by default it is a Model/ModelArray name.
    sub_calculators : list[Explainer, ExplainerArray], optional
        A list of calculators for nested Datasets/DatasetArrays.
    N : int, optional
        Number of observations that will be sampled from the test Dataset before the calculation of profiles
        (PDP/ALE curves). None means all data.
    explanation_type : {'PDP', 'ALE', 'VI'}, default='PDP'
        An explanation type to be calculated.
    verbose : bool
        Print messages during calculations.
    processes : int
        Number of processes for the calculation of explanations.
        If -1, it is replaced with the number of available CPU cores.
    random_state : int, optional
        Random state seed
    B : int, optional
        Number of permutation rounds to perform on each variable - applicable only if explanation_type='VI'.

    """

    def __init__(self, models: Union[Model, ModelArray], N: Optional[int] = None,
                 explanation_type: Literal['PDP', 'ALE', 'VI'] = 'PDP', verbose: bool = False, processes: int = 1,
                 random_state: Optional[int] = None, B: Optional[int] = 10) -> None:
        self.models = models
        self.sub_calculators = None
        self.name = models.name
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
        Fit the ExplainerArray object and create an explainer attribute.
        """

        def create_sub_calculator(model: Union[Model, ModelArray]):
            if isinstance(model, Model):
                calc = Explainer(model=model, N=self.N, explanation_type=self.explanation_type, verbose=self.verbose,
                                 processes=self.processes, random_state=self.random_state, B=self.B)
            else:
                calc = ExplainerArray(models=model, N=self.N, explanation_type=self.explanation_type,
                                      verbose=self.verbose, processes=self.processes, random_state=self.random_state,
                                      B=self.B)

            calc.fit()
            return calc

        self.sub_calculators = [create_sub_calculator(m) for m in self.models.get_models()]

        if self.verbose:
            print_unbuffered(f'dalex explainers inside {self.__repr__()} were created')

    def transform(self, variables=None) -> ExplanationArray:
        """
        Calculate the explanation.

        Parameters
        ----------
        variables : list[str], optional
            List of variables for which the explanation should be calculated.

        Returns
        -------
        ExplanationArray
        """
        if self.verbose:
            print_unbuffered(f'{self.explanation_type}s are being calculated in {self.__repr__()}')

        if self.sub_calculators is None:
            raise Exception('Explainer was not fitted!')

        if variables is None:
            res = [calc.transform() for calc in self.sub_calculators]
        else:
            def transform_given_variables(calc):
                if isinstance(calc, Explainer):
                    var = list(set(variables).intersection(calc.model.get_train_dataset().data.columns))
                    return calc.transform(variables=var)
                else:
                    return calc.transform(variables=variables)

            res = [transform_given_variables(calc) for calc in self.sub_calculators]

        if len(res) == 1:
            out = res[0]
        else:
            cls = ModelProfileExplanationArray if self.explanation_type == 'PDP' or \
                                                  self.explanation_type == 'ALE' else ModelPartsExplanationArray
            out = cls(
                results=res,
                name=self.name,
                explanation_type=self.explanation_type
            )

        if self.verbose:
            print_unbuffered(f'{self.explanation_type}s were calculated in {self.__repr__()}')

        return out

    def __iter__(self) -> ExplainerArray:
        self.current_i = 0
        return self

    def __next__(self) -> Explainer:
        if self.sub_calculators is None:
            raise StopIteration
        if self.current_i < len(self.sub_calculators):
            out = self.sub_calculators[self.current_i]
            self.current_i += 1
            return out
        else:
            raise StopIteration

    def __str__(self) -> str:
        length = 0
        if self.sub_calculators is not None:
            length = len(self.sub_calculators)
        return f"ExplainerArray with {length} calculators with {self.explanation_type} explanation type"

    def __repr__(self) -> str:
        length = 0
        if self.sub_calculators is not None:
            length = len(self.sub_calculators)
        return f"<ExplainerArray with {length} calculators with {self.explanation_type} explanation type>"
