from __future__ import annotations

import math
import matplotlib.pyplot as plt

from typing import List, Literal, Optional, Union, Tuple

import numpy as np

from edgaro.explain.explainer_result import ExplainerResult


class ExplainerResultArray:
    """
    The class which represent the PDP/ALE curves for all variables in Model/ModelArray object.

    Parameters
    ----------
    results : list[ExplainerResult, ExplainerResultArray]
        A list of ExplainerResult/ExplainerResultArray with results.
    name : str
        The name of ExplainerResultArray. It is best if it is a Model/ModelArray name.
    curve_type : {'PDP', 'ALE'}, default='PDP'
        A curve type.

    Attributes
    ----------
    results : list[ExplainerResult, ExplainerResultArray]
        A list of ExplainerResult/ExplainerResultArray with results.
    name : str
        The name of ExplainerResultArray. It is best if it is a Model/ModelArray name.
    curve_type : {'PDP', 'ALE'}
        A curve type.

    """

    def __init__(self, results: List[Union[ExplainerResult, ExplainerResultArray]], name: str,
                 curve_type: Literal['PDP', 'ALE'] = 'PDP') -> None:
        self.results = results
        self.name = name
        self.curve_type = curve_type

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, key: Union[Union[str, int], List[Union[str, int]]]) \
            -> Optional[Union[ExplainerResult, ExplainerResultArray]]:
        if isinstance(key, list):
            outs = [self.__getitem__(k) for k in key]
            outs = [o for o in outs if o is not None]
            if len(outs) == 0:
                return None
            else:
                return ExplainerResultArray(results=outs, name=self.name + "_subset", curve_type=self.curve_type)
        elif isinstance(key, str):
            for result in self.results:
                if result.name == key:
                    return result
        elif isinstance(key, int):
            if key <= len(self.results):
                return self.results[key]
        return None

    @staticmethod
    def __check_colnames(res_array, col_names_in):
        if isinstance(res_array, ExplainerResult):
            if len(col_names_in) == 0:
                col_names_in += res_array.results.keys()
                return True
            else:
                return set(res_array.results.keys()) == set(col_names_in)
        else:
            for res in res_array.results:
                correct = ExplainerResultArray.__check_colnames(res, col_names_in)
                if not correct:
                    return False
            return True

    @staticmethod
    def __find_matching_explainer_result_array(res_array, results_in, variables):
        if isinstance(res_array, ExplainerResult):
            if np.all([var in res_array.results.keys() for var in variables]):
                results_in.append(res_array)
        else:
            for res in res_array.results:
                ExplainerResultArray.__find_matching_explainer_result_array(res, results_in, variables)

    def plot(self, variables: Optional[List[str]] = None, n_col: int = 3, figsize: Optional[Tuple[int, int]] = None):
        """
        The function plots the PDP/ALE curves for given variables using all available Curves in the object.

        Parameters
        ----------
        variables : list[str], optional, default=None
            Variables for which the plot should be generated. If None, plots for all variables are generated if all the
            available ExplainerResult objects have exactly the same set of column names.
        n_col : int, default=3
            Number of columns in the final plot.
        figsize : tuple(int, int), optional, default=None
            The size of a figure. If None, the figure size is calculates as (8 * n_col, 8 * n_rows).
        """
        if variables is None:
            col_names = []
            if not ExplainerResultArray.__check_colnames(self, col_names):
                raise Exception('The ResultArray contains curves for models with different column names!')
            variables = col_names

        results = []
        ExplainerResultArray.__find_matching_explainer_result_array(self, results, variables)

        if len(results) == 0:
            raise Exception('There are not matching models!')

        n_rows = math.ceil(len(variables) / n_col)
        if figsize is None:
            figsize = (8 * n_col, 8 * n_rows)
        fig, axes = plt.subplots(n_rows, n_col, figsize=figsize)

        for i in range(len(variables)):
            ax_to_pass = axes[math.floor(i / n_col)][i - n_col * math.floor(i / n_col)] \
                if n_rows > 1 else axes[math.floor(i / n_col)]

            results[0].plot(
                add_plot=[results[j] for j in range(1, len(results))],
                variable=variables[i],
                ax=ax_to_pass,
                show_legend=False
            )

        fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        fig.legend([x.name for x in results], ncol=n_col, loc='lower center')
        plt.suptitle(f"PDP curves for {self.name}", fontsize=18)

    def __str__(self) -> str:
        return f"ExplainerResultArray {self.name} for {len(self.results)} variables: {list(self.results)} with {self.curve_type} curve type"

    def __repr__(self) -> str:
        return f"<ExplainerResultArray {self.name} with {self.curve_type} curve type>"
