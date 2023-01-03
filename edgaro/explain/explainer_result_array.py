from __future__ import annotations

import math
import re
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Literal, Optional, Union, Tuple

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
    def __find_matching_explainer_result_array(res_array, results_in, variables, model_filter):
        if isinstance(res_array, ExplainerResult):
            if np.all([var in res_array.results.keys() for var in variables]):
                if model_filter is not None:
                    if re.search(model_filter, res_array.name) is not None:
                        results_in.append(res_array)
                else:
                    results_in.append(res_array)
        else:
            for res in res_array.results:
                ExplainerResultArray.__find_matching_explainer_result_array(res, results_in, variables, model_filter)

    def plot(self, variables: Optional[List[str]] = None, n_col: int = 3, figsize: Optional[Tuple[int, int]] = None,
             model_filter: Optional[str] = None, index_base: Union[str, int] = -1,):
        """
        The function plots the PDP/ALE curves for given variables using all available Curves in the object.

        Parameters
        ----------
        index_base : int, str, default=-1
            Index of a curve to be a base for comparisons.
        variables : list[str], optional, default=None
            Variables for which the plot should be generated. If None, plots for all variables are generated if all the
            available ExplainerResult objects have exactly the same set of column names.
        n_col : int, default=3
            Number of columns in the final plot.
        figsize : tuple(int, int), optional, default=None
            The size of a figure. If None, the figure size is calculates as (8 * n_col, 8 * n_rows).
        model_filter : str, optional, default=None
            A regex expression to filter the names of the ExplainerResult objects for comparing.
        """
        if variables is None:
            col_names = []
            if not ExplainerResultArray.__check_colnames(self, col_names):
                raise Exception('The ResultArray contains curves for models with different column names!')
            variables = col_names

        results = []
        ExplainerResultArray.__find_matching_explainer_result_array(self, results, variables, model_filter)

        if len(results) == 0:
            raise Exception('There are not matching models!')

        base_model = self.results[index_base]
        results.insert(0, base_model)

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
        plt.suptitle(f"{self.curve_type} curves for {self.name}", fontsize=18)

    def compare(self, variable: Optional[Union[str, List[str]]] = None, index_base: Union[str, int] = -1,
                return_raw: bool = True, return_raw_per_variable: bool = True, model_filter: Optional[str] = None) \
            -> List[Union[float, List]]:
        """
        The function compares the curves in the array.

        Parameters
        ----------
        variable : list[str], optional, default=None
            List of variable names to calculate the metric distances. If None, the metrics are calculated for
            all the columns in this object.
        index_base : int, str, default=-1
            Index of a curve to be a base for comparisons.
        return_raw : bool, default=True
            If True, the metrics for each of the model are returned. Otherwise, the mean of the values is returned.
        return_raw_per_variable : bool, default=True
            If True, the metrics for each of the variables are returned. Otherwise, the mean of the values is returned.
        model_filter : str, optional, default=None
            A regex expression to filter the names of the ExplainerResult objects for comparing.

        Returns
        -------
        list[float, list]

        """

        if isinstance(self.results[index_base], ExplainerResult):
            if isinstance(index_base, int) and index_base < 0:
                index_base = self.results.index(self.results[index_base])

            def filter_objects(obj):
                if model_filter is not None and \
                        re.search(model_filter, obj.name) is None:
                    return False
                return True

            def flatten(lst):
                out = []
                for i in range(len(lst)):
                    if not (isinstance(lst[i], list) or isinstance(lst[i], ExplainerResultArray)):
                        out.append(lst[i])
                    else:
                        tmp = flatten(lst[i])
                        out = out + tmp
                return out

            base_model = self[index_base]
            if base_model is None:
                raise Exception('Wrong index_base argument!')

            res = flatten(self.results)
            res.remove(self.results[index_base])

            if return_raw:
                out = []
                for i in range(len(res)):
                    if not filter_objects(res[i]):
                        continue

                    if not return_raw_per_variable:
                        out.append(base_model.compare(res[i], variable=variable,
                                                      return_raw_per_variable=False)[0])
                    else:
                        out.append(base_model.compare(res[i], variable=variable,
                                                      return_raw_per_variable=True))
                return out
            else:
                tab = [res[i] for i in range(len(res)) if filter_objects(res[i])]
                return base_model.compare(tab, variable=variable, return_raw_per_variable=return_raw_per_variable)
        elif np.alltrue([isinstance(res, ExplainerResultArray) for res in self.results]):
            return [
                res.compare(variable=variable, index_base=index_base, return_raw=return_raw,
                            return_raw_per_variable=return_raw_per_variable, model_filter=model_filter)
                for res in self.results
            ]
        else:
            raise Exception('Wrong result structure!')

    def plot_summary(self, model_filters: Optional[List[str]] = None, filter_labels: [List[str]] = None,
                     variables: Optional[List[str]] = None, figsize: Optional[Tuple[int, int]] = None,
                     index_base: Union[str, int] = -1):
        """
        The function plots boxplots of comparison metrics of curves in the object.

        Parameters
        ----------
        variables : list[str], optional, default=None
            Variables for which the plot should be generated. If None, plots for all variables are generated if all the
            available ExplainerResult objects have exactly the same set of column names.
        figsize : tuple(int, int), optional, default=None
            The size of a figure. If None, the figure size is calculates as (8 * n_col, 8 * n_rows).
        model_filters : list[str], optional, default=None
            List of regex expressions to filter the names of the ExplainerResult objects for comparing.
            Each element in the list creates a new boxplot. If None, one boxplot of all results is plotted.
        filter_labels : list[str], optional, default=None
            Labels of model filters.
        index_base : int, str, default=-1
            Index of a curve to be a base for comparisons.
        """

        fig, ax = plt.subplots(figsize=figsize)
        plt.suptitle(f'Summary of {self.name}')
        plt.xlabel('Filter')
        plt.ylabel(r'VOD values [$10^{-5}$]')

        def format_func(value, tick_number):
            return str(int(value * 10 ** 5))

        def flatten(lst):
            out = []
            for i in range(len(lst)):
                if not isinstance(lst[i], list):
                    out.append(lst[i])
                else:
                    tmp = flatten(lst[i])
                    out = out + tmp
            return out

        ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

        if model_filters is None:
            results = self.compare(variable=variables, index_base=index_base,
                                   return_raw=True, return_raw_per_variable=True)

            results = flatten(results)

            if filter_labels is not None and len(filter_labels) == 1 and \
                    filter_labels is not None and len(filter_labels) == 1:
                plt.boxplot(results, labels=filter_labels, patch_artist=True)
            else:
                plt.boxplot(results, labels=['All values'], patch_artist=True)

        else:
            results = []
            for f in model_filters:
                tmp_out = self.compare(variable=variables, index_base=index_base,
                                       return_raw=True, return_raw_per_variable=True, model_filter=f)
                tmp_out = flatten(tmp_out)
                results.append(tmp_out)

            if filter_labels is not None:
                if len(filter_labels) == len(model_filters):
                    plt.boxplot(results, labels=filter_labels, patch_artist=True)
                else:
                    raise Exception('Incorrect length of filter_labels!')
            else:
                plt.boxplot(results, patch_artist=True, labels=model_filters)

    def __str__(self) -> str:
        return f"ExplainerResultArray {self.name} for {len(self.results)} variables: {list(self.results)} with {self.curve_type} curve type"

    def __repr__(self) -> str:
        return f"<ExplainerResultArray {self.name} with {self.curve_type} curve type>"
