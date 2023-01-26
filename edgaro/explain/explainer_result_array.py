from __future__ import annotations

import math
import re

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

from typing import List, Literal, Optional, Union, Tuple
from matplotlib.axes import Axes
from statsmodels.stats.multitest import fdrcorrection

from edgaro.explain.explainer_result import ModelProfileExplanation, ModelPartsExplanation


class ExplanationArray(ABC):
    def __int__(self) -> None:
        pass

    @abstractmethod
    def plot(self) -> None:
        pass

    @abstractmethod
    def compare(self) -> List[Union[float, list]]:
        pass

    @abstractmethod
    def plot_summary(self) -> None:
        pass


class ModelProfileExplanationArray(ExplanationArray):
    """
    The class which represent the PDP/ALE curves for all variables in Model/ModelArray object.

    Parameters
    ----------
    results : list[ModelProfileExplanation, ModelProfileExplanationArray]
        A list of ModelProfileExplanation/ModelProfileExplanationArray with results.
    name : str
        The name of ModelProfileExplanationArray. It is best if it is a Model/ModelArray name.
    explanation_type : {'PDP', 'ALE'}, default='PDP'
        A curve type.

    Attributes
    ----------
    results : list[ModelProfileExplanation, ModelProfileExplanationArray]
        A list of ModelProfileExplanation/ModelProfileExplanationArray with results.
    name : str
        The name of ModelProfileExplanationArray. It is best if it is a Model/ModelArray name.
    explanation_type : {'PDP', 'ALE'}
        A curve type.

    """

    def __init__(self, results: List[Union[ModelProfileExplanation, ModelProfileExplanationArray]], name: str,
                 explanation_type: Literal['PDP', 'ALE'] = 'PDP') -> None:
        self.results = results
        self.name = name
        self.explanation_type = explanation_type

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, key: Union[Union[str, int], List[Union[str, int]]]) \
            -> Optional[Union[ModelProfileExplanation, ModelProfileExplanationArray]]:
        if isinstance(key, list):
            outs = [self.__getitem__(k) for k in key]
            outs = [o for o in outs if o is not None]
            if len(outs) == 0:
                return None
            else:
                return ModelProfileExplanationArray(results=outs, name=self.name + "_subset",
                                                    explanation_type=self.explanation_type)
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
        if isinstance(res_array, ModelProfileExplanation):
            if len(col_names_in) == 0:
                col_names_in += res_array.results.keys()
                return True
            else:
                return set(res_array.results.keys()) == set(col_names_in)
        else:
            for res in res_array.results:
                correct = ModelProfileExplanationArray.__check_colnames(res, col_names_in)
                if not correct:
                    return False
            return True

    @staticmethod
    def __find_matching_explainer_result_array(res_array, results_in, variables, model_filter):
        if isinstance(res_array, ModelProfileExplanation):
            if np.all([var in res_array.results.keys() for var in variables]):
                if model_filter is not None:
                    if re.search(model_filter, res_array.name) is not None:
                        results_in.append(res_array)
                else:
                    results_in.append(res_array)
        else:
            for res in res_array.results:
                ModelProfileExplanationArray.__find_matching_explainer_result_array(res, results_in, variables,
                                                                                    model_filter)

    def plot(self, variables: Optional[List[str]] = None, n_col: int = 3, figsize: Optional[Tuple[int, int]] = None,
             model_filter: Optional[str] = None, index_base: Union[str, int] = -1, centered: bool = False):
        """
        The function plots the PDP/ALE curves for given variables using all available Curves in the object.

        Parameters
        ----------
        index_base : int, str, default=-1
            Index of a curve to be a base for comparisons.
        variables : list[str], optional, default=None
            Variables for which the plot should be generated. If None, plots for all variables are generated if all the
            available ModelProfileExplanation objects have exactly the same set of column names.
        n_col : int, default=3
            Number of columns in the final plot.
        figsize : tuple(int, int), optional, default=None
            The size of a figure. If None, the figure size is calculates as (8 * n_col, 8 * n_rows).
        model_filter : str, optional, default=None
            A regex expression to filter the names of the ModelProfileExplanation objects for comparing.
        centered : bool, default = False
            If True, the plots will be centered to start at 0.

        """
        if variables is None:
            col_names = []
            if not ModelProfileExplanationArray.__check_colnames(self, col_names):
                raise Exception('The ResultArray contains curves for models with different column names!')
            variables = col_names

        results = []
        ModelProfileExplanationArray.__find_matching_explainer_result_array(self, results, variables, model_filter)

        if len(results) == 0:
            raise Exception('There are not matching models!')

        base_model = self.results[index_base]

        try:
            results.remove(base_model)
        except (Exception,):
            pass

        results.insert(0, base_model)

        n_rows = math.ceil(len(variables) / n_col)
        if figsize is None:
            figsize = (8 * n_col, 8 * n_rows)
        fig, axes = plt.subplots(n_rows, n_col, figsize=figsize)

        for i in range(len(variables)):
            if n_rows > 1:
                ax_to_pass = axes[math.floor(i / n_col)][i - n_col * math.floor(i / n_col)]
            else:
                if n_col > 1:
                    ax_to_pass = axes[math.floor(i / n_col)]
                else:
                    ax_to_pass = axes

            results[0].plot(
                add_plot=[results[j] for j in range(1, len(results))],
                variable=variables[i],
                ax=ax_to_pass,
                show_legend=False,
                centered=centered
            )

        fig.tight_layout(rect=[0, 0.05, 1, 0.97])
        fig.legend([x.name for x in results], ncol=n_col, loc='lower center')
        plt.suptitle(f"{self.explanation_type} curves for {self.name}", fontsize=18)

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
            A regex expression to filter the names of the ModelProfileExplanation objects for comparing.

        Returns
        -------
        list[float, list]

        """

        if isinstance(self.results[index_base], ModelProfileExplanation):
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
                    if not (isinstance(lst[i], list) or isinstance(lst[i], ModelProfileExplanationArray)):
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
        elif np.alltrue([isinstance(res, ModelProfileExplanationArray) for res in self.results]):
            return [
                res.compare(variable=variable, index_base=index_base, return_raw=return_raw,
                            return_raw_per_variable=return_raw_per_variable, model_filter=model_filter)
                for res in self.results
            ]
        else:
            raise Exception('Wrong result structure!')

    def plot_summary(self, model_filters: Optional[List[str]] = None, filter_labels: [List[str]] = None,
                     variables: Optional[List[str]] = None, figsize: Optional[Tuple[int, int]] = None,
                     index_base: Union[str, int] = -1, return_df: bool = False):
        """
        The function plots boxplots of comparison metrics of curves in the object.

        Parameters
        ----------
        variables : list[str], optional, default=None
            Variables for which the plot should be generated. If None, plots for all variables are generated if all the
            available ModelProfileExplanation objects have exactly the same set of column names.
        figsize : tuple(int, int), optional, default=None
            The size of a figure.
        model_filters : list[str], optional, default=None
            List of regex expressions to filter the names of the ModelProfileExplanation objects for comparing.
            Each element in the list creates a new boxplot. If None, one boxplot of all results is plotted.
        filter_labels : list[str], optional, default=None
            Labels of model filters.
        index_base : int, str, default=-1
            Index of a curve to be a base for comparisons.
        return_df : bool, default=False
            If True, the method returns a dataframe on which a plot is created.
        """

        fig, ax = plt.subplots(figsize=figsize)
        plt.title(f'Summary of {self.explanation_type} for {self.name}')
        plt.xlabel('Filter')
        plt.ylabel(r'SDD values [$10^{-3}$]')

        def format_func(value, tick_number):
            return str(int(value * 10 ** 3))

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

        if return_df:
            return results

    def __str__(self) -> str:
        return f"ModelProfileExplanationArray {self.name} for {len(self.results)} variables: {list(self.results)} with {self.explanation_type} curve type"

    def __repr__(self) -> str:
        return f"<ModelProfileExplanationArray {self.name} with {self.explanation_type} curve type>"


class ModelPartsExplanationArray(ExplanationArray):
    """
    The class which represent the Variable Importance for all variables in Model/ModelArray object.

    Parameters
    ----------
    results : list[ModelPartsExplanation, ModelPartsExplanationArray]
        A list of ModelPartsExplanation/ModelPartsExplanationArray with results.
    name : str
        The name of ModelPartsExplanationArray. It is best if it is a Model/ModelArray name.
    explanation_type : {'VI'}, default='VI'
        An explanation type.

    Attributes
    ----------
    results : list[ModelPartsExplanation, ModelPartsExplanationArray]
        A list of ModelPartsExplanation/ModelPartsExplanationArray with results.
    name : str
        The name of ModelPartsExplanationArray. It is best if it is a Model/ModelArray name.
    explanation_type : {'VI'}, default='VI'
        An explanation type.

    """

    def __init__(self, results: List[Union[ModelPartsExplanation, ModelPartsExplanationArray]],
                 name: str, explanation_type: Literal['VI'] = 'VI') -> None:
        self.results = results
        self.name = name
        self.explanation_type = explanation_type

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, key: Union[Union[str, int], List[Union[str, int]]]) \
            -> Optional[Union[ModelPartsExplanation, ModelPartsExplanationArray]]:
        if isinstance(key, list):
            outs = [self.__getitem__(k) for k in key]
            outs = [o for o in outs if o is not None]
            if len(outs) == 0:
                return None
            else:
                return ModelPartsExplanationArray(results=outs, name=self.name + "_subset",
                                                  explanation_type=self.explanation_type)
        elif isinstance(key, str):
            for result in self.results:
                if result.name == key:
                    return result
        elif isinstance(key, int):
            if key <= len(self.results):
                return self.results[key]
        return None

    def plot(self, variable: Optional[Union[str, List[str]]] = None, max_variables: Optional[int] = None,
             index_base: Union[str, int] = -1, figsize: Optional[Tuple[int, int]] = (8, 8),
             ax: Optional[Axes] = None, show_legend: bool = True, x_lim: Optional[Tuple[float, float]] = None,
             metric_precision: int = 3) -> None:
        """
        The function plots the Variable Importance profile using all ModelPartsExplanation objects.

        Parameters
        ----------
        variable : str, list[str], optional, default=None
            Variable for which the VI should be plotted. If None, the all columns is plotted.
        figsize : tuple(int, int), optional, default=(8, 8)
            Size of a figure.
        max_variables : int, optional, default=None
            Maximal number of variables from the current object to be taken into account.
        ax : matplotlib.axes.Axes, optional, default=None
            The parameter should be passed if the plot is to be created in a certain Axis. In that situation, `figsize`
            parameter is ignored.
        show_legend : bool, default=True
            The parameter indicates whether the legend should be plotted.
        x_lim : tuple(float, float), optional, default=None
            The limits of 0X axis.
        metric_precision : int, default=5
            Number of digits to round the value of the metric value.
        index_base : int, str, default=-1
            Index of an explanation to be a base for comparisons.
        """

        base = self.results[index_base]
        plots = self.results.copy()
        plots.remove(base)
        plots = ModelPartsExplanationArray.__flatten(plots)

        base.plot(variable=variable, figsize=figsize, max_variables=max_variables,
                  add_plot=plots, ax=ax, show_legend=show_legend, x_lim=x_lim, metric_precision=metric_precision)

    def compare(self, variable: Optional[Union[str, List[str]]] = None, max_variables: Optional[int] = None,
                return_raw: bool = True, index_base: Union[str, int] = -1, model_filter: Optional[str] = None) \
            -> List[Union[float, list]]:
        """
        The function compares variable importance in the array.

        Parameters
        ----------
        variable : str, list[str], optional, default=None
            List of variable names to calculate the metric distances. If None, the metrics are calculated for
            all the columns in this object.
        max_variables : int, optional, default=None
            Maximal number of variables from the current object to be taken into account.
        return_raw : bool, default=True
            If True, the p-values are returned for each model. Otherwise, the mean value is returned.
        index_base : int, str, default=-1
            Index of an explanation to be a base for comparisons.
        model_filter : str, optional, default=None
            A regex expression to filter the names of the ModelPartsExplanation objects for comparing.

        Returns
        -------
        list[float, list]

        """

        if isinstance(self.results[index_base], ModelPartsExplanation):
            if isinstance(index_base, int) and index_base < 0:
                index_base = self.results.index(self.results[index_base])

            def filter_objects(obj):
                if model_filter is not None and \
                        re.search(model_filter, obj.name) is None:
                    return False
                return True

            base_model = self[index_base]
            if base_model is None:
                raise Exception('Wrong index_base argument!')

            res = ModelPartsExplanationArray.__flatten(self.results)
            res.remove(self.results[index_base])

            res_filtered = [res[i] for i in range(len(res)) if filter_objects(res[i])]
            return base_model.compare(other=res_filtered, variable=variable, max_variables=max_variables,
                                      return_raw=return_raw)
        elif np.alltrue([isinstance(res, ModelPartsExplanationArray) for res in self.results]):
            return [
                res.compare(variable=variable, max_variables=max_variables, return_raw=return_raw,
                            index_base=index_base, model_filter=model_filter)
                for res in self.results
            ]
        else:
            raise Exception('Wrong result structure!')

    def plot_summary(self, model_filters: Optional[List[str]] = None, filter_labels: [List[str]] = None,
                     variables: Optional[List[str]] = None, max_variables: Optional[int] = None,
                     figsize: Optional[Tuple[int, int]] = None, index_base: Union[str, int] = -1,
                     significance_level: Optional[float] = None, fdr_correction: bool = True,
                     return_df: bool = False) -> None:
        """
        The function plots boxplots of comparison metrics of VI in the object if significance_level is provided.
        Otherwise, the results of the statistical test are plotted as barplots according to the significance_level.

        Parameters
        ----------
        variables : str, list[str], optional, default=None
            Variable for which the VI should be plotted. If None, the all columns is plotted.
        figsize : tuple(int, int), optional, default=(8, 8)
            Size of a figure.
        model_filters : list[str], optional, default=None
            List of regex expressions to filter the names of the ModelPartsExplanation objects for comparing.
            Each element in the list creates a new boxplot. If None, one boxplot / barplot of all results is plotted.
        filter_labels : list[str], optional, default=None
            Labels of model filters.
        index_base : int, str, default=-1
            Index of an explanation to be a base for comparisons.
        max_variables : int, optional, default=None
            Maximal number of variables from the current object to be taken into account.
        significance_level : float, optional, default=None
            A significance level of the statistical test (metric).
        fdr_correction : bool, default=True
            Add p-value correction for false discovery rate. Note that it is used only if significance_level is not None.
        return_df : bool, default=False
            If True, the method returns a dataframe on which a plot is created.

        """

        def extract_accepted_rejected(tab):
            n = len(tab)
            res = np.array(tab)
            reject = np.sum(res <= significance_level)
            accept = n - reject
            return accept / n, reject / n

        plt.subplots(figsize=figsize)

        if model_filters is None:
            results = self.compare(variable=variables, index_base=index_base,
                                   return_raw=True, max_variables=max_variables)

            results = ModelPartsExplanationArray.__flatten(results)

            if significance_level is not None and fdr_correction:
                _, results = fdrcorrection(results, significance_level)

            if filter_labels is not None:
                if len(filter_labels) == 1:
                    lbl = filter_labels
                else:
                    raise Exception('Incorrect length of filter_labels!')
            else:
                lbl = ['All values']

            if significance_level is None:
                plt.boxplot(results, labels=lbl, patch_artist=True)
            else:
                accepted, rejected = extract_accepted_rejected(results)

                plt.bar([0.8], accepted, 0.4, label='Accepted')
                plt.bar([1.2], rejected, 0.4, label='Rejected')
                plt.xticks([1], lbl)
                plt.legend()

        else:
            results = []
            for f in model_filters:
                tmp_out = self.compare(variable=variables, index_base=index_base,
                                       return_raw=True, model_filter=f, max_variables=max_variables)
                tmp_out = ModelPartsExplanationArray.__flatten(tmp_out)
                results.append(tmp_out)

            if filter_labels is not None:
                if len(filter_labels) == len(model_filters):
                    lbl = filter_labels
                else:
                    raise Exception('Incorrect length of filter_labels!')
            else:
                lbl = model_filters

            if significance_level is not None and fdr_correction:
                res_lengths = [len(r) for r in results]
                flat_results = [item for sublist in results for item in sublist]
                _, flat_results = fdrcorrection(flat_results, significance_level)
                q = 0
                for i in range(len(results)):
                    results[i] = flat_results[q:(q + res_lengths[i])]
                    q += res_lengths[i]

            if significance_level is None:
                plt.boxplot(results, labels=lbl, patch_artist=True)
            else:
                accepted = []
                rejected = []

                for r in results:
                    acc, rej = extract_accepted_rejected(r)
                    accepted.append(acc)
                    rejected.append(rej)

                bar_width = 0.4
                x = lbl
                x_axis = np.arange(len(x))

                plt.bar(x_axis, accepted, bar_width, label='Accepted')
                plt.bar(x_axis + bar_width, rejected, bar_width, label='Rejected')
                plt.legend()

                plt.xticks(x_axis + bar_width / 2, x)

        if significance_level is None:
            plt.ylabel(r'p-values')
            plt.title(f'Summary of {self.explanation_type} for {self.name}')
            plt.xlabel('Filter')
        else:
            plt.ylabel('Fraction of tests')
            plt.title(
                f'Summary of {self.explanation_type} for {self.name} with p-value significance level {significance_level}')
            plt.xlabel('Filter')

        if return_df:
            return results

    @staticmethod
    def __flatten(lst):
        out = []
        for i in range(len(lst)):
            if not (isinstance(lst[i], list) or isinstance(lst[i], ModelPartsExplanationArray)):
                out.append(lst[i])
            else:
                tmp = ModelPartsExplanationArray.__flatten(lst[i])
                out = out + tmp
        return out

    def __str__(self) -> str:
        return f"ModelPartsExplanationArray {self.name}"

    def __repr__(self) -> str:
        return f"<ModelPartsExplanationArray {self.name}>"
