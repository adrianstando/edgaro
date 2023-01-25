from __future__ import annotations

import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, List, Optional, Literal
from numpy import ndarray
from matplotlib.axes import Axes
from scipy.stats import wilcoxon


class Curve:
    """
    The class which represents the PDP/ALE curve for one variable.

    Parameters
    ----------
    x : np.ndarray
        Points on 0X axis.
    y : np.ndarray
        Points on 0Y axis.

    Attributes
    ----------
    x : np.ndarray
        Points on 0X axis.
    y : np.ndarray
        Points on 0Y axis.
    """

    def __init__(self, x: Optional[ndarray, pd.Series], y: Optional[ndarray, pd.Series]) -> None:
        self.x = x if not isinstance(x, pd.Series) else x.reset_index(drop=True)
        self.y = y if not isinstance(y, pd.Series) else y.reset_index(drop=True)

    def __str__(self) -> str:
        return f"Curve with {len(self.x)} points\nx:\n" + str(self.x) + "\ny:\n" + str(self.y)

    def __repr__(self) -> str:
        return f"<Curve with {len(self.x)} points>"


class Explanation(ABC):
    def __int__(self) -> None:
        pass

    @abstractmethod
    def plot(self) -> None:
        pass

    @abstractmethod
    def compare(self, other: List[Explanation]) -> List[Union[float, list]]:
        pass


class ModelProfileExplanation(Explanation):
    """
    The class which represent the PDP/ALE curves for all variables in one Model.

    Parameters
    ----------
    results : Dict[str, Curve]
        A dictionary of pairs (column name, Curve object), which represents curves for all variables in one Model.
    name : str
        The name of ModelProfileExplanation. It is best if it is a Model name.
    categorical_columns : list[str]
        List of categorical variables.
    explanation_type : {'PDP', 'ALE'}, default='PDP'
        A curve type.

    Attributes
    ----------
    results : Dict[str, Curve]
        A dictionary of pairs (column name, Curve object), which represents curves for all variables in one Model.
    name : str
        The name of ModelProfileExplanation. It is best if it is a Model name.
    categorical_columns : list[str]
        List of categorical variables.
    explanation_type : {'PDP', 'ALE'}
        A curve type.

    """

    def __init__(self, results: Dict[str, Curve], name: str, categorical_columns: List[str],
                 explanation_type: Literal['PDP', 'ALE'] = 'PDP') -> None:
        self.results = results
        self.name = name
        self.categorical_columns = categorical_columns
        self.explanation_type = explanation_type

    def __getitem__(self, key: str) -> Optional[Curve]:
        if key in self.results.keys():
            return self.results[key]
        else:
            return None

    def plot(self, variable: Optional[str] = None, figsize: Optional[Tuple[int, int]] = (8, 8),
             add_plot: Optional[List[ModelProfileExplanation]] = None, ax: Optional[Axes] = None,
             show_legend: bool = True, y_lim: Optional[Tuple[float, float]] = None, metric_precision: int = 2,
             centered: bool = False) -> None:
        """
        The function plots the PDP/ALE curve.

        Parameters
        ----------
        variable : str, optional, default=None
            Variable for which the plot should be generated. If None, the first column is plotted.
        figsize : tuple(int, int), optional, default=(8, 8)
            Size of a figure.
        add_plot : list[ModelProfileExplanation], optional, default=None
            List of other ModelProfileExplanation objects that also contain the `variable` and should be plotted.
        ax : matplotlib.axes.Axes, optional, default=None
            The parameter should be passed if the plot is to be created in a certain Axis. In that situation, `figsize`
            parameter is ignored.
        show_legend : bool, default=True
            The parameter indicates whether the legend should be plotted.
        y_lim : tuple(float, float), optional, default=None
            The limits of 0Y axis.
        metric_precision : int, default=5
            Number of digits to round the value of `metric value*10^5`.
        centered : bool, default = False
            If True, the plots will be centered to start at 0.

        """
        if figsize is None and ax is None:
            figsize = (8, 8)

        if variable is None:
            variable = list(self.results.keys())[0]

        if add_plot is None:
            ModelProfileExplanation.__plot_not_add(self.results, self.categorical_columns, self.explanation_type, self.name,
                                                   variable, ax, figsize, show_legend, y_lim, centered)
        else:
            curve_base = self.results[variable]
            if curve_base is None:
                raise Exception('Variable is not available!')

            curves_add = [pdp.results[variable] for pdp in add_plot]
            if None in curves_add:
                warnings.warn(f'There is not variable {variable} in one of the added plots!')

            if len(curves_add) == 0:
                warnings.warn(f'None of the added plots have variable called {variable}!')
                ModelProfileExplanation.__plot_not_add(self.results, self.categorical_columns, self.explanation_type,
                                                       self.name, variable, ax, figsize, show_legend, y_lim, centered)
            else:
                self.__plot_add(variable, ax, figsize, curve_base, curves_add, add_plot, show_legend,
                                y_lim, metric_precision, centered)
        plt.ylabel(self.explanation_type)

    @staticmethod
    def __plot_not_add(results, categorical_columns, explanation_type, name, variable, ax, figsize, show_legend,
                       y_lim, centered) -> None:
        curve = results[variable]
        if curve is None:
            raise Exception('Variable is not available!')

        if ax is not None:
            plt.sca(ax)
        elif figsize is not None:
            plt.subplots(figsize=figsize)

        x, y = ModelProfileExplanation.__get_x_y(curve, centered)

        if variable not in categorical_columns:
            plt.plot(x, y)
        else:
            plt.bar(x, y)

        if explanation_type == 'PDP':
            plt.title("PDP curve for variable: " + variable)
        elif explanation_type == 'ALE':
            plt.title("ALE curve for variable: " + variable)
        else:
            raise Exception('Wrong curve type!')

        plt.xlabel(variable)
        if show_legend:
            plt.legend([name])
        if y_lim is not None:
            plt.ylim(y_lim)

    @staticmethod
    def __get_x_y(curve, centered):
        x = curve.x
        y = copy.deepcopy(curve.y)
        if centered:
            y = y - y[0]
        return x, y

    @staticmethod
    def __plot_add_continuous(ax, figsize, curve_base, curves_add, centered) -> None:
        if ax is not None:
            plt.sca(ax)
        elif figsize is not None:
            plt.subplots(figsize=figsize)

        x, y = ModelProfileExplanation.__get_x_y(curve_base, centered)
        plt.plot(x, y)
        for curve in curves_add:
            x, y = ModelProfileExplanation.__get_x_y(curve, centered)
            plt.plot(x, y)

    @staticmethod
    def __plot_add_categorical(name, variable, ax, figsize, curve_base, curves_add, add_plot_names, title, centered) -> None:
        _, y = ModelProfileExplanation.__get_x_y(curve_base, centered)

        df = pd.DataFrame({
            'x': curves_add[0].x,
            name: y
        })

        for i in range(len(curves_add)):
            _, y = ModelProfileExplanation.__get_x_y(curves_add[i], centered)
            df[add_plot_names[i]] = y

        if ax is not None:
            df.plot(x='x', kind='bar', ax=ax, legend=False, xlabel=variable, title=title)
        else:
            df.plot(x='x', kind='bar', figsize=figsize, legend=False, xlabel=variable, title=title)

        plt.xticks(rotation=0)

    def __plot_add(self, variable, ax, figsize, curve_base, curves_add, add_plot, show_legend,
                   y_lim, metric_precision, centered) -> None:
        curves_add = [c for c in curves_add if c is not None]
        add_plot_names = [c.name for c in add_plot if c.results[variable] is not None]

        p = 1
        for ind in range(len(add_plot_names)):
            if add_plot_names[ind] == self.name:
                add_plot_names[ind] += ('_' + str(p))
                p += 1

        if variable not in self.categorical_columns:
            ModelProfileExplanation.__plot_add_continuous(ax, figsize, curve_base, curves_add, centered)
            plt.title(f"{self.explanation_type} curve for variable: " + variable)
        else:
            ModelProfileExplanation.__plot_add_categorical(self.name, variable, ax, figsize,
                                                           curve_base, curves_add, add_plot_names,
                                                           f"{self.explanation_type} curve for variable: " + variable,
                                                           centered)

        if show_legend:
            plt.legend([self.name] + add_plot_names)
        if y_lim is not None:
            plt.ylim(y_lim)
        plt.xlabel(variable)

        compare_results = self.compare(add_plot, variable=variable)
        if not np.alltrue([r is None for r in compare_results]):
            if ax is None:
                ax = plt.gca()
            y_min, y_max = ax.get_ylim()
            x_min, x_max = ax.get_xlim()

            text = ""
            if compare_results[0] is not None:
                if len(add_plot) == 1:
                    text += 'SDD[$10^{-3}$]=' + f'{round(compare_results[0] * 10 ** 3, metric_precision)}'
                else:
                    text += 'ASDD[$10^{-3}$]=' + f'{round(compare_results[0] * 10 ** 3, metric_precision)}'

            if text != "":
                ax.text(
                    x_min + 0.7 * (x_max - x_min),
                    y_min + 0.2 * (y_max - y_min),
                    text,
                    fontsize='medium'
                )

    @staticmethod
    def __retrieve_explainer_results(inp, explain_results_in):
        if isinstance(inp, ModelProfileExplanation):
            explain_results_in.append(inp)
        else:
            if isinstance(inp, list):
                for inp_i in inp:
                    ModelProfileExplanation.__retrieve_explainer_results(inp_i, explain_results_in)
            else:
                for inp_i in inp.results:
                    ModelProfileExplanation.__retrieve_explainer_results(inp_i, explain_results_in)

    def compare(self, other: List[ModelProfileExplanation], variable: Optional[Union[str, List[str]]] = None,
                return_raw_per_variable: bool = False) -> List[Union[float, list]]:
        """
        The function calculates the metric to compare the curves for a given variable(s).

        Currently, there is only one comparison metric called `SDD` (Standard Deviation of Distances). It is the variance of
        the distances between curve in this object and curves in `other` in intermediate points.
        If there is more than one `other` object and `return_raw_per_variable=False`, the mean variance
        is returned (`ASDD` - Averaged SDD).

        Parameters
        ----------
        other : list[ModelProfileExplanation]
            List of ModelProfileExplanation objects to compare the curve against.
        variable : str, list[str], optional, default=None
            List of variable names to calculate the metric distances. If None, the metrics are calculated for
            all the columns in this object.
        return_raw_per_variable : bool, default=False
            If True, raw values for each variable are returned.

        Returns
        ----------
        list[float, list]

        """

        result_tab = []
        if isinstance(variable, str):
            if self[variable] is None:
                raise Exception('Variable is not available!')

            explain_results = []
            ModelProfileExplanation.__retrieve_explainer_results(other, explain_results)

            if np.all([o[variable] is None for o in explain_results]):
                raise Exception('Variable in \'other\' is not available!')

            result_tab = self.__calculate_metrics_one_variable(variable, explain_results)
        else:
            var = list(self.results.keys()) if variable is None else variable
            outs = [self.compare(variable=v, other=other) for v in var]

            if not return_raw_per_variable:
                for i in range(len(outs[0])):
                    result_tab.append(float(np.mean([
                        o[i] for o in outs
                    ])))
            else:
                result_tab = [tmp[0] for tmp in outs]

        return result_tab

    def __calculate_metrics_one_variable(self, variable: str, explain_results: List[ModelProfileExplanation]):

        def dist(res):
            y1 = res[variable].y
            y1 = y1.reset_index(drop=True) if isinstance(y1, pd.Series) else y1

            y2 = self[variable].y
            y2 = y2.reset_index(drop=True) if isinstance(y2, pd.Series) else y2
            return y1 - y2

        result_tab = []
        distances_to_original = [
            dist(res)
            for res in explain_results
        ]

        # fligner test
        # if len(explain_results) < 2:
        #    result_tab.append(None)
        # else:
        #    _, out = fligner(*distances_to_original)
        #    result_tab.append(out)

        # variance
        variances = [np.std(deltas) for deltas in distances_to_original]
        result_tab.append(np.mean(variances))

        # mean of abs differences
        # abs_deltas = [np.mean(np.abs(deltas)) for deltas in distances_to_original]
        # result_tab.append(np.mean(abs_deltas))

        return result_tab

    def __str__(self) -> str:
        return f"ModelProfileExplanation {self.name} for {len(self.results.keys())} variables: {list(self.results.keys())} with {self.explanation_type} curve type"

    def __repr__(self) -> str:
        return f"<ModelProfileExplanation {self.name} with {self.explanation_type} curve type>"


class ModelPartsExplanation(Explanation):
    """
    The class represents the Variable Importance for all variables in one Model.

    Parameters
    ----------
    results : Dict[str, float]
        A dictionary of pairs (column name, value), which represents Variable Importance for all variables in one Model.
    name : str
        The name of ModelProfileExplanation. It is best if it is a Model name.
    explanation_type : {'VI'}, default='VI'
        An explanation type.

    Attributes
    ----------
    results : Dict[str, float]
        A dictionary of pairs (column name, value), which represents Variable Importance for all variables in one Model.
    name : str
        The name of ModelProfileExplanation. It is best if it is a Model name.
    explanation_type : {'VI'}, default='VI'
        An explanation type.
    """

    def __init__(self, results: Dict[str, float], name: str, explanation_type: Literal['VI'] = 'VI') -> None:
        self.results = results
        self.name = name
        self.explanation_type = explanation_type

    def __getitem__(self, key: str) -> Optional[float]:
        if key in self.results.keys():
            return self.results[key]
        else:
            return None

    @staticmethod
    def __extract_res(obj, variable, max_variables):
        res = pd.DataFrame.from_dict(obj.results, orient='index').reset_index()
        res.columns = ['colname', 'val']
        res['colname'] = res['colname'].astype("string")
        res = res.sort_values('val')

        if variable is not None:
            res = res[res['colname'].isin(variable)]
            if res.shape[0] != len(variable):
                raise Exception('Wrong variable names were provided!')

        if max_variables is not None:
            r = min(max_variables, len(variable)) if variable is not None else max_variables
            r = min(r, res.shape[0])
            res = res.iloc[list(range(r))]

        return res

    def plot(self, variable: Optional[Union[str, List[str]]] = None,
             add_plot: Optional[List[ModelPartsExplanation]] = None, max_variables: Optional[int] = None,
             figsize: Optional[Tuple[int, int]] = (8, 8), ax: Optional[Axes] = None, show_legend: bool = False,
             x_lim: Optional[Tuple[float, float]] = None, metric_precision: int = 3) -> None:
        """
        The function plots the Variable Importance profile.

        Parameters
        ----------
        variable : str, list[str], optional, default=None
            Variable for which the VI should be plotted. If None, the all columns is plotted.
        figsize : tuple(int, int), optional, default=(8, 8)
            Size of a figure.
        add_plot : list[ModelPartsExplanation], optional, default=None
            List of other ModelPartsExplanation objects that also contain the `variable` and should be plotted.
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

        """

        res = ModelPartsExplanation.__extract_res(self, variable, max_variables)
        column_order = res['colname'].to_list()

        if isinstance(variable, str):
            variable = [variable]

        if ax is not None:
            plt.sca(ax)
        elif figsize is not None:
            plt.subplots(figsize=figsize)

        if add_plot is not None:
            def extraction(a):
                res_other = ModelPartsExplanation.__extract_res(a, variable, None)
                res_other['colname'] = res_other['colname'].astype("string")
                res_other = res_other.set_index('colname')
                res_other = res_other.loc[column_order]
                res_other = res_other.reset_index()
                return res_other

            group_height = 0.8
            y = res['colname'].astype("string")
            bar_height = group_height / (len(add_plot) + 1)
            y_axis = np.arange(len(y))

            plt.barh(y_axis, res['val'], bar_height, label=self.name)

            for i in range(len(add_plot)):
                p = extraction(add_plot[i])
                plt.barh(y_axis + (i + 1) * bar_height, p['val'], bar_height, label=add_plot[i].name)

            plt.yticks(y_axis + group_height / 2, y)
        else:
            plt.barh(res['colname'], res['val'])

        plt.title(f"Variable importance for {self.name}")

        plt.xlabel("Variable Importance")
        plt.ylabel("Variable")
        if show_legend:
            if add_plot is None:
                plt.legend([self.name])
            else:
                plt.legend()
        if x_lim is not None:
            plt.xlim(x_lim)

        if add_plot is not None:
            compare_results = self.compare(add_plot, variable=variable, max_variables=max_variables, return_raw=False)
            if not np.alltrue([r is None for r in compare_results]):
                if ax is None:
                    ax = plt.gca()
                y_min, y_max = ax.get_ylim()
                x_min, x_max = ax.get_xlim()

                text = ""
                if compare_results[0] is not None:
                    if len(add_plot) == 1:
                        text += 'p-value=' + f'{round(compare_results[0], metric_precision)}'
                    else:
                        text += 'MEAN(p-value)=' + f'{round(compare_results[0], metric_precision)}'

                if text != "":
                    ax.text(
                        x_min + 0.7 * (x_max - x_min),
                        y_min + 0.2 * (y_max - y_min),
                        text,
                        fontsize='medium'
                    )

    def compare(self, other: List[ModelPartsExplanation], variable: Optional[Union[str, List[str]]] = None,
                max_variables: Optional[int] = None, return_raw: bool = True) -> List[Union[float, list]]:
        """
        The function calculates the metric to compare model parts of two or more models.

        Currently, there is only one metric based on the Wilcoxon statistical test [3]_. The metric value is the
        p-value of this test, where the inputs are variable importance values.
        The idea is based on the approach presented in the article [4]_.

        Parameters
        ----------
        other : list[ModelPartsExplanation]
            List of ModelPartsExplanation objects to compare the curve against.
        variable : str, list[str], optional, default=None
            List of variable names to calculate the metric distances. If None, the metrics are calculated for
            all the columns in this object.
        max_variables : int, optional, default=None
            Maximal number of variables from the current object to be taken into account.
        return_raw : bool, default=True
            If True, the p-values are returned for each model. Otherwise, the mean value is returned.

        Returns
        -------
        list[float, list]

        References
        ----------
        .. [3] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
        .. [4] https://www.sciencedirect.com/science/article/pii/S2666764922000145

        """

        out = []

        res = ModelPartsExplanation.__extract_res(self, variable, max_variables)

        column_order = res['colname'].to_list()
        for obj in other:
            res_other = pd.DataFrame.from_dict(obj.results, orient='index').reset_index()
            res_other.columns = ['colname', 'val']
            res_other['colname'] = res_other['colname'].astype("string")
            res_other = res_other.set_index('colname')
            res_other = res_other.loc[column_order]
            res_other = res_other.reset_index()

            stat_test = wilcoxon(res['val'].tolist(), res_other['val'].to_list())
            p_value = stat_test[1]
            out.append(p_value)

        if not return_raw:
            return [float(np.mean(out))]
        else:
            return out

    def __str__(self) -> str:
        return f"ModelPartsExplanation {self.name} for {len(self.results.keys())} variables: {list(self.results.keys())} with {self.explanation_type} curve type"

    def __repr__(self) -> str:
        return f"<ModelPartsExplanation {self.name} with {self.explanation_type} curve type>"
