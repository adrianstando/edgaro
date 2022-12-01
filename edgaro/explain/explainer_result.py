from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Union, Tuple, List, Optional, Literal
from numpy import ndarray
from matplotlib.axes import Axes
from scipy.stats import fligner


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

    def __init__(self, x: ndarray, y: ndarray) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"Curve with {len(self.x)} points\nx:\n" + str(self.x) + "\ny:\n" + str(self.y)

    def __repr__(self) -> str:
        return f"<Curve with {len(self.x)} points>"


class ExplainerResult:
    """
    The class which represent the PDP/ALE curves for all variables in one Model.

    Parameters
    ----------
    results : Dict[str, Curve]
        A dictionary of pairs (column name, Curve object), which represents curves for all variables in one Model.
    name : str
        The name of ExplainerResult. It is best if it is a Model name.
    categorical_columns : list[str]
        List of categorical variables.
    curve_type : {'PDP', 'ALE'}, default='PDP'
        A curve type.

    Attributes
    ----------
    results : Dict[str, Curve]
        A dictionary of pairs (column name, Curve object), which represents curves for all variables in one Model.
    name : str
        The name of ExplainerResult. It is best if it is a Model name.
    categorical_columns : list[str]
        List of categorical variables.
    curve_type : {'PDP', 'ALE'}
        A curve type.

    """

    def __init__(self, results: Dict[str, Curve], name: str, categorical_columns: List[str],
                 curve_type: Literal['PDP', 'ALE'] = 'PDP') -> None:
        self.results = results
        self.name = name
        self.categorical_columns = categorical_columns
        self.curve_type = curve_type

    def __getitem__(self, key: str) -> Optional[Curve]:
        if key in self.results.keys():
            return self.results[key]
        else:
            return None

    def plot(self, variable: str, figsize: Optional[Tuple[int, int]] = (8, 8),
             add_plot: Optional[List[ExplainerResult]] = None, ax: Optional[Axes] = None,
             show_legend: bool = True, y_lim: Optional[Tuple[float, float]] = None, metric_precision: int = 5
             ) -> None:
        """
        The function plots the PDP/ALE curve.

        Parameters
        ----------
        variable : str
            Variable for which the plot should be generated.
        figsize : tuple(int, int), optional, default=(8, 8)
            Size of a figure.
        add_plot : list[ExplainerResult], optional, default=None
            List of other ExplainerResult objects that also contain the `variable` and should be plotted.
        ax : matplotlib.axes.Axes, optional, default=None
            The parameter should be passed if the plot is to be created in a certain Axis. In that situation, `figsize`
            parameter is ignored.
        show_legend : bool, default=True
            The parameter indicates whether the legend should be plotted.
        y_lim : tuple(float, float), optional, default=None
            The limits of 0Y axis.
        metric_precision : int, default=5
            Number of digits to round the value of metrics.

        """
        if figsize is None and ax is None:
            figsize = (8, 8)

        if add_plot is None:
            ExplainerResult.__plot_not_add(self.results, self.categorical_columns, self.curve_type, self.name,
                                           variable, ax, figsize, show_legend, y_lim)
        else:
            curve_base = self.results[variable]
            if curve_base is None:
                raise Exception('Variable is not available!')

            curves_add = [pdp.results[variable] for pdp in add_plot]
            if None in curves_add:
                warnings.warn(f'There is not variable {variable} in one of the added plots!')

            if len(curves_add) == 0:
                warnings.warn(f'None of the added plots have variable called {variable}!')
                ExplainerResult.__plot_not_add(self.results, self.categorical_columns, self.curve_type, self.name,
                                               variable, ax, figsize, show_legend, y_lim)
            else:
                self.__plot_add(variable, ax, figsize, curve_base, curves_add, add_plot, show_legend,
                                y_lim, metric_precision)

    @staticmethod
    def __plot_not_add(results, categorical_columns, curve_type, name, variable, ax, figsize, show_legend,
                       y_lim) -> None:
        curve = results[variable]
        if curve is None:
            raise Exception('Variable is not available!')

        if ax is not None:
            plt.sca(ax)
        elif figsize is not None:
            plt.subplots(figsize=figsize)

        if variable not in categorical_columns:
            plt.plot(curve.x, curve.y)
        else:
            plt.bar(curve.x, curve.y)

        if curve_type == 'PDP':
            plt.title("PDP curve for variable: " + variable)
        elif curve_type == 'ALE':
            plt.title("ALE curve for variable: " + variable)
        else:
            raise Exception('Wrong curve type!')

        plt.xlabel(variable)
        if show_legend:
            plt.legend([name])
        if y_lim is not None:
            plt.ylim(y_lim)

    @staticmethod
    def __plot_add_continuous(ax, figsize, curve_base, curves_add) -> None:
        if ax is not None:
            plt.sca(ax)
        elif figsize is not None:
            plt.subplots(figsize=figsize)

        plt.plot(curve_base.x, curve_base.y)
        for curve in curves_add:
            plt.plot(curve.x, curve.y)

    @staticmethod
    def __plot_add_categorical(name, variable, ax, figsize, curve_base, curves_add, add_plot_names, title) -> None:
        df = pd.DataFrame({
            'x': curves_add[0].x,
            name: curve_base.y
        })
        for i in range(len(curves_add)):
            df[add_plot_names[i]] = curves_add[i].y

        if ax is not None:
            df.plot(x='x', kind='bar', ax=ax, legend=False, xlabel=variable, title=title)
        else:
            df.plot(x='x', kind='bar', figsize=figsize, legend=False, xlabel=variable, title=title)

        plt.xticks(rotation=0)

    def __plot_add(self, variable, ax, figsize, curve_base, curves_add, add_plot, show_legend,
                   y_lim, metric_precision) -> None:
        curves_add = [c for c in curves_add if c is not None]
        add_plot_names = [c.name for c in add_plot if c.results[variable] is not None]

        p = 1
        for ind in range(len(add_plot_names)):
            if add_plot_names[ind] == self.name:
                add_plot_names[ind] += ('_' + str(p))
                p += 1

        if variable not in self.categorical_columns:
            ExplainerResult.__plot_add_continuous(ax, figsize, curve_base, curves_add)
            plt.title(f"{self.curve_type} curve for variable: " + variable)
        else:
            ExplainerResult.__plot_add_categorical(self.name, variable, ax, figsize,
                                                   curve_base, curves_add, add_plot_names,
                                                   f"{self.curve_type} curve for variable: " + variable)

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
                text += f'p-value={round(compare_results[0], metric_precision)}'
            if compare_results[1] is not None:
                if text != "":
                    text += '\n'
                text += f'mean_variance={round(compare_results[1], metric_precision)}'
            if compare_results[2] is not None:
                if text != "":
                    text += '\n'
                text += f'mean_abs={round(compare_results[2], metric_precision)}'

            if text != "":
                ax.text(
                    x_min + 0.7 * (x_max - x_min),
                    y_min + 0.2 * (y_max - y_min),
                    text,
                    fontsize='medium'
                )

    @staticmethod
    def __retrieve_explainer_results(inp, explain_results_in):
        if isinstance(inp, ExplainerResult):
            explain_results_in.append(inp)
        else:
            if isinstance(inp, list):
                for inp_i in inp:
                    ExplainerResult.__retrieve_explainer_results(inp_i, explain_results_in)
            else:
                for inp_i in inp.results:
                    ExplainerResult.__retrieve_explainer_results(inp_i, explain_results_in)

    def compare(self, other: List[ExplainerResult], variable: Optional[Union[str, List[str]]] = None) \
            -> List[Optional[float]]:
        """
        The function calculates the metrics to compare the curves for a given variable.

        There are three metrics:
            1. The p-value of the Fligner-Killeen [3]_ test for the distances between the curve in this object and
               curves in `other`. To calculate this metric, the list `other` must have at least two elements.
            2. The variance of the distances between curve in this object and curves in `other` in intermediate points.
               If there is more than one `other` object, the mean variance is returned.
            3. The mean of absolute values of distances between curve in this object and curves in `other` in
               intermediate points. If there is more than one `other` object, the mean value of all calculated values
               is returned.

        Parameters
        ----------
        other : list[ExplainerResult]
            List of ExplainerResult objects to compare the curve against.
        variable : str, list[str], optional, default=None
            List of variable names to calculate the metric distances. If None, the metrics are calculated for
            all the columns in this object.

        Returns
        ----------
        list[float, None]

        References
        ----------
        .. [3] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fligner.html

        """

        result_tab = []
        if isinstance(variable, str):
            if self[variable] is None:
                raise Exception('Variable is not available!')

            explain_results = []
            ExplainerResult.__retrieve_explainer_results(other, explain_results)

            if np.all([o[variable] is None for o in explain_results]):
                raise Exception('Variable in \'other\' is not available!')

            result_tab = self.__calculate_metrics_one_variable(variable, explain_results)
        else:
            var = list(self.results.keys()) if variable is None else variable

            outs = [self.compare(variable=var, other=other) for var in var]
            if len(outs) > 0 and isinstance(outs[0], list):
                for i in range(len(outs[0])):
                    result_tab.append(np.mean([
                        o[i] for o in outs
                    ]))
            else:
                raise Exception('Wrong output!')

        if np.alltrue([np.isscalar(val) or val is None for val in result_tab]):
            return [float(val) if val is not None else val for val in result_tab]
        else:
            raise Exception('Wrong output!')

    def __calculate_metrics_one_variable(self, variable: str, explain_results: List[ExplainerResult]):
        result_tab = []
        distances_to_original = [
            res[variable].y - self[variable].y
            for res in explain_results
        ]

        # fligner test
        if len(explain_results) < 2:
            result_tab.append(None)
        else:
            _, out = fligner(*distances_to_original)
            result_tab.append(out)

        # variance
        variances = [np.var(deltas) for deltas in distances_to_original]
        result_tab.append(np.mean(variances))

        # mean of abs differences
        abs_deltas = [np.mean(np.abs(deltas)) for deltas in distances_to_original]
        result_tab.append(np.mean(abs_deltas))

        return result_tab

    def __str__(self) -> str:
        return f"ExplainerResult {self.name} for {len(self.results.keys())} variables: {list(self.results.keys())} with {self.curve_type} curve type"

    def __repr__(self) -> str:
        return f"<ExplainerResult {self.name} with {self.curve_type} curve type>"
