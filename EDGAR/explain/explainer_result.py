from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Union, Tuple, List, Optional, Literal
from numpy import ndarray
from matplotlib.axes import Axes


class Curve:
    def __init__(self, x: ndarray, y: ndarray) -> None:
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"Curve with {len(self.x)} points\nx:\n" + str(self.x) + "\ny:\n" + str(self.y)

    def __repr__(self) -> str:
        return f"<Curve with {len(self.x)} points>"


class ExplainerResult:
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

    @staticmethod
    def __plot_not_add(results, categorical_columns, curve_type, name, variable, ax, figsize, show_legend, y_lim) -> None:
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
    def __plot_add_categorical(name, variable, ax, figsize, curve_base, curves_add, add_plot_names) -> None:
        df = pd.DataFrame({
            'x': curves_add[0].x,
            name: curve_base.y
        })
        for i in range(len(curves_add)):
            df[add_plot_names[i]] = curves_add[i].y

        if ax is not None:
            df.plot(x='x', kind='bar', ax=ax, legend=False, xlabel=variable)
        else:
            df.plot(x='x', kind='bar', figsize=figsize, legend=False, xlabel=variable)

        plt.xticks(rotation=0)

    def __plot_add(self, variable, ax, figsize, curve_base, curves_add, add_plot, show_legend, y_lim) -> None:
        curves_add = [c for c in curves_add if c is not None]
        add_plot_names = [c.name for c in add_plot if c.results[variable] is not None]

        p = 1
        for ind in range(len(add_plot_names)):
            if add_plot_names[ind] == self.name:
                add_plot_names[ind] += ('_' + str(p))
                p += 1

        if variable not in self.categorical_columns:
            self.__plot_add_continuous(ax, figsize, curve_base, curves_add)

        else:
            ExplainerResult.__plot_add_categorical(self.name, variable, ax, figsize,
                                                   curve_base, curves_add, add_plot_names)

        if self.curve_type == 'PDP':
            plt.title("PDP curve for variable: " + variable)
        elif self.curve_type == 'ALE':
            plt.title("ALE curve for variable: " + variable)
        else:
            raise Exception('Wrong curve type!')

        if show_legend:
            plt.legend([self.name] + add_plot_names)
        if y_lim is not None:
            plt.ylim(y_lim)
        plt.xlabel(variable)

    def plot(self, variable: str, figsize: Optional[Tuple[int, int]] = (8, 8),
             add_plot: Optional[List[ExplainerResult]] = None, ax: Optional[Axes] = None,
             show_legend: bool = True, y_lim: Optional[Tuple[float, float]] = None
             ) -> None:
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
                self.__plot_add(variable, ax, figsize, curve_base, curves_add, add_plot, show_legend, y_lim)

    def compare(self, other: ExplainerResult, variable: Optional[Union[str, List[str]]] = None) -> np.ndarray:
        if isinstance(variable, str):
            if self[variable] is None:
                raise Exception('Variable is not available!')

            if other[variable] is None:
                raise Exception('Variable in \'other\' is not available!')

            y_this = self.results[variable].y
            y_other = other.results[variable].y
            return np.sum(np.abs(y_this - y_other))
        else:
            if variable is None:
                variable_all = list(self.results.keys())
                return np.mean(
                    [self.compare(variable=var, other=other) for var in variable_all]
                )
            else:
                return np.mean(
                    [self.compare(variable=var, other=other) for var in variable]
                )

    def __str__(self) -> str:
        return f"ExplainerResult {self.name} for {len(self.results.keys())} variables: {list(self.results.keys())} with {self.curve_type} curve type"

    def __repr__(self) -> str:
        return f"<ExplainerResult {self.name} with {self.curve_type} curve type>"
