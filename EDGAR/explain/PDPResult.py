from __future__ import annotations

from typing import Dict, Union, Tuple, List, Optional
from numpy import ndarray
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Curve:
    def __init__(self, x: ndarray, y: ndarray):
        self.x = x
        self.y = y


class PDPResult:
    def __init__(self, results: Dict[str, Curve], name: str, categorical_columns: List[str]):
        self.results = results
        self.name = name
        self.categorical_columns = categorical_columns

    def __getitem__(self, key: Union[str]):
        if key in self.results.keys():
            return self.results[key]
        else:
            return None

    def plot(self, variable: str, figsize: Tuple[int, int] = (8, 8), add_plot: Optional[List[PDPResult]] = None):
        if add_plot is None:
            curve = self.results[variable]
            if curve is None:
                raise Exception('Variable is not available!')
            plt.subplots(figsize=figsize)

            if variable not in self.categorical_columns:
                plt.plot(curve.x, curve.y)
            else:
                plt.bar(curve.x, curve.y)

            plt.title("PDP curve for variable: " + variable)
            plt.legend([self.name])
            plt.xlabel(variable)
            plt.ylim([0, 1])
        else:
            curve_base = self.results[variable]
            if curve_base is None:
                raise Exception('Variable is not available!')

            curves_add = [pdp.results[variable] for pdp in add_plot]
            if None in curves_add:
                warnings.warn(f'There is not variable {variable} in one of the added plots!')

            curves_add = [c for c in curves_add if c is not None]
            add_plot_names = [c.name for c in add_plot if c.results[variable] is not None]

            if len(curves_add) == 0:
                warnings.warn(f'None of the added plots have variable called {variable}!')
                plt.subplots(figsize=figsize)
                plt.title("PDP curve for variable: " + variable)
                if variable not in self.categorical_columns:
                    plt.plot(curve_base.x, curve_base.y)
                else:
                    plt.bar(curve_base.x, curve_base.y)

                plt.legend([self.name])
            else:
                if variable not in self.categorical_columns:
                    for curve in curves_add:
                        plt.plot(curve.x, curve.y)
                else:
                    df = pd.DataFrame({
                        'x': curves_add[0].x,
                        self.name: curve_base.y
                    })
                    for i in range(len(add_plot)):
                        df[add_plot_names[i]] = curves_add[i].y
                    df.plot(x='x', kind='bar', figsize=figsize)
                    plt.xticks(rotation=0)

                plt.title("PDP curves for variable: " + variable)
                plt.legend([self.name] + add_plot_names)
            plt.xlabel(variable)
            plt.ylim([0, 1])

    def compare(self, other: PDPResult, variable: Optional[Union[str, List[str]]] = None):
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
