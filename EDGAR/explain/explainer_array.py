from __future__ import annotations

from typing import Union, Optional, Literal, List

from EDGAR.model.model import Model
from EDGAR.model.model_array import ModelArray
from EDGAR.explain.explainer import Explainer
from EDGAR.explain.explainer_result import ExplainerResult


class ExplainerArray:
    def __init__(self, models: Union[Model, ModelArray], N: Optional[int] = None,
                 curve_type: Literal['PDP', 'ALE'] = 'PDP') -> None:
        self.models = models
        self.sub_calculators = None
        self.name = models.name
        self.N = N
        self.curve_type = curve_type

    def fit(self) -> None:
        def create_sub_calculator(model: Union[Model, ModelArray]):
            if isinstance(model, Model):
                calc = Explainer(model=model, N=self.N, curve_type=self.curve_type)
            else:
                calc = ExplainerArray(models=model, N=self.N, curve_type=self.curve_type)

            calc.fit()
            return calc

        self.sub_calculators = [create_sub_calculator(m) for m in self.models.get_models()]

    def transform(self, variables=None) -> List[ExplainerResult]:
        if self.sub_calculators is None:
            raise Exception('Explainer was not fitted!')
        if variables is None:
            return [calc.transform() for calc in self.sub_calculators]
        else:
            def transform_given_variables(calc):
                if isinstance(calc, Explainer):
                    var = list(set(variables).intersection(calc.model.get_train_dataset().data.columns))
                    return calc.transform(variables=var)
                else:
                    return calc.transform(variables=variables)

            return [transform_given_variables(calc) for calc in self.sub_calculators]

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
        return f"ExplainerArray with {length} calculators with {self.curve_type} curve type"

    def __repr__(self) -> str:
        length = 0
        if self.sub_calculators is not None:
            length = len(self.sub_calculators)
        return f"<ExplainerArray with {length} calculators with {self.curve_type} curve type>"
