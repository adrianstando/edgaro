from typing import Union, Optional, Literal
from EDGAR.model.Model import Model
from EDGAR.model.ModelArray import ModelArray
from EDGAR.explain.PDPCalculator import PDPCalculator


class PDPCalculatorArray:
    def __init__(self, models: Union[Model, ModelArray], N: Optional[int] = None,
                 curve_type: Literal['PDP', 'ALE'] = 'PDP'):
        if isinstance(models, Model):
            models = ModelArray
        self.models = models
        self.sub_calculators = None
        self.name = models.name
        self.N = N
        self.curve_type = curve_type

    def fit(self):
        def create_sub_calculator(model: Union[Model, ModelArray]):
            if isinstance(model, Model):
                calc = PDPCalculator(model=model, N=self.N, curve_type=self.curve_type)
            else:
                calc = PDPCalculatorArray(models=model, N=self.N, curve_type=self.curve_type)

            calc.fit()
            return calc

        self.sub_calculators = [create_sub_calculator(m) for m in self.models.get_models()]

    def transform(self, variables=None):
        if variables is None:
            return [calc.transform() for calc in self.sub_calculators]
        else:
            def transform_given_variables(calc):
                if isinstance(calc, PDPCalculator):
                    var = list(set(variables).intersection(calc.model.get_train_dataset().data.columns))
                    return calc.transform(variables=var)
                else:
                    return calc.transform(variables=variables)

            return [transform_given_variables(calc) for calc in self.sub_calculators]

    def __iter__(self):
        self.current_i = 0
        return self

    def __next__(self):
        if self.current_i < len(self.sub_calculators):
            out = self.sub_calculators[self.current_i]
            self.current_i += 1
            return out
        else:
            raise StopIteration

    def __str__(self):
        return f"PDPCalculatorArray with {len(self.sub_calculators) if self.sub_calculators is not None else 0} calculators with {self.curve_type} curve type"

    def __repr__(self):
        return f"<PDPCalculatorArray with {len(self.sub_calculators) if self.sub_calculators is not None else 0} calculators with {self.curve_type} curve type>"
