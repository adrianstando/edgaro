from typing import Union
from EDGAR.model.Model import Model
from EDGAR.model.ModelArray import ModelArray
from EDGAR.explain.PDPCalculator import PDPCalculator


# TODO:
# deal with variables argument

class PDPCalculatorArray:
    def __init__(self, models: Union[Model, ModelArray]):
        if isinstance(models, Model):
            models = ModelArray
        self.models = models
        self.sub_calculators = None

    def fit(self):
        def create_sub_calculator(model: Model):
            calc = PDPCalculator(model)
            calc.fit()
            return calc

        self.sub_calculators = [create_sub_calculator(m) for m in self.models.get_models()]

    def transform(self, variables=None):
        if variables is None:
            return [calc.transform() for calc in self.sub_calculators]
        else:
            return self.transform(variables=None)
