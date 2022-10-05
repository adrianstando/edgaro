import dalex as dx
from typing import List, Optional
from EDGAR.model.Model import Model
from EDGAR.explain.PDPResult import PDPResult, Curve


# TODO:
# Categorical variables
# N=300 or more or chosen by user

class PDPCalculator:
    def __init__(self, model: Model):
        self.model = model
        self.explainer = None
        self.name = model.name

    def fit(self):
        dataset = self.model.get_train_dataset()
        self.explainer = dx.Explainer(self.model, dataset.data, dataset.target, label=dataset.name, verbose=False)

    def transform(self, variables: Optional[List[str]] = None):
        if variables is None:
            out = self.explainer.model_profile(verbose=False)
        else:
            out = self.explainer.model_profile(verbose=False, variables=variables)

        variable_names = out.result['_vname_'].unique()

        y = out.result['_yhat_']
        x = out.result['_x_']
        length = y.size / len(variable_names)

        dict_output = {}
        for i in range(len(variable_names)):
            lower = int(i * length)
            higher = int((i + 1) * length)
            dict_output[str(variable_names[i])] = Curve(x[lower:higher], y[lower:higher])

        return PDPResult(dict_output, self.name)

    def set_params(self, **params):
        if 'model' in params.keys():
            if isinstance(params['model'], Model):
                self.model = params['model']

    def get_params(self):
        return {
            'model': self.model
        }
