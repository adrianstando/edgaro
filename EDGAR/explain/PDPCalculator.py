import dalex as dx
from typing import List, Optional
from EDGAR.model.Model import Model
from EDGAR.explain.PDPResult import PDPResult, Curve
from EDGAR.data.Dataset import Dataset


# TODO:
# N=300 or more or chosen by user

class PDPCalculator:
    def __init__(self, model: Model):
        self.model = model
        self.explainer = None
        self.name = model.name

    def fit(self):
        def predict_func(model, data):
            return model.predict_proba(Dataset('', data, None)).target[:, 1]

        dataset = self.model.get_train_dataset()
        self.explainer = dx.Explainer(self.model, dataset.data, dataset.target, label=dataset.name, verbose=False,
                                      predict_function=predict_func)

    def transform(self, variables: Optional[List[str]] = None, N: int = 1000):
        category_colnames_base = self.model.get_category_colnames()
        dict_output = {}

        if variables is None:
            variables = self.model.get_train_dataset().data.columns

        category_colnames = list(set(variables).intersection(set(category_colnames_base)))
        if len(category_colnames) > 0:
            for col in category_colnames:
                out_category = self.explainer.model_profile(verbose=False, variables=[col],
                                                            variable_type='categorical',
                                                            N=N)
                y = out_category.result['_yhat_']
                x = out_category.result['_x_']
                dict_output[str(col)] = Curve(x, y)

        other_colnames = list(set(variables).difference(set(category_colnames_base)))
        if len(other_colnames) > 0:
            out_others = self.explainer.model_profile(verbose=False, variables=other_colnames,
                                                      variable_type='numerical',
                                                      N=N)
            variable_names = out_others.result['_vname_'].unique()
            y = out_others.result['_yhat_']
            x = out_others.result['_x_']
            length = y.size / len(variable_names)
            for i in range(len(variable_names)):
                lower = int(i * length)
                higher = int((i + 1) * length)
                dict_output[str(variable_names[i])] = Curve(x[lower:higher], y[lower:higher])

        return PDPResult(dict_output, self.name, self.model.get_category_colnames())

    def set_params(self, **params):
        if 'model' in params.keys():
            if isinstance(params['model'], Model):
                self.model = params['model']

    def get_params(self):
        return {
            'model': self.model
        }
