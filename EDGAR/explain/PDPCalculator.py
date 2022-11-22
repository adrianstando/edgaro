import dalex as dx

from typing import List, Optional, Union, Literal, Dict, Any

from EDGAR.data.Dataset import Dataset
from EDGAR.model.Model import Model
from EDGAR.explain.PDPResult import PDPResult, Curve


class PDPCalculator:
    def __init__(self, model: Model, N: Optional[int] = None, curve_type: Literal['PDP', 'ALE'] = 'PDP') -> None:
        self.model = model
        self.explainer = None
        self.name = model.name
        self.N = N
        self.curve_type = curve_type

    def fit(self) -> None:
        def predict_func(model, data):
            return model.predict_proba(Dataset('', data, None)).target

        dataset = self.model.get_test_dataset()

        if dataset is None:
            raise Exception('Error with dataset!')

        if dataset.target is None:
            raise Exception('Target data is not provided!')
        if dataset.data is None:
            raise Exception('Data in dataset is not provided!')

        self.explainer = dx.Explainer(self.model, dataset.data, dataset.target, label=dataset.name, verbose=False,
                                      predict_function=predict_func)

    def transform(self, variables: Optional[List[str]] = None) -> PDPResult:
        if self.explainer is None:
            raise Exception('Explainer was not created!')
        category_colnames_base = self.model.get_category_colnames()
        dict_output = {}

        if variables is None:
            variables = list(self.model.get_train_dataset().data.columns)

        if self.curve_type == 'PDP':
            curve_type = 'partial'
        elif self.curve_type == 'ALE':
            curve_type = 'accumulated'
        else:
            raise Exception('Wrong curve type!')

        category_colnames = list(set(variables).intersection(set(category_colnames_base)))
        if len(category_colnames) > 0:
            for col in category_colnames:
                out_category = self.explainer.model_profile(verbose=False, variables=[col],
                                                            variable_type='categorical',
                                                            N=self.N, type=curve_type)
                y = out_category.result['_yhat_']
                x = out_category.result['_x_']
                dict_output[str(col)] = Curve(x, y)

        other_colnames = list(set(variables).difference(set(category_colnames_base)))
        if len(other_colnames) > 0:
            out_others = self.explainer.model_profile(verbose=False, variables=other_colnames,
                                                      variable_type='numerical',
                                                      N=self.N, type=curve_type)
            variable_names = out_others.result['_vname_'].unique()
            y = out_others.result['_yhat_']
            x = out_others.result['_x_']
            length = y.size / len(variable_names)
            for i in range(len(variable_names)):
                lower = int(i * length)
                higher = int((i + 1) * length)
                dict_output[str(variable_names[i])] = Curve(x[lower:higher], y[lower:higher])

        return PDPResult(dict_output, self.name, self.model.get_category_colnames())

    def set_params(self, **params) -> None:
        if 'model' in params.keys():
            if isinstance(params['model'], Model):
                self.model = params['model']

    def get_params(self) -> Dict[str, Any]:
        return {
            'model': self.model
        }

    def __str__(self) -> str:
        return f"PDPCalculator for model {self.name} with {self.curve_type} curve type"

    def __repr__(self) -> str:
        return f"<PDPCalculator for model {self.name} with {self.curve_type} curve type>"
