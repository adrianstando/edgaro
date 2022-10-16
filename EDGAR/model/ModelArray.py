from typing import Optional, List, Dict, Any, Union
from EDGAR.base.BaseTransformerArray import BaseTransformerArray
from EDGAR.model.Model import Model
from EDGAR.data.Dataset import Dataset
from EDGAR.data.DatasetArray import DatasetArray


class ModelArray(BaseTransformerArray):
    def __init__(self, base_model: Model, parameters: Optional[List[Dict[str, Any]]] = None, name: str = ''):
        super().__init__(base_transformer=base_model, parameters=parameters)
        self.name = name

    def fit(self, dataset: Union[Dataset, DatasetArray]):
        if self.name == '':
            self.name = dataset.name
        super().fit(dataset)
        for i in range(len(self.get_transformers())):
            self.get_transformers()[i] = self.__base_transformer_array_to_model_array(self.get_transformers()[i])

    def predict(self, dataset: Union[Dataset, DatasetArray]):
        return super().transform(dataset)

    def predict_proba(self, dataset: Union[Dataset, DatasetArray]):
        for model_arr in self.get_models():
            model_arr.set_transform_to_probabilities()
        out = super().transform(dataset)
        for model_arr in self.get_models():
            model_arr.set_transform_to_classes()
        return out

    def get_models(self):
        return self.get_transformers()

    def set_transform_to_probabilities(self):
        for m in self.get_models():
            m.set_transform_to_probabilities()

    def set_transform_to_classes(self):
        for m in self.get_models():
            m.set_transform_to_classes()

    def __base_transformer_array_to_model_array(self, base):
        if isinstance(base, Model):
            return base
        elif not isinstance(base, ModelArray):
            out = ModelArray(base_model=self.get_base_transformer())
            out.__class__ = self.__class__
            for key, val in base.__dict__.items():
                out.__dict__[key] = val

            return out
        else:
            return base
