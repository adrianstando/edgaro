from typing import Optional, List, Dict, Any
from EDGAR.base.BaseTransformerArray import BaseTransformerArray
from EDGAR.model.Model import Model
from EDGAR.data.Dataset import Dataset


# TODO:
# delete inheritance and implement fit
class ModelArray(BaseTransformerArray):
    def __init__(self, base_model: Model, parameters: Optional[List[Dict[str, Any]]] = None):
        super().__init__(base_transfer=base_model, parameters=parameters)

    def predict(self, dataset: Dataset):
        return super().transform(dataset)

    def predict_proba(self, dataset: Dataset):
        for model in self.__transformers:
            model.set_transform_to_probabilities()
        out = super().transform(dataset)
        for model in self.__transformers:
            model.set_transform_to_classes()
        return out

    def get_models(self):
        return self.get_transformers()
