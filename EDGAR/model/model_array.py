from __future__ import annotations

import pandas as pd

from typing import Optional, List, Dict, Any, Union

from EDGAR.base.base_transformer_array import BaseTransformerArray
from EDGAR.model.model import Model
from EDGAR.data.dataset import Dataset
from EDGAR.data.dataset_array import DatasetArray


class ModelArray(BaseTransformerArray):
    def __init__(self, base_model: Model, parameters: Optional[List[Dict[str, Any]]] = None, name: str = '') -> None:
        super().__init__(base_transformer=base_model, parameters=parameters)
        self.name = name

    def fit(self, dataset: Union[Dataset, DatasetArray]) -> None:
        if self.name == '':
            self.name = dataset.name
        super().fit(dataset)
        for i in range(len(self.transformers)):
            self.transformers[i] = self.__base_transformer_array_to_model_array(
                self.transformers[i],
                dataset[i].name if isinstance(dataset, DatasetArray) else dataset.name
            )

    def predict(self, dataset: Union[Dataset, DatasetArray]) -> Union[Dataset, DatasetArray]:
        return super().transform(dataset)

    @staticmethod
    def __set_to_probs(inp: Union[List, Model, ModelArray]):
        if isinstance(inp, list):
            for i in inp:
                ModelArray.__set_to_probs(i)
        else:
            inp.set_transform_to_probabilities()

    @staticmethod
    def set_to_class(inp: Union[List, Model, ModelArray]):
        if isinstance(inp, list):
            for i in inp:
                ModelArray.__set_to_probs(i)
        else:
            inp.set_transform_to_classes()

    def predict_proba(self, dataset: Union[Dataset, DatasetArray]) -> Union[Dataset, DatasetArray]:
        ModelArray.__set_to_probs(self.get_models())
        out = super().transform(dataset)
        ModelArray.set_to_class(self.get_models())
        return out

    def get_models(self) -> List[Union[Model, ModelArray, List[Union[Model, ModelArray]]]]:
        return self.transformers

    def set_transform_to_probabilities(self) -> None:
        ModelArray.__set_to_probs(self.get_models())

    def set_transform_to_classes(self) -> None:
        ModelArray.set_to_class(self.get_models())

    def __base_transformer_array_to_model_array(self, base: Union[Model, ModelArray, BaseTransformerArray, List[Any]],
                                                name: str) -> Union[Model, ModelArray]:
        if isinstance(base, Model):
            return base
        elif not isinstance(base, ModelArray):
            out = ModelArray(base_model=self.base_transformer)
            out.__class__ = self.__class__
            for key, val in base.__dict__.items():
                out.__dict__[key] = val
            out.name = name

            return out
        else:
            return base

    def evaluate(self, metrics_output_class=None, metrics_output_probabilities=None,
                 ds: Optional[DatasetArray] = None) -> pd.DataFrame:

        def _evaluate(mod, ds_, out_in):
            m = mod
            data = ds_ if isinstance(ds_, Dataset) else None
            eval_model = m.evaluate(metrics_output_class=metrics_output_class,
                                    metrics_output_probabilities=metrics_output_probabilities, ds=data)
            if 'model' not in eval_model.columns:
                eval_model['model'] = m.name
            return pd.concat([out_in, eval_model])

        def _eval_all(mod, ds_, out_in):
            if isinstance(mod, Model):
                out_out = _evaluate(mod, ds_, out_in)
            elif isinstance(mod, ModelArray):
                out_out = out_in
                for j in range(len(mod.get_models())):
                    m = mod.get_models()[j]
                    if isinstance(ds_, DatasetArray):
                        out_out = _eval_all(m, ds_[j], out_out)
                    else:
                        out_out = _eval_all(m, ds_, out_in)
            elif isinstance(mod, list):
                out_out = out_in
                for j in range(len(mod)):
                    if isinstance(ds_, DatasetArray):
                        out_out = _eval_all(mod[j], ds_[j], out_out)
                    else:
                        out_out = _eval_all(mod[j], ds_, out_out)
            return out_out

        out = pd.DataFrame({'model': [], 'metric': [], 'value': []})
        out = _eval_all(self.get_models(), ds, out)
        return out

    @property
    def transformers(self) -> List[Union[Model, ModelArray, List[Any]]]:
        return super().transformers

    @transformers.setter
    def transformers(self, val: List[Union[Model, ModelArray, List[Any]]]) -> None:
        super().transformers = val

    @property
    def base_transformer(self) -> Model:
        return super().base_transformer

    @base_transformer.setter
    def base_transformer(self, val: Model):
        super().transformers = val

    def __str__(self) -> str:
        return f"ModelArray {self.name} with {len(self.get_models())} models"

    def __repr__(self) -> str:
        return f"<ModelArray {self.name} with {len(self.get_models())} models>"
