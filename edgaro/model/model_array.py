from __future__ import annotations

import pandas as pd

from typing import Optional, List, Dict, Any, Union

from edgaro.base.base_transformer_array import BaseTransformerArray
from edgaro.model.model import Model
from edgaro.data.dataset import Dataset
from edgaro.data.dataset_array import DatasetArray
from edgaro.base.utils import print_unbuffered


class ModelArray(BaseTransformerArray):
    def __init__(self, base_model: Model, parameters: Optional[List[Dict[str, Any]]] = None,
                 name: str = '', verbose: bool = False) -> None:
        super().__init__(base_transformer=base_model, parameters=parameters)
        self.name = name
        self.verbose = verbose

    def fit(self, dataset: Union[Dataset, DatasetArray]) -> None:
        if self.name == '':
            self.name = dataset.name

        if self.verbose:
            print_unbuffered(f'ModelArray {self.__repr__()} is being fitted with {dataset.name}')

        super().fit(dataset)
        self.__fix_classes(dataset)

        if self.verbose:
            print_unbuffered(f'ModelArray {self.__repr__()} was fitted with {dataset.name}')

    def predict(self, dataset: Union[Dataset, DatasetArray]) -> Union[Dataset, DatasetArray]:
        out = super().transform(dataset)

        if self.verbose:
            print_unbuffered(f'ModelArray {self.__repr__()} predicted on {dataset.name}')

        return out

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

        if self.verbose:
            print_unbuffered(f'ModelArray {self.__repr__()} predicted probabilities on {dataset.name}')

        return out

    def get_models(self) -> List[Union[Model, ModelArray, List[Union[Model, ModelArray]]]]:
        return self.transformers

    def set_transform_to_probabilities(self) -> None:
        ModelArray.__set_to_probs(self.get_models())

    def set_transform_to_classes(self) -> None:
        ModelArray.set_to_class(self.get_models())

    def __fix_classes(self, dataset: Union[Dataset, DatasetArray]) -> None:
        for i in range(len(self.transformers)):
            self.transformers[i] = self.__base_transformer_array_to_model_array(
                self.transformers[i],
                dataset[i].name if isinstance(dataset, DatasetArray) else dataset.name
            )
            if isinstance(self.transformers[i], BaseTransformerArray):
                self.transformers[i].__fix_classes(dataset if isinstance(dataset, Dataset) else dataset[i])

    def __base_transformer_array_to_model_array(self, base: Union[Model, ModelArray, BaseTransformerArray],
                                                name: str) -> Union[Model, ModelArray]:
        if isinstance(base, Model):
            return base
        elif not isinstance(base, ModelArray) and isinstance(base, BaseTransformerArray):
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
            eval_model['model'] = m.name
            return pd.concat([out_in, eval_model])

        def _eval_all(mod, ds_, out_in):
            out_out = None
            if isinstance(mod, Model):
                out_out = _evaluate(mod, ds_, out_in)
            elif isinstance(mod, ModelArray):
                tmp = []
                for j in range(len(mod.get_models())):
                    m = mod.get_models()[j]
                    if isinstance(ds_, DatasetArray):
                        tmp.append(_eval_all(m, ds_[j], out_out))
                    else:
                        tmp.append(_eval_all(m, ds_, out_in))
                tmp.append(out_out)
                out_out = pd.concat(tmp)
            elif isinstance(mod, list):
                tmp = []
                for j in range(len(mod)):
                    if isinstance(ds_, DatasetArray):
                        tmp.append(_eval_all(mod[j], ds_[j], out_out))
                    else:
                        tmp.append(_eval_all(mod[j], ds_, out_out))
                tmp.append(out_out)
                out_out = pd.concat(tmp)
            return out_out

        if self.verbose:
            print_unbuffered(f'ModelArray {self.__repr__()} is being evaluated')

        out = pd.DataFrame({'model': [], 'metric': [], 'value': []})
        out = _eval_all(self.get_models(), ds, out)
        out = out[['model', 'metric', 'value']]
        out = out.reset_index(drop=True)

        if self.verbose:
            print_unbuffered(f'ModelArray {self.__repr__()} was evaluated')

        return out

    @property
    def transformers(self) -> List[Union[Model, ModelArray, List]]:
        return super().transformers

    @transformers.setter
    def transformers(self, val: List[Union[Model, ModelArray, List]]) -> None:
        super().transformers = val

    @property
    def base_transformer(self) -> Model:
        out = super().base_transformer
        if isinstance(out, Model):
            return out
        else:
            raise Exception('Wrong base_transformer attribute')

    @base_transformer.setter
    def base_transformer(self, val: Model):
        super().transformers = val

    def __str__(self) -> str:
        return f"ModelArray {self.name} with {len(self.get_models())} models"

    def __repr__(self) -> str:
        return f"<ModelArray {self.name} with {len(self.get_models())} models>"
