from __future__ import annotations

from typing import List, Literal, Optional, Union

from EDGAR.explain.explainer_result import ExplainerResult


class ExplainerResultArray:
    def __init__(self, results: List[Union[ExplainerResult, ExplainerResultArray]], name: str, curve_type: Literal['PDP', 'ALE'] = 'PDP') -> None:
        self.results = results
        self.name = name
        self.curve_type = curve_type

    def __getitem__(self, key: Union[Union[str, int], List[Union[str, int]]]) \
            -> Optional[Union[ExplainerResult, ExplainerResultArray]]:
        if isinstance(key, list):
            outs = [self.__getitem__(k) for k in key]
            outs = [o for o in outs if o is not None]
            if len(outs) == 0:
                return None
            else:
                return ExplainerResultArray(results=outs, name=self.name + "_subset", curve_type=self.curve_type)
        elif isinstance(key, str):
            for result in self.results:
                if result.name == key:
                    return result
        elif isinstance(key, int):
            if key <= len(self.results):
                return self.results[key]
        return None
