from abc import ABC, abstractmethod
from typing import Dict, Any

from edgaro.data.dataset import Dataset


class BaseTransformer(ABC):
    """
    The abstract class to define transformations for a single Dataset.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, dataset: Dataset) -> None:
        """
        Fit the transformer.

        Parameters
        ----------
        dataset : Dataset
            The object to fit BaseTransformer on.
        """
        pass

    @abstractmethod
    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the object.

        Parameters
        ----------
        dataset : Dataset
            The object to be transformed.

        Returns
        -------
        Dataset
            The transformed object.
        """
        pass

    @abstractmethod
    def set_params(self, **params) -> None:
        """
        Set params for BaseTransformer.

        Parameters
        ----------
        params : dict
            The parameters to be set.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get parameters of BaseTransformer.

        Returns
        -------
        Dict[str, Any]
            The parameters.
        """
        pass

    @property
    @abstractmethod
    def was_fitted(self) -> bool:
        """
        The information whether the BaseTransformer was fitted.

        Returns
        -------
        bool
        """
        pass

    def __str__(self) -> str:
        return f"BaseTransformer {self.__class__.__name__}"

    def __repr__(self) -> str:
        return f"<BaseTransformer {self.__class__.__name__}>"
