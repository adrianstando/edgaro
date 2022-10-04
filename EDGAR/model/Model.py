from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from EDGAR.base.BaseTransformer import BaseTransformer
from EDGAR.data.Dataset import Dataset


class Model(BaseTransformer, ABC):
    def __init__(self):
        super().__init__()
        self.__transform_to_probabilities = False

    @abstractmethod
    def fit(self, dataset: Dataset):
        pass

    @abstractmethod
    def predict(self, dataset: Dataset):
        pass

    @abstractmethod
    def predict_proba(self, dataset: Dataset):
        pass

    @abstractmethod
    def set_params(self, **params):
        pass

    @abstractmethod
    def get_params(self):
        pass

    def transform(self, dataset: Dataset):
        return self.__transform(dataset)

    def __transform(self, dataset: Dataset):
        if self.__transform_to_probabilities:
            return self.predict_proba(dataset)
        else:
            return self.predict(dataset)

    def set_transform_to_probabilities(self):
        self.__transform_to_probabilities = True

    def set_transform_to_classes(self):
        self.__transform_to_probabilities = False


class RandomForest(Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__model = RandomForestClassifier(*args, **kwargs)

    def fit(self, dataset: Dataset):
        if dataset.target is None:
            raise Exception('Target data is not provided!')

        self.__model.fit(dataset.data, dataset.target)

    def predict(self, dataset: Dataset):
        return Dataset(
            name=dataset.name + '_predicted',
            dataframe=None,
            target=self.__model.predict(dataset.data)
        )

    def predict_proba(self, dataset: Dataset):
        return Dataset(
            name=dataset.name + '_predicted_probabilities',
            dataframe=None,
            target=self.__model.predict_proba(dataset.data)
        )

    def set_params(self, **params):
        return self.__model.set_params(**params)

    def get_params(self):
        return self.__model.get_params()
