from imblearn.base import BaseSampler
from EDGAR.data.transformators import Transformator
from EDGAR.data.dataset import Dataset


class TransformatorFromIMBLEARN(Transformator):
    """
    for example:

    from imblearn.over_sampling import RandomOverSampler
    dataset = DatasetFromOpenML(task_id=3)
    transformator = TransformatorFromIMBLEARN(dataset, RandomUnderSampler(sampling_strategy=n_minority/n_majority, random_state=42))
    transformator.fit(dataset)
    transformator.transform(dataset)
    """
    def __init__(self, transformator: BaseSampler):
        self.__transformator = transformator
        super(Transformator, self).__init__()

    def fit(self, dataset: Dataset):
        return self.__transformator.fit(dataset.data, dataset.target)

    def transform(self, dataset: Dataset) -> Dataset:
        X, y = self.__transformator.fit_resample(dataset.data, dataset.target)
        return Dataset(name=dataset.name, dataframe=X, target=y)

    def get_imblearn_transformator(self):
        return self.__transformator
