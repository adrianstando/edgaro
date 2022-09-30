from imblearn.base import BaseSampler
from EDGAR.data.transformators import Transformator
from EDGAR.data.dataset import Dataset


class TransformatorFromIMBLEARN(Transformator):
    """
    for example:

    from imblearn.over_sampling import RandomOverSampler
    dataset = DatasetFromOpenML(task_id=3)
    transformator = TransformatorFromIMBLEARN(dataset, RandomUnderSampler(sampling_strategy=n_minority/n_majority, random_state=42))
    """
    def __init__(self, dataset: Dataset, transformator: BaseSampler):
        self.__transformator = transformator
        super(Transformator, self).__init__(dataset=dataset)

    def fit(self, X, y):
        return self.__transformator.fit(X, y)

    def transform(self, X, y):
        return self.__transformator.fit_resample(X, y)

    def get_imblearn_transformator(self):
        return self.__transformator
