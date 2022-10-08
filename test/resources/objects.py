import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import os

df_1 = pd.DataFrame({
    'a': [1, 2, 3],
    'b': ['a', 'b', 'c']
})
target_1 = pd.Series([0, 0, 1])
target_1_fake = pd.Series([2, 0, 1])
name_1 = 'name_1'

df_2 = pd.DataFrame({
    'a': [2.1, 2.2, 2.3],
    'b': [2, 3, 1]
})
target_2 = pd.Series([1, 0, 1])
name_2 = 'name_2'

df_3, target_3 = load_breast_cancer(return_X_y=True, as_frame=True)
name_3 = 'breast_cancer'

df_4_nans = pd.DataFrame({
    'a': [1, 2, np.NaN],
    'b': ['a', 'b', 'c']
})
target_4_nans = pd.Series([np.NaN, 0, 1])
name_4_nans = 'name_4'

example_path = os.path.join('resources', 'data_1.csv')
example_target = 'target'

task_id_1 = 3
task_id_2 = 15
