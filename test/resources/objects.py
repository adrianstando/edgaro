import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
import os

df_1 = pd.DataFrame({
    'a': [1.0, 2.0, 3.0],
    'b': ['a', 'b', 'c']
})
df_1_x = pd.DataFrame({
    'c': [1.0, 2.0, 3.0],
    'd': ['a', 'b', 'c']
})
df_1_categorical = pd.DataFrame({
    'a': [1, 2, 2],
    'b': ['a', 'b', 'c']
})
target_1 = pd.Series([0, 0, 1])
target_1_fake = pd.Series([2, 0, 1])
name_1 = 'name_1'
IR_1 = 2/1

df_2 = pd.DataFrame({
    'a': [2.1, 2.2, 2.3],
    'b': [2.0, 3.0, 1.0]
})
target_2 = pd.Series([1, 0, 1])
target_2_fake = pd.Series([2, 10, 1])
name_2 = 'name_2'
IR_2 = 2/1

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

task_id_1 = 29
task_id_2 = 15

suite_name_1 = 'OpenML100'
suite_name_2 = 'OpenML-CC18'

APIKEY = os.environ.get('OPENML_CREDENTIALS') if os.environ.get('OPENML_CREDENTIALS') is not None else ''
