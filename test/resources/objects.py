import pandas as pd
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

example_path = os.path.join('resources', 'data_1.csv')
example_target = 'target'
