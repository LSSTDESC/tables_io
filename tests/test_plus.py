import pandas as pd
import numpy as np

from tables_io import tablePlus

d = {}
d['col1'] = np.array([1,2,3])
d['col2'] = np.array([4,5,6])
df = pd.DataFrame(d)

dfPlus = tablePlus.TablePlus(df, 'dfTable')
print(dir(dfPlus))

print('An element: ', dfPlus['col1'][0])
print('\nA column:\n', dfPlus['col1'])
