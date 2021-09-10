import pandas as pd
import numpy as np

from tables_io import tablePlus

d = {}
d['col1'] = np.array([1,2,3])
d['col2'] = np.array([4.1,5.2,6.3])
df = pd.DataFrame(d)

dfPlus = tablePlus.TablePlus(df, 'dfTable')
print(dir(dfPlus))

print('An element: ', dfPlus['col1'][0])
print('\nColumn col1:\n', dfPlus['col1'])
print('\nColumn col2:\n', dfPlus['col2'])
print('\nApply max:\n', dfPlus.max())
