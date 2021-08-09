import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

np.random.seed(101)

df = pd.DataFrame(np.random.random((10,5)), columns=list('ABCDE'))
df.index = df.index * 10
print (df)

#added some parameters
kf = KFold(n_splits = 5, shuffle = True)
result = kf.split(df)

for train,test in result:
    print (train,test)
    print (df.iloc[train])
    print (df.iloc[test])
    print (" --- ")


#train = df.iloc[result[0]]
#test =  df.iloc[result[1]]
#
#print (train)
