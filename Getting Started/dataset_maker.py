import pandas as pd
import numpy as np


def random_string(length):
    letters = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'
    res = ''.join((letters[np.random.randint(0, 26)] for i in range(length)))
    return res

def make_toy_dataset(l=200, null_prop=0.05):

    df = pd.DataFrame()
    t = []
    p = []
    q = []

    mean, std = np.random.randint(-10, 10, size=(2, ))

    for _ in range(l):
        x = np.random.randint(1, 4)
        p.append(x)
        if x == 1:
            t.append('r')
        elif x == 2:
            t.append('g')
        else:
            t.append('p')
        q.append(random_string(np.random.randint(7, 14)))

    p = np.array(p)
    df['x0'] = q
    df['x1'] = np.random.randn(l, )
    df['x2'] = np.random.random(size=(l, )) * 10
    df['x3'] = std * np.random.randn(l, ) + mean
    df['x4'] = t
    z = (df['x2'] * df['x3'] / p) 

    t = np.mean(z) + 10
    for i in range(len(z)):
        if z[i] > t:
            z[i] = 1
        else:
            z[i] = 0

    df['y'] = z

    mean, std = np.random.randint(-10, 10, size=(2, ))
    df['x1'] = df['x1'] * std + mean

    null = int(null_prop * l)
    null1 = np.random.randint(1, 5, size=(null, ))
    null2 = np.random.randint(0, l, size=(null, ))

    for a, b in zip(null1, null2):
        if a == 4:
            df.loc[b, ('x'+str(a))] = ''
        else:
            df.loc[b, ('x'+str(a))] = np.nan
    
    return df

data = make_toy_dataset()
data.to_csv('dataset.csv', index=False)
