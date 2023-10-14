from src.utils import *
from src.chunking import *
from src.models import get_LR
from src.plots import conf_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/allseqs_20191230.csv')

x = 'sequence'
y = 'is_viable'
lim = 43
alphabet = AA_alphabet
n = 1

df = df.sample(frac=1).reset_index(drop=True)
df_test = df.iloc[:len(df)//10]
df_train = df.iloc[len(df)//10:]

subsets = chunk_seed(data=df_train, frac=0.01, n=n)

print('1/5 divided training data into subsets')

models = [get_LR() for i in range(n)]

xs = [data_prep(i, x=x, y=y, lim=lim,
                alphabet=alphabet, flat=True)[0] for i in subsets]

ys = [data_prep(i, x=x, y=y, lim=lim,
                alphabet=alphabet, flat=True)[1] for i in subsets]

xt = data_prep(df_test, x=x, y=y, lim=lim,
               alphabet=alphabet, flat=True)[0]

yt = data_prep(df_test, x=x, y=y, lim=lim,
               alphabet=alphabet, flat=True)[1]

print('2/5 encoded training data')

print('3/5 training ML models')

for i, model in enumerate(models):
    model.fit(xs[i], ys[i])

print('4/5 training ML models done, these are the model accuracy metrics')

for i, model in enumerate(models):
    print(model.score(xt, yt))

predictions = np.array([model.predict(xt) for model in models])

fig = conf_plot(ground_truth=yt, predictions=predictions,
                title='AAV_random_LR')

fig.savefig(f'output/AAV_random_subset_LR.png')

print('5/5 saved the confidence plot to the output folder')
