import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

paths = sys.argv[1:]
fig, axs = plt.subplots(len(paths))
if len(paths) <= 1:
    axs = [axs]
steps = 0
for i,path in enumerate(paths):
    ax = axs[i]
    n = 1
    #ax.set_xscale("log")
    while os.path.isfile(f'{path}/rewards{n}.csv'):
        df = pd.read_csv(f'{path}/rewards{n}.csv')
        #ax = ax.plot(df['num_episodes'], df['R_avg'])
        #df = df.groupby(np.arange(len(df)) // 50).mean()
        steps += df['num_episodes'].max()
        ax.scatter(steps, n, label = n)
        n += 1
    ax.legend()
plt.show()