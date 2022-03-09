import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

paths = sys.argv[1:]
fig, axs = plt.subplots(len(paths))
if len(paths) <= 1:
    axs = [axs]
for i,path in enumerate(paths):
    ax = axs[i]
    n = 1
    ax.set_xscale("log")
    while os.path.isfile(f'{path}/rewards{n}.csv'):
        df = pd.read_csv(f'{path}/rewards{n}.csv')
        #ax = ax.plot(df['num_episodes'], df['R_avg'])
        #df = df.groupby(np.arange(len(df)) // 50).mean()
        print(df.columns)
        ax.plot(df['num_episodes'], df['R_avg_tot'], label = n)
        n += 1
    ax.legend()
plt.show()