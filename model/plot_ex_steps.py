import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import gym
import gym_builderarch
import torch

#self.get_examples(filename=f"{difficulty}{self.size[0]}.squares")[:n]
examples = gym.make('BuilderArch-v1').get_examples(filename=f"{sys.argv[1]}.squares")

paths = sys.argv[2:]
fig, axs = plt.subplots(len(paths))
if len(paths) <= 1:
    axs = [axs]
steps = 0
colors = ["red", "blue", "orange", "cyan", "magenta"]*8

for i,path in enumerate(paths):
    ax = axs[i]
    n = 1
    #ax.set_xscale("log")
    while os.path.isfile(f'{path}/rewards{n}.csv'):
        df = pd.read_csv(f'{path}/rewards{n}.csv')
        #ax = ax.plot(df['num_episodes'], df['R_avg'])
        #df = df.groupby(np.arange(len(df)) // 50).mean()
        steps += df['num_episodes'].max()
        ones = int(torch.sum(examples[n-1][1]))//2
        
        ax.scatter(steps, n, c = colors[ones])
        n += 1
    ax.legend()
plt.show()