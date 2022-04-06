import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import gym
import gym_builderarch
import torch
import re

#self.get_examples(filename=f"{difficulty}{self.size[0]}.squares")[:n]
examples = gym.make('BuilderArch-v1').get_examples(filename=f"{sys.argv[1]}.squares")

paths = sys.argv[2:]
fig, axs = plt.subplots(len(paths))
if len(paths) <= 1:
    axs = [axs]
repeats = 8
colors = ["red", "blue", "orange", "cyan", "magenta"] * repeats

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
#plt.vlines(x, 0, y, linestyle="dashed")
for i,path in enumerate(paths):
    ax = axs[i]
    t = str(re.search('\d+$', paths[i]).group())
    ax.set_title(f"{t} abstractions")
    ax.set_xlabel('Iterations')
    ax.xaxis.set_label_coords(0.5, 0.1)
    ax.set_ylabel('#Examples cleared')
    n = 1
    unsuc = False
    steps = 0
    used_blocks = []
    #ax.set_xscale("log")
    while os.path.isfile(f'{path}/rewards{n}.csv'):
        df = pd.read_csv(f'{path}/rewards{n}.csv')

        #ax = ax.plot(df['num_episodes'], df['R_avg'])
        #df = df.groupby(np.arange(len(df)) // 50).mean()
        steps += df['num_episodes'].max()
        blocks = int(torch.sum(examples[n-1][1]))//2
        used_blocks.append(blocks)
        if os.path.isfile(f'{path}/a{n}.csv'):
            ax.scatter(steps, n, c = colors[blocks-1])
        else:
            ax.scatter(steps, n, c = 'gray')
            unsuc = True
        n += 1
    ps = [mpatches.Patch(color=c, label = f'{i+1} blocks') for i,c in enumerate(colors[:(len(colors))]) if i+1 in used_blocks]
    if unsuc:
        ax.legend(handles = [mpatches.Patch(color='gray', label = f'timed out')] + ps) 
    else:
        ax.legend(handles = ps)
plt.show()