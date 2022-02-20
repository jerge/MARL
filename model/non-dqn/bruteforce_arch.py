import gym
import gym_builderarch
import itertools
import torch
#import multiprocessing
from joblib import Parallel, delayed

import csv
import pandas

import sys


def all_goals(start = 0, stop = 123456789, thrs = 4, catalog = []):
    env = gym.make('BuilderArch-v1')
    env.reset()
    gns = env.get_all_examples()[start:stop]

    names = [gn[0] for gn in gns]
    goals = [gn[1] for gn in gns]
    envs = [gym.make('BuilderArch-v1') for g in goals]

    l = []
    outs = Parallel(n_jobs=thrs)(delayed(find_acts)(envs[i], goals[i], catalog) for i in range(len(goals)))
    for i in range(len(outs)):
        write_to(names[i], outs[i])
        l.append((names[i], outs[i]))
    return l

def find_acts(env, goal, catalog = []):
    
    max_length = 7

    for i in range(max_length+1):
        print(f"Length: {i}")
        # Initialise
        actions = list(range(env.action_space.n)) + catalog
        for action_combination in itertools.product(actions,repeat=i):
            env.reset()
            env.set_goal(goal)
            for action in action_combination:
                if action in catalog:
                    for a in action.split(","):
                        env.take_action(a)
                else:
                    env.take_action(action)
            if torch.all(goal.eq(env.state)):
                return action_combination
    return ()



def write_to(name, actions):
    row = [name, actions]
    with open('bruteforce.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


thrs = int(sys.argv[1])
start = int(sys.argv[2])
stop = int(sys.argv[3])
def get_catalog():
    lines = open('catalog.csv','r').readlines()
    return [line.strip() for line in lines]
print(all_goals(start, stop, thrs,get_catalog()))
