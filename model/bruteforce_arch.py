import gym
import gym_builderarch
import itertools
import torch


def all_goals(start = 0, stop = 123456789, thrs = 4):
    env = gym.make('BuilderArch-v1')
    env.reset()
    gns = env.get_all_examples()[start:stop]

    names = [gn[0] for gn in gns]
    goals = [gn[1] for gn in gns]
    envs = [gym.make('BuilderArch-v1') for g in goals]

    l = []
    outs = Parallel(n_jobs=thrs)(delayed(find_acts)(envs[i], goals[i]) for i in range(len(goals)))
    for i in range(len(outs)):
        write_to(names[i], outs[i])
        l.append((names[i], outs[i]))
    
    # for c in range(len(goals) // thrs):
    #     vs = goals[c*thrs:(c+1)*thrs]
    #     g4 = [v[1] for v in vs]
    #     n4 = [v[0] for v in vs]

    #     outs = Parallel(n_jobs=thrs)(delayed(find_acts)(i) for i in zip(e4,g4))

    #     #pool = multiprocessing.Pool(thrs)
    #     #outs = list(zip(*pool.map(find_acts, zip(e4, g4))))
    #     for i in range(thrs):
    #         write_to(n4[i], outs[i])
    #         l.append((n4[i], outs[i]))


    # for i,(name,goal) in enumerate(goals):
    #     acts = find_acts(env, goal)
    #     print(f"{i}:{name}: {acts}")
    #     print(goal)
    #     write_to(name, acts)
    #     l.append((name,acts))
    return l

def find_acts(env, goal):
    
    max_length = 9

    for i in range(max_length+1):
        print(f"Length: {i}")
        # Initialise
        actions = range(env.action_space.n)
        for acts in itertools.product(actions,repeat=i):
            env.reset()
            env.set_goal(goal)
            for a in acts:
                env.take_action(a)
            if torch.all(goal.eq(env.state)):
                return acts
    return ()

import csv
import pandas



def write_to(name, actions):
    row = [name, actions]
    with open('bruteforce.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def get_solution(name):
    df = pd.read_csv(f)
    index = df.loc[df['0'] == name]
    return df.at['index', 1]

import multiprocessing
from joblib import Parallel, delayed

import sys
thrs = int(sys.argv[1])
start = int(sys.argv[2])
stop = int(sys.argv[3])
print(all_goals(start, stop, thrs))

#sleep

# def process(i):
#     return i * i
    
# results = Parallel(n_jobs=2)(delayed(process)(i) for i in range(10))
# print(results)