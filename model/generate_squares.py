# NOTE: Also resets state to zeros
# NOTE: Will be centered around the easiest with n_blocks
import torch
import gym
import gym_builderarch
import copy
import numpy as np

def tensor_to_tuple(tensor):
    return tuple(tensor.tolist())

env = gym.make('BuilderArch-v1')

states = []
steps = 0
for n_blocks in range(1,10):
    max_amount = 10
    for x in range(max_amount):
        steps += 1 # Counter for the name
        env.reset(n=1, difficulty = "template")
        env.state = torch.zeros(env.size)
        while int(torch.sum(env.get_state()[1])) // 2 < n_blocks:
            action = np.random.choice([0,1,2,3],p=[0.1,0.1,0.4,0.4])
            #action = env.action_space.sample()
            env.step(torch.tensor(action))
        st = torch.roll(env.get_state()[1],env.loc,1).long()
        if tensor_to_tuple(st) in [tensor_to_tuple(state) for state in states]:
            continue
        else:
            states.append(st)
        f = open(f"generated{env.size[0]}.squares","a")
        f.write(f"{n_blocks}_{steps}--\n")
        for line in st:
            f.write(str(line.tolist()))
            f.write("\n")
        f.write("\n")
        print(st)
        ex = copy.deepcopy(st)
