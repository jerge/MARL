import torch
import dqn_model
import gym
from dqn import train_loop_dqn
import gym_builderarch
from collections import namedtuple
from dqn_model import DeepQLearningModel, ExperienceReplay
import random
#import examples as ex
import numpy as np
import os
import sys


if torch.cuda.is_available():
    print('cuda')
    device = torch.device("cuda")
else:
    print('cpu')
    device = torch.device("cpu")


# Create the environment
env = gym.make('BuilderArch-v1')
env.reset()
enable_visualization = False

# Initializations
actions = env.action_space
num_actions = actions.n
num_states = env.size
num_channels = 2
input_channels = 2

num_episodes = 3000000
batch_size = 128
gamma = .95
learning_rate = 1e-4

catalog = [torch.tensor([3,0]),torch.tensor([3,1]),torch.tensor([1,1])]

# Network type, difficulty, ex_end, ex_start
n_args = len(sys.argv)
network_type = "dense" if n_args <= 1 else sys.argv[1]

difficulty = "normal" if n_args <= 2 else sys.argv[2]

max_size = len(env.get_examples(filename=f"{difficulty}{env.size[0]}.squares"))
end = max_size if n_args <= 3 else int(sys.argv[3])
ex_end = min(end, max_size)

ex_start = 0 if n_args <= 4 else int(sys.argv[4]) # Amount of examples deemed correct

name=f"{env.size[0]}{difficulty}cc"
# TODO: make it work for not dense
assert network_type == "dense", "non-dense netowrk is not implemented"
dqn = DeepQLearningModel(device, num_states[0] * num_states[1], num_actions + len(catalog), num_channels, learning_rate, network_type)

# dqn.online_model.load_state_dict(torch.load(f"./model_checkpoints/{name}{ex_start}_interrupted.saved"))
# dqn.offline_model.load_state_dict(torch.load(f"./model_checkpoints/{name}{ex_start}_interrupted.saved"))
if not os.path.isfile(f'./model_checkpoints/{name}{ex_start}.saved'):
    torch.save(dqn.online_model.state_dict(),  f"./model_checkpoints/{name}{ex_start}.saved")

replay_buffer = ExperienceReplay(device, num_states, input_channels=input_channels)
for i in range(ex_start, ex_end):
    # Object holding our online / offline Q-Networks
    dqn.online_model.load_state_dict(torch.load(f"./model_checkpoints/{name}{i}.saved"))
    dqn.offline_model.load_state_dict(torch.load(f"./model_checkpoints/{name}{i}.saved"))
    # Train
    try:
        R_avg, timed_out = train_loop_dqn(dqn, env, replay_buffer, num_episodes, device, 
                                                difficulty=difficulty,
                                                enable_visualization=enable_visualization, batch_size=batch_size, 
                                                gamma=gamma, n_examples=i+1, catalog = catalog)
        if timed_out:
            torch.save(dqn.online_model.state_dict(), f"./model_checkpoints/{name}{i+1}_unsuc.saved")
        else:
            torch.save(dqn.online_model.state_dict(), f"./model_checkpoints/{name}{i+1}.saved")
    except KeyboardInterrupt:
        torch.save(dqn.online_model.state_dict(), f"./model_checkpoints/{name}{i}_interrupted.saved")
        break
    