import torch
import dqn_model
import gym
from marldqn import train_loop
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

# TODO: gör det skillnad om man randomizar ordning på katalog? Typ så den inte overfittar på de 4 första
catalog = [] #catalog = [[2,2],[3,3]]

# Initializations
actions = env.action_space
num_actions = actions.n
num_messages = num_actions + len(catalog)
num_states = env.size
num_channels = 2

num_episodes = 3000000
batch_size = 128
gamma = .95
learning_rate = 1e-4

# Network type, difficulty, ex_end, ex_start
#------Sys args-----
n_args = len(sys.argv)
network_type = "dense" if n_args <= 1 else sys.argv[1]
assert network_type == "dense", "Not dense network has not been implemented, note the state size of the dqn model"

difficulty = "normal" if n_args <= 2 else sys.argv[2]

max_size = len(env.get_examples(filename=f"{difficulty}{env.size[0]}.squares"))
end = max_size if n_args <= 3 else int(sys.argv[3])
ex_end = min(end, max_size)

ex_start = 0 if n_args <= 4 else int(sys.argv[4]) # Amount of examples deemed correct
#------Sys args-----

name=f"{env.size[0]}marl{difficulty}"
a_dqn = DeepQLearningModel(device, num_states[0] * num_states[1], num_messages, num_channels, learning_rate, network_type)
b_dqn = DeepQLearningModel(device, num_messages, num_actions, 1, learning_rate, network_type)

# a_dqn.online_model.load_state_dict(torch.load( f"./model_checkpoints/{name}a{ex_start}_interrupted.saved"))
# a_dqn.offline_model.load_state_dict(torch.load(f"./model_checkpoints/{name}a{ex_start}_interrupted.saved"))
if not os.path.isfile(f'./model_checkpoints/{name}a{ex_start}.saved'):
    torch.save(a_dqn.online_model.state_dict(),  f"./model_checkpoints/{name}a{ex_start}.saved")
if not os.path.isfile(f'./model_checkpoints/{name}b{ex_start}.saved'):
    torch.save(b_dqn.online_model.state_dict(),  f"./model_checkpoints/{name}b{ex_start}.saved")

a_replay_buffer = ExperienceReplay(device, num_states, input_channels=2)
b_replay_buffer = ExperienceReplay(device, num_messages, input_channels=1)
for i in range(ex_start, ex_end):
    # Object holding our online / offline Q-Networks
    a_dqn.online_model.load_state_dict( torch.load(f"./model_checkpoints/{name}a{i}.saved"))
    a_dqn.offline_model.load_state_dict(torch.load(f"./model_checkpoints/{name}a{i}.saved"))
    b_dqn.online_model.load_state_dict( torch.load(f"./model_checkpoints/{name}b{i}.saved"))
    b_dqn.offline_model.load_state_dict(torch.load(f"./model_checkpoints/{name}b{i}.saved"))
    # Train
    try:
        R, R_avg, timed_out = train_loop(env, a_dqn, b_dqn, num_episodes, a_replay_buffer, b_replay_buffer,
                                            device, i+1, difficulty,
                                            training_architect = True, training_builder = False, batch_size=batch_size)
        if timed_out:
            torch.save(a_dqn.online_model.state_dict(), f"./model_checkpoints/{name}a{i+1}_unsuc.saved")
            torch.save(b_dqn.online_model.state_dict(), f"./model_checkpoints/{name}b{i+1}_unsuc.saved")
        else:
            torch.save(a_dqn.online_model.state_dict(), f"./model_checkpoints/{name}a{i+1}.saved")
            torch.save(b_dqn.online_model.state_dict(), f"./model_checkpoints/{name}b{i+1}.saved")
    except KeyboardInterrupt:
        torch.save(a_dqn.online_model.state_dict(), f"./model_checkpoints/{name}a{i}_interrupted.saved")
        torch.save(b_dqn.online_model.state_dict(), f"./model_checkpoints/{name}b{i}_interrupted.saved")
        break
    