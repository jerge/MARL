import torch
import gym
from marlabsdqn import train_loop
import gym_builderarch
import os
import sys
from agents import Architect, Builder

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

num_episodes = 300000000

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

name=f"{env.size[0]}absmarl{difficulty}"
# num_states, num_actions, num_channels, device, network_type, catalog = [], max_catalog_size = 0, learning_rate = 0.9
architect   = Architect(num_states,               num_actions, 2, device, network_type)
builder     = Builder(architect.dqn._num_actions, num_actions, 1, device, "small" + network_type, training = True)

# a_dqn.online_model.load_state_dict(torch.load( f"./model_checkpoints/{name}a{ex_start}_interrupted.saved"))
# a_dqn.offline_model.load_state_dict(torch.load(f"./model_checkpoints/{name}a{ex_start}_interrupted.saved"))
if not os.path.isfile(f'./model_checkpoints/{name}a{ex_start}.saved'):
    torch.save(architect.dqn.online_model.state_dict(), f"./model_checkpoints/{name}a{ex_start}.saved")
if not os.path.isfile(f'./model_checkpoints/{name}b{ex_start}.saved'):
    torch.save(builder.dqn.online_model.state_dict(),   f"./model_checkpoints/{name}b{ex_start}.saved")

for i in range(ex_start, ex_end):
    # Object holding our online / offline Q-Networks
    architect.dqn.online_model.load_state_dict( torch.load(f"./model_checkpoints/{name}a{i}.saved"))
    architect.dqn.offline_model.load_state_dict(torch.load(f"./model_checkpoints/{name}a{i}.saved"))

    builder.dqn.online_model.load_state_dict(   torch.load(f"./model_checkpoints/{name}b{i}.saved"))
    builder.dqn.offline_model.load_state_dict(  torch.load(f"./model_checkpoints/{name}b{i}.saved"))
    # Train
    try:
        R_avg, timed_out = train_loop(env, architect, builder, num_episodes,
                                            device, i+1, difficulty)
        if timed_out:
            torch.save(architect.dqn.online_model.state_dict(), f"./model_checkpoints/{name}a{i+1}_unsuc.saved")
            torch.save(builder.dqn.online_model.state_dict(), f"./model_checkpoints/{name}b{i+1}_unsuc.saved")
        else:
            torch.save(architect.dqn.online_model.state_dict(), f"./model_checkpoints/{name}a{i+1}.saved")
            torch.save(builder.dqn.online_model.state_dict(), f"./model_checkpoints/{name}b{i+1}.saved")
    except KeyboardInterrupt:
        torch.save(architect.dqn.online_model.state_dict(), f"./model_checkpoints/{name}a{i}_interrupted.saved")
        torch.save(builder.dqn.online_model.state_dict(), f"./model_checkpoints/{name}b{i}_interrupted.saved")
        break
    