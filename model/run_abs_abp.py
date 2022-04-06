import torch
import gym
from marlabsdqn import train_loop, test_examples
import gym_builderarch
import os
import sys
from agents import Architect, Builder
from sleeping import save_catalog, import_catalog

if torch.cuda.is_available():
    print('cuda')
    device = torch.device("cuda")
else:
    print('cpu')
    device = torch.device("cpu")

# Network type, difficulty, ex_end, ex_start
#------Sys args-----
n_args = len(sys.argv)

a_network_type = "dense" if n_args <= 1 else sys.argv[1]
b_network_type = "smalldense" if n_args <= 2 else sys.argv[2]

difficulty = "normal" if n_args <= 3 else sys.argv[3]

# Create the environment
env = gym.make('BuilderArch-v1')
env.reset(difficulty = difficulty)

max_size = len(env.get_examples(filename=f"{difficulty}{env.size[0]}.squares"))
end = max_size if n_args <= 4 else int(sys.argv[4])
ex_end = min(end, max_size)

ex_start = 0 if n_args <= 5 else int(sys.argv[5]) # Amount of examples deemed correct

suffix = "" if n_args <= 6 or sys.argv[6] == 'no' else sys.argv[6]

supervise = False if n_args <= 7 else bool(sys.argv[7])

testing = False if n_args <= 8 else bool(sys.argv[8])

#/-----Sys args----\

# Initializations
# actions = env.action_space
# num_actions = actions.n
grouped_actions = env.action_space
num_actions = grouped_actions[0].n * grouped_actions[1].n
num_states = env.size

num_episodes = 100000

max_catalog_size = 3
name=f"{env.size[0]}{a_network_type[:3]}{b_network_type[:3]}{difficulty[:3]}{max_catalog_size}"
# num_states, num_actions, num_channels, device, network_type, catalog = [], max_catalog_size = 0, learning_rate = 0.9
architect   = Architect(num_states,                    num_actions, 2, device, a_network_type, training = True, max_catalog_size = max_catalog_size, grouped = True)
builder     = Builder(int(architect.dqn._num_actions), num_actions, 1, device, b_network_type, training = False, max_catalog_size = max_catalog_size, grouped = True)

path = f"./model_checkpoints/{name}"
if not os.path.exists(path):
    os.makedirs(path)

# a_dqn.online_model.load_state_dict(torch.load( f"{path}/a{ex_start}_interrupted.saved"))
# a_dqn.offline_model.load_state_dict(torch.load(f"{path}/a{ex_start}_interrupted.saved"))
if not os.path.isfile(f'{path}/a{ex_start}{suffix}.saved'):
    torch.save(architect.dqn.online_model.state_dict(), f"{path}/a{ex_start}{suffix}.saved")
if not os.path.isfile(f'{path}/b{ex_start}{suffix}.saved'):
    torch.save(builder.dqn.online_model.state_dict(),   f"{path}/b{ex_start}{suffix}.saved")

if ex_start != 0 and os.path.isfile(f"{path}/a{ex_start}.replay") and os.path.isfile(f"{path}/b{ex_start}.replay"):
    architect.replay_buffer.load(f"{path}/a{ex_start}.replay")
    builder.replay_buffer.load(f"{path}/b{ex_start}.replay")

for i in range(ex_start, ex_end):
    # Object holding our online / offline Q-Networks
    architect.dqn.online_model.load_state_dict( torch.load(f"{path}/a{i}{suffix}.saved"))
    architect.dqn.offline_model.load_state_dict(torch.load(f"{path}/a{i}{suffix}.saved"))
    import_catalog(f"{path}/a{i}{suffix}", architect)

    builder.dqn.online_model.load_state_dict(   torch.load(f"{path}/b{i}{suffix}.saved"))
    builder.dqn.offline_model.load_state_dict(  torch.load(f"{path}/b{i}{suffix}.saved"))
    import_catalog(f"{path}/b{i}{suffix}", builder)

    if supervise:
        supervise_location = f"./gym_builderarch/envs/supervised_play/{difficulty}{env.size[0]}.replay"
        n_loaded = architect.replay_buffer.load(supervise_location)
        print(f'Loaded {n_loaded} supervised transitions to the architect')

    if testing:
        #n_examples, architect, builder, env, device, difficulty="normal", pretty_test=False
        test_examples(ex_end, architect, builder, env, device, difficulty=difficulty, pretty_test=True)
        break
    # Train
    try:
        R_avg, timed_out = train_loop(env, architect, builder, num_episodes,
                                            device, i+1, difficulty, df_path = path, n_plot_examples = ex_end)
        if timed_out:
            torch.save(architect.dqn.online_model.state_dict(), f"{path}/a{i+1}_unsuc.saved")
            torch.save(builder.dqn.online_model.state_dict(), f"{path}/b{i+1}_unsuc.saved")
            save_catalog(f"{path}/a{i+1}_unsuc", architect.catalog)
            save_catalog(f"{path}/b{i+1}_unsuc", builder.catalog)
        else:
            torch.save(architect.dqn.online_model.state_dict(), f"{path}/a{i+1}.saved")
            torch.save(builder.dqn.online_model.state_dict(), f"{path}/b{i+1}.saved")
            save_catalog(f"{path}/a{i+1}", architect.catalog)
            save_catalog(f"{path}/b{i+1}", builder.catalog)
            suffix = ""
    except KeyboardInterrupt:
        torch.save(architect.dqn.online_model.state_dict(), f"{path}/a{i}_interrupted.saved")
        torch.save(builder.dqn.online_model.state_dict(), f"{path}/b{i}_interrupted.saved")
        save_catalog(f"{path}/a{i}_interrupted", architect.catalog)
        save_catalog(f"{path}/b{i}_interrupted", builder.catalog)
        break

#store(self, path, amount, replace = True):
architect.replay_buffer.store(f"{path}/a{i}.replay", architect.replay_buffer.buffer_length)
builder.replay_buffer.store(f"{path}/b{i}.replay"  , builder.replay_buffer.buffer_length)