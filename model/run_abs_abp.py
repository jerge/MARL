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
assert "dense" in network_type, "Not dense network has not been implemented, note the state size of the dqn model"

difficulty = "normal" if n_args <= 2 else sys.argv[2]

max_size = len(env.get_examples(filename=f"{difficulty}{env.size[0]}.squares"))
end = max_size if n_args <= 3 else int(sys.argv[3])
ex_end = min(end, max_size)

ex_start = 0 if n_args <= 4 else int(sys.argv[4]) # Amount of examples deemed correct

suffix = "" if n_args <= 5 else sys.argv[5]

testing = False if n_args <= 6 else bool(sys.argv[6])

#------Sys args-----

name=f"{env.size[0]}absmarl{difficulty}"
# num_states, num_actions, num_channels, device, network_type, catalog = [], max_catalog_size = 0, learning_rate = 0.9
max_catalog_size = 4
architect   = Architect(num_states,               num_actions, 2, device, network_type, training = True, max_catalog_size = max_catalog_size)
builder     = Builder(architect.dqn._num_actions, num_actions, 1, device, "small" + network_type, training = False, max_catalog_size = max_catalog_size)

# a_dqn.online_model.load_state_dict(torch.load( f"./model_checkpoints/{name}a{ex_start}_interrupted.saved"))
# a_dqn.offline_model.load_state_dict(torch.load(f"./model_checkpoints/{name}a{ex_start}_interrupted.saved"))
if not os.path.isfile(f'./model_checkpoints/{name}a{ex_start}{suffix}.saved'):
    torch.save(architect.dqn.online_model.state_dict(), f"./model_checkpoints/{name}a{ex_start}{suffix}.saved")
if not os.path.isfile(f'./model_checkpoints/{name}b{ex_start}{suffix}.saved'):
    torch.save(builder.dqn.online_model.state_dict(),   f"./model_checkpoints/{name}b{ex_start}{suffix}.saved")

for i in range(ex_start, ex_end):
    # Object holding our online / offline Q-Networks
    architect.dqn.online_model.load_state_dict( torch.load(f"./model_checkpoints/{name}a{i}{suffix}.saved"))
    architect.dqn.offline_model.load_state_dict(torch.load(f"./model_checkpoints/{name}a{i}{suffix}.saved"))
    import_catalog(f"./model_checkpoints/{name}a{i}{suffix}", architect)

    builder.dqn.online_model.load_state_dict(   torch.load(f"./model_checkpoints/{name}b{i}{suffix}.saved"))
    builder.dqn.offline_model.load_state_dict(  torch.load(f"./model_checkpoints/{name}b{i}{suffix}.saved"))
    import_catalog(f"./model_checkpoints/{name}b{i}{suffix}", builder)

    if testing:
        #n_examples, architect, builder, env, device, difficulty="normal", pretty_test=False
        test_examples(ex_end, architect, builder, env, device, difficulty=difficulty, pretty_test=True)
        break
    # Train
    try:
        R_avg, timed_out = train_loop(env, architect, builder, num_episodes,
                                            device, i+1, difficulty)
        if timed_out:
            torch.save(architect.dqn.online_model.state_dict(), f"./model_checkpoints/{name}a{i+1}_unsuc.saved")
            torch.save(builder.dqn.online_model.state_dict(), f"./model_checkpoints/{name}b{i+1}_unsuc.saved")
            save_catalog(f"./model_checkpoints/{name}a{i+1}_unsuc", architect.catalog)
            save_catalog(f"./model_checkpoints/{name}b{i+1}_unsuc", builder.catalog)
        else:
            torch.save(architect.dqn.online_model.state_dict(), f"./model_checkpoints/{name}a{i+1}.saved")
            torch.save(builder.dqn.online_model.state_dict(), f"./model_checkpoints/{name}b{i+1}.saved")
            save_catalog(f"./model_checkpoints/{name}a{i+1}", architect.catalog)
            save_catalog(f"./model_checkpoints/{name}b{i+1}", builder.catalog)
            suffix = ""
    except KeyboardInterrupt:
        torch.save(architect.dqn.online_model.state_dict(), f"./model_checkpoints/{name}a{i}_interrupted.saved")
        torch.save(builder.dqn.online_model.state_dict(), f"./model_checkpoints/{name}b{i}_interrupted.saved")
        save_catalog(f"./model_checkpoints/{name}a{i}_interrupted", architect.catalog)
        save_catalog(f"./model_checkpoints/{name}b{i}_interrupted", builder.catalog)
        break
    