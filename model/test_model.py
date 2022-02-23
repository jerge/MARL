import gym
#import examples as ex
import torch
import random
import dqn_model
import gym
import gym_builderarch
from collections import namedtuple
from dqn_model import DeepQLearningModel, ExperienceReplay
from dqn import test_examples


def eps_greedy_policy(q_values, eps):
    if random.random() < eps:
        return random.randint(0,q_values.shape[1]-1)
    return torch.argmax(q_values)

def calc_q_and_take_action(dqn, state, eps):
    q_online_curr = dqn.online_model(state)
    action_i = eps_greedy_policy(q_online_curr, eps) # 

    return q_online_curr, torch.tensor(action_i)

# def test_examples(n_examples, dqn, env, difficulty="normal"):
#     eps = 0
#     for i in range(n_examples):
#         env.reset()
#         env.set_goal(env.get_examples(filename=f"{difficulty}{env.size[0]}.squares")[i][1])

#         done = False
#         while not done:
#             state = env.get_state()
#             state = state[None,:]

#             q_o_c, a = calc_q_and_take_action(dqn, state, eps)

#             ob, r, done, _ = env.step(a)
#         if not r >= 0.9:
#             print(f"Could not solve examples {i}")
#             return False
#     print(f"Solved all {n_examples} examples")
#     return True

env = gym.make('BuilderArch-v1')
# TODO: Make this cleaner using system arguments
n = 7
difficulty = "normal"
env.reset(n=n, difficulty = difficulty)

print("------GOAL------")
env.render_state(env.goal)
print("----------------")
device = torch.device("cpu")
actions = env.action_space
num_actions = actions.n

catalog = [torch.tensor([3,0]),torch.tensor([3,1]),torch.tensor([1,1])]
action_list = ['Vert','Hori','Left','Right'] + catalog

num_states = env.size
num_channels = 2
# TODO: Make this cleaner using system arguments
network_type = "dense"
eps = 0


num_episodes = batch_size = gamma = learning_rate = 1 # Unnecessary variables
assert "dense" in network_type, "current implementation only supports dense (num_states[0]*num_states[1])"

dqn = DeepQLearningModel(device, num_states[0] * num_states[1], num_actions + len(catalog), num_channels, learning_rate, network_type)
name = f"{env.size[0]}{difficulty}cc{n}"
dqn.online_model.load_state_dict(torch.load(f"./model_checkpoints/{name}_interrupted.saved"))

steps = 0
finish_episode = False
while not finish_episode:
    print("\n")
    env.render()
    state = env.get_state()
    state = state[None,:]
    #print(state)
    q_o_c, curr_action = calc_q_and_take_action(dqn, state, eps)
    if int(curr_action) >= env.action_space.n:
        curr_action_list = catalog[int(curr_action-env.action_space.n)]
        new_state, reward, finish_episode, _ = env.step(curr_action_list) # take one step in the evironment
    else:
        new_state, reward, finish_episode, _ = env.step(curr_action)
    steps += 1
    print(list(zip(action_list,[round(x,3) for x in q_o_c.tolist()[0]])))
    print(f"Action: {action_list[curr_action]}, Reward: {reward}, New loc: {env.loc}")
print(f"\n-----RESULT----- in {steps} steps with {eps*100}% randomness")
env.render()
print("----------------")
print("------GOAL------")
env.render_state(env.goal)
print("----------------")

test_examples(n,dqn,env, difficulty=difficulty, device = device, catalog = catalog)