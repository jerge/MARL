import gym
import examples as ex
import torch
import random
import dqn_model
import gym
import gym_builderarch
from collections import namedtuple
from dqn_model import DeepQLearningModel, ExperienceReplay

def eps_greedy_policy(q_values, eps):
    if random.random() < eps:
        return random.randint(0,q_values.shape[1]-1)
    return torch.argmax(q_values)

def calc_q_and_take_action(dqn, state, eps):
    q_online_curr = dqn.online_model(state)
    action_i = eps_greedy_policy(q_online_curr, eps) # 

    return q_online_curr, torch.tensor(action_i)



env = gym.make('BuilderArch-v1')
env.reset()
ex1 = ex.get_examples7()[4]
print(ex1)
env.set_goal(ex1)
device = torch.device("cpu")
actions = env.action_space
num_actions = actions[0].n * actions[1].n
num_states = env.size

num_episodes = 3000
batch_size = 128
gamma = .94
learning_rate = 1e-4

dqn = DeepQLearningModel(device, num_states, num_actions, learning_rate)
mod = dqn.online_model.load_state_dict(torch.load("./l.asdf"))

done = False
while not done:
    state = env.state
    state = state[None,:]
    state = state[None,:]
    q_o_c, a = calc_q_and_take_action(dqn, state, 0)
    ob, reward, done, _ = env.step(a)
    print(reward)
env.render()
